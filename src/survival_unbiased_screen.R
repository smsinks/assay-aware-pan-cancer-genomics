#!/usr/bin/env Rscript

# Comprehensive outcome-independent survival screen.  Python defines the mutation-
# frequency-qualified contexts and model-adequacy rules; this script fits the same
# Efron, study-stratified Cox parameterisations to every eligible context.

suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: survival_unbiased_screen.R <analysis.csv> <specifications.csv> <output.csv>")
}

analysis <- read.csv(args[[1L]], stringsAsFactors = FALSE, check.names = FALSE)
specifications <- read.csv(args[[2L]], stringsAsFactors = FALSE, check.names = FALSE)

empty_result <- function(spec, contrast, status) {
  data.frame(
    scope = spec$scope,
    cancer = spec$cancer,
    geneA = spec$geneA,
    geneB = spec$geneB,
    context = if (spec$scope == "cancer-specific") {
      sprintf("%s–%s (%s)", spec$geneA, spec$geneB, spec$cancer)
    } else sprintf("%s–%s (pan-cancer)", spec$geneA, spec$geneB),
    contrast = contrast,
    nPatients = spec$nPatients,
    nEvents = spec$nEvents,
    nStudies = spec$nStudies,
    nStrata = spec$nStrata,
    nDoubleNegative = spec$nDoubleNegative,
    nAOnly = spec$nAOnly,
    nBOnly = spec$nBOnly,
    nDoubleMutant = spec$nDoubleMutant,
    coefficient = NA_real_,
    standardError = NA_real_,
    hazardRatio = NA_real_,
    ciLow = NA_real_,
    ciHigh = NA_real_,
    z = NA_real_,
    p = NA_real_,
    phTestP = NA_real_,
    tiesMethod = "Efron",
    fitStatus = status,
    stringsAsFactors = FALSE
  )
}

fit_context <- function(spec) {
  call_a <- paste0("callable_", spec$geneA)
  call_b <- paste0("callable_", spec$geneB)
  mut_a <- paste0("mut_", spec$geneA)
  mut_b <- paste0("mut_", spec$geneB)
  required <- c(call_a, call_b, mut_a, mut_b)
  if (!all(required %in% names(analysis))) {
    return(rbind(
      empty_result(spec, "A+B versus A−/B−", "missing mutation flag"),
      empty_result(spec, "multiplicative A×B interaction", "missing mutation flag")
    ))
  }
  d <- if (spec$scope == "cancer-specific") {
    analysis[analysis$broadCancerCode == spec$cancer, , drop = FALSE]
  } else analysis
  covered_a <- as.logical(d[[call_a]])
  covered_b <- as.logical(d[[call_b]])
  d <- d[!is.na(covered_a) & !is.na(covered_b) & covered_a & covered_b, , drop = FALSE]
  d$mutA <- as.integer(as.numeric(d[[mut_a]]))
  d$mutB <- as.integer(as.numeric(d[[mut_b]]))
  d$interaction <- d$mutA * d$mutB
  d$A_only <- as.integer(d$mutA == 1L & d$mutB == 0L)
  d$B_only <- as.integer(d$mutA == 0L & d$mutB == 1L)
  d$Both <- as.integer(d$mutA == 1L & d$mutB == 1L)
  d$genotype <- 2L * d$mutA + d$mutB
  d$stratumId <- if (spec$scope == "cancer-specific") {
    d$studyId
  } else paste(d$studyId, d$broadCancerCode, sep = "::")
  stratum_split <- split(seq_len(nrow(d)), d$stratumId)
  retain <- vapply(stratum_split, function(index) {
    length(index) >= 10L && sum(d$event[index]) >= 2L &&
      length(unique(d$genotype[index])) >= 2L
  }, logical(1L))
  retained_strata <- names(retain)[retain]
  d <- d[d$stratumId %in% retained_strata, , drop = FALSE]

  fit_one <- function(parameterisation) {
    if (parameterisation == "four-state") {
      formula <- Surv(months, event) ~ A_only + B_only + Both + strata(stratumId)
      term <- "Both"
      contrast <- "A+B versus A−/B−"
    } else {
      formula <- Surv(months, event) ~ mutA + mutB + interaction + strata(stratumId)
      term <- "interaction"
      contrast <- "multiplicative A×B interaction"
    }
    warnings_seen <- character()
    fit <- tryCatch(
      withCallingHandlers(
        coxph(
          formula, data = d, ties = "efron", singular.ok = FALSE,
          model = TRUE, x = TRUE, y = TRUE
        ),
        warning = function(w) {
          warnings_seen <<- c(warnings_seen, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) e
    )
    if (inherits(fit, "error")) {
      return(empty_result(spec, contrast, paste("fit failed:", conditionMessage(fit))))
    }
    table <- summary(fit)$coefficients
    if (!(term %in% rownames(table))) {
      return(empty_result(spec, contrast, "target coefficient not estimable"))
    }
    coefficient <- unname(table[term, "coef"])
    standard_error <- unname(table[term, "se(coef)"])
    z_value <- coefficient / standard_error
    p_value <- 2 * pnorm(abs(z_value), lower.tail = FALSE)
    ph_p <- NA_real_
    ph <- tryCatch(cox.zph(fit, transform = "rank", terms = FALSE), error = function(e) NULL)
    if (!is.null(ph) && term %in% rownames(ph$table)) {
      ph_p <- unname(ph$table[term, "p"])
    }
    status <- if (length(warnings_seen)) {
      paste("estimated with warning:", paste(unique(warnings_seen), collapse = " | "))
    } else "estimated"
    result <- empty_result(spec, contrast, status)
    result$coefficient <- coefficient
    result$standardError <- standard_error
    result$hazardRatio <- exp(coefficient)
    result$ciLow <- exp(coefficient - qnorm(0.975) * standard_error)
    result$ciHigh <- exp(coefficient + qnorm(0.975) * standard_error)
    result$z <- z_value
    result$p <- p_value
    result$phTestP <- ph_p
    result
  }
  rbind(fit_one("four-state"), fit_one("interaction"))
}

results <- vector("list", nrow(specifications))
for (index in seq_len(nrow(specifications))) {
  if (index %% 100L == 0L) {
    message(sprintf("Fitted %d/%d survival contexts", index, nrow(specifications)))
  }
  results[[index]] <- fit_context(specifications[index, , drop = FALSE])
}
write.csv(do.call(rbind, results), args[[3L]], row.names = FALSE, na = "")
