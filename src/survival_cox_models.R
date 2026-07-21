#!/usr/bin/env Rscript

# Fit study-stratified Cox models with explicit Efron handling of tied event
# times.  Python prepares one row per analysis subject and a compact model-spec
# table; this script deliberately contains no cohort-selection logic.

suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: survival_cox_models.R <analysis_data.csv> <model_specs.csv> <output.csv>")
}

data_path <- args[[1L]]
spec_path <- args[[2L]]
output_path <- args[[3L]]

analysis_data <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)
model_specs <- read.csv(spec_path, stringsAsFactors = FALSE, check.names = FALSE)

required_data <- c(
  "datasetId", "months", "event", "studyId", "stratumId",
  "mutA", "mutB", "interaction", "A_only", "B_only", "Both",
  "age_z", "female"
)
missing_data <- setdiff(required_data, names(analysis_data))
if (length(missing_data) > 0L) {
  stop(sprintf("Analysis data missing columns: %s", paste(missing_data, collapse = ", ")))
}

required_specs <- c(
  "datasetId", "context", "scope", "cancer", "geneA", "geneB",
  "analysisPopulation", "model", "parameterization", "varianceEstimator",
  "clusterThreshold", "baselineHazards"
)
missing_specs <- setdiff(required_specs, names(model_specs))
if (length(missing_specs) > 0L) {
  stop(sprintf("Model specifications missing columns: %s", paste(missing_specs, collapse = ", ")))
}

empty_row <- function(spec, status, n = 0L, events = 0L, studies = 0L,
                      strata = 0L, cancers = 0L, clusters = 0L) {
  data.frame(
    datasetId = spec$datasetId,
    context = spec$context,
    scope = spec$scope,
    cancer = spec$cancer,
    geneA = spec$geneA,
    geneB = spec$geneB,
    analysisPopulation = spec$analysisPopulation,
    model = spec$model,
    parameterization = spec$parameterization,
    varianceEstimator = spec$varianceEstimator,
    term = "MODEL",
    nPatients = n,
    nEvents = events,
    nStudies = studies,
    nStrata = strata,
    nCancerGroups = cancers,
    nClusters = clusters,
    coefficient = NA_real_,
    standardError = NA_real_,
    hazardRatio = NA_real_,
    ciLow = NA_real_,
    ciHigh = NA_real_,
    z = NA_real_,
    p = NA_real_,
    phTestP = NA_real_,
    tiesMethod = "Efron",
    baselineHazards = spec$baselineHazards,
    adjustmentTerms = "none",
    fitStatus = status,
    stringsAsFactors = FALSE
  )
}

fit_one <- function(spec) {
  d <- analysis_data[analysis_data$datasetId == spec$datasetId, , drop = FALSE]
  d <- d[is.finite(d$months) & d$months > 0 & d$event %in% c(0, 1), , drop = FALSE]

  mutation_covariates <- if (spec$parameterization == "four-group") {
    c("A_only", "B_only", "Both")
  } else if (spec$parameterization == "interaction") {
    c("mutA", "mutB", "interaction")
  } else {
    return(empty_row(spec, sprintf("unknown parameterization: %s", spec$parameterization)))
  }
  requested_adjustment <- if (grepl("age/sex", spec$analysisPopulation, fixed = TRUE)) {
    c("age_z", "female")
  } else character()
  requested_covariates <- c(mutation_covariates, requested_adjustment)

  complete <- complete.cases(d[, c("months", "event", "studyId", "stratumId", requested_covariates), drop = FALSE])
  d <- d[complete, , drop = FALSE]
  n <- nrow(d)
  events <- sum(d$event)
  studies <- length(unique(d$studyId))
  strata <- length(unique(d$stratumId))
  cancers <- length(unique(d$broadCancerCode))
  clusters <- studies

  if (n < 80L || events < 20L) {
    return(empty_row(spec, "insufficient sample/event count", n, events, studies, strata, cancers, clusters))
  }
  if (any(vapply(d[, mutation_covariates, drop = FALSE], function(x) length(unique(x)) < 2L, logical(1L)))) {
    return(empty_row(spec, "non-varying mutation covariate", n, events, studies, strata, cancers, clusters))
  }

  # A recorded-sex covariate is structurally invariant in some organ-specific
  # cohorts. A term that varies only between baseline strata is likewise not
  # identifiable in a stratified Cox model. Preserve the declared complete-case
  # population while omitting only such non-estimable adjustment terms.
  estimable_within_strata <- function(variable) {
    length(unique(d[[variable]])) >= 2L && any(vapply(
      split(d[[variable]], d$stratumId),
      function(values) length(unique(values)) >= 2L,
      logical(1L)
    ))
  }
  active_adjustment <- requested_adjustment[
    vapply(requested_adjustment, estimable_within_strata, logical(1L))
  ]
  omitted_adjustment <- setdiff(requested_adjustment, active_adjustment)
  covariates <- c(mutation_covariates, active_adjustment)
  adjustment_terms <- if (length(requested_adjustment) == 0L) {
    "none"
  } else {
    display_adjustment <- c(age_z = "standardised age", female = "recorded sex")
    active_label <- if (length(active_adjustment)) {
      paste(unname(display_adjustment[active_adjustment]), collapse = " + ")
    } else "none estimable"
    omitted_label <- if (length(omitted_adjustment)) {
      paste0(
        "; omitted as non-estimable within strata: ",
        paste(unname(display_adjustment[omitted_adjustment]), collapse = ", ")
      )
    } else ""
    paste0(active_label, omitted_label)
  }
  if (spec$varianceEstimator == "study-clustered sandwich" &&
      studies < as.integer(spec$clusterThreshold)) {
    return(empty_row(
      spec,
      sprintf("not estimated: %d study clusters below threshold %d", studies, as.integer(spec$clusterThreshold)),
      n, events, studies, strata, cancers, clusters
    ))
  }

  rhs <- paste(c(covariates, "strata(stratumId)"), collapse = " + ")
  if (spec$varianceEstimator == "study-clustered sandwich") {
    rhs <- paste(rhs, "+ cluster(studyId)")
  }
  formula <- as.formula(paste("Surv(months, event) ~", rhs))

  warnings_seen <- character()
  fitter <- tryCatch(
    withCallingHandlers(
      coxph(
        formula,
        data = d,
        ties = "efron",
        singular.ok = FALSE,
        model = TRUE,
        x = TRUE,
        y = TRUE
      ),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )
  if (inherits(fitter, "error")) {
    return(empty_row(
      spec,
      paste("fit failed:", conditionMessage(fitter)),
      n, events, studies, strata, cancers, clusters
    ))
  }

  summary_fit <- summary(fitter)
  coef_table <- summary_fit$coefficients
  if (is.null(dim(coef_table))) {
    coef_table <- matrix(coef_table, nrow = 1L, dimnames = list(names(coef(fitter)), names(coef_table)))
  }
  se_column <- if (spec$varianceEstimator == "study-clustered sandwich") "robust se" else "se(coef)"
  if (!(se_column %in% colnames(coef_table))) {
    return(empty_row(
      spec,
      sprintf("fit did not return required %s column", se_column),
      n, events, studies, strata, cancers, clusters
    ))
  }

  ph_p <- setNames(rep(NA_real_, length(covariates)), covariates)
  # cox.zph is evaluated on the same Efron partial-likelihood fit.  For a
  # clustered fit, its coefficient-level test uses the fitted robust variance.
  ph <- tryCatch(cox.zph(fitter, transform = "rank", terms = FALSE), error = function(e) NULL)
  if (!is.null(ph)) {
    ph_table <- ph$table
    common <- intersect(covariates, rownames(ph_table))
    ph_p[common] <- ph_table[common, "p"]
  }

  status <- "estimated"
  if (length(warnings_seen) > 0L) {
    status <- paste("estimated with warning:", paste(unique(warnings_seen), collapse = " | "))
  }
  z_values <- coef_table[, "coef"] / coef_table[, se_column]
  p_values <- 2 * pnorm(abs(z_values), lower.tail = FALSE)
  ci_low <- exp(coef_table[, "coef"] - qnorm(0.975) * coef_table[, se_column])
  ci_high <- exp(coef_table[, "coef"] + qnorm(0.975) * coef_table[, se_column])

  rows <- data.frame(
    datasetId = spec$datasetId,
    context = spec$context,
    scope = spec$scope,
    cancer = spec$cancer,
    geneA = spec$geneA,
    geneB = spec$geneB,
    analysisPopulation = spec$analysisPopulation,
    model = spec$model,
    parameterization = spec$parameterization,
    varianceEstimator = spec$varianceEstimator,
    term = rownames(coef_table),
    nPatients = n,
    nEvents = events,
    nStudies = studies,
    nStrata = strata,
    nCancerGroups = cancers,
    nClusters = clusters,
    coefficient = coef_table[, "coef"],
    standardError = coef_table[, se_column],
    hazardRatio = exp(coef_table[, "coef"]),
    ciLow = ci_low,
    ciHigh = ci_high,
    z = z_values,
    p = p_values,
    phTestP = unname(ph_p[rownames(coef_table)]),
    tiesMethod = "Efron",
    baselineHazards = spec$baselineHazards,
    adjustmentTerms = adjustment_terms,
    fitStatus = status,
    stringsAsFactors = FALSE
  )
  rows
}

results <- lapply(seq_len(nrow(model_specs)), function(i) fit_one(model_specs[i, , drop = FALSE]))
output <- do.call(rbind, results)
write.csv(output, output_path, row.names = FALSE, na = "")
