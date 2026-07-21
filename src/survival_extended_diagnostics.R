#!/usr/bin/env Rscript

# Extended diagnostics for the 29 prespecified primary joint-genotype survival
# models. Cohort construction remains in 22_curated_survival.py; this helper
# operates only on the frozen, strictly positive-time model populations passed
# by Python. It uses survival::coxph with Efron's method throughout.

suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop(
    "Usage: survival_extended_diagnostics.R <analysis_data.csv> ",
    "<primary_specs.csv> <output_directory>"
  )
}

data_path <- args[[1L]]
spec_path <- args[[2L]]
output_directory <- args[[3L]]
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

analysis_data <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)
model_specs <- read.csv(spec_path, stringsAsFactors = FALSE, check.names = FALSE)

required_data <- c(
  "datasetId", "sampleId", "studyId", "stratumId", "broadCancerCode",
  "sampleTypeGroup", "hasAssayScopeConflict", "months", "event",
  "A_only", "B_only", "Both"
)
missing_data <- setdiff(required_data, names(analysis_data))
if (length(missing_data) > 0L) {
  stop(sprintf("Analysis data missing columns: %s", paste(missing_data, collapse = ", ")))
}
analysis_data$hasAssayScopeConflict <- tolower(
  as.character(analysis_data$hasAssayScopeConflict)
) %in% c("true", "t", "1")

required_specs <- c(
  "datasetId", "context", "scope", "cancer", "geneA", "geneB",
  "analysisPopulation", "baselineHazards"
)
missing_specs <- setdiff(required_specs, names(model_specs))
if (length(missing_specs) > 0L) {
  stop(sprintf("Model specifications missing columns: %s", paste(missing_specs, collapse = ", ")))
}

if (anyDuplicated(model_specs$datasetId)) {
  stop("Primary model specifications must contain one row per datasetId")
}

bind_rows <- function(rows) {
  rows <- Filter(function(x) !is.null(x) && nrow(x) > 0L, rows)
  if (length(rows) == 0L) data.frame() else do.call(rbind, rows)
}

write_result <- function(frame, filename) {
  write.csv(frame, file.path(output_directory, filename), row.names = FALSE, na = "")
}

context_columns <- function(spec) {
  list(
    datasetId = as.character(spec$datasetId),
    context = as.character(spec$context),
    scope = as.character(spec$scope),
    cancer = as.character(spec$cancer),
    geneA = as.character(spec$geneA),
    geneB = as.character(spec$geneB)
  )
}

primary_formula <- Surv(months, event) ~ A_only + B_only + Both + strata(stratumId)

fit_primary <- function(d) {
  d <- d[
    is.finite(d$months) & d$months > 0 & d$event %in% c(0, 1),
    , drop = FALSE
  ]
  if (nrow(d) < 80L || sum(d$event) < 20L) {
    return(list(fit = NULL, status = "insufficient sample/event count", data = d))
  }
  mutation_covariates <- c("A_only", "B_only", "Both")
  if (any(vapply(
    d[, mutation_covariates, drop = FALSE],
    function(x) length(unique(x)) < 2L,
    logical(1L)
  ))) {
    return(list(fit = NULL, status = "non-varying genotype covariate", data = d))
  }
  warnings_seen <- character()
  fitter <- tryCatch(
    withCallingHandlers(
      coxph(
        primary_formula,
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
    return(list(
      fit = NULL,
      status = paste("fit failed:", conditionMessage(fitter)),
      data = d
    ))
  }
  status <- if (length(warnings_seen)) {
    paste("estimated with warning:", paste(unique(warnings_seen), collapse = " | "))
  } else "estimated"
  list(fit = fitter, status = status, data = d)
}

estimate_row <- function(fit, term) {
  if (is.null(fit) || !(term %in% names(coef(fit)))) {
    return(c(
      coefficient = NA_real_, standardError = NA_real_, hazardRatio = NA_real_,
      ciLow = NA_real_, ciHigh = NA_real_, z = NA_real_, p = NA_real_
    ))
  }
  beta <- unname(coef(fit)[term])
  variance <- vcov(fit)
  se <- sqrt(unname(variance[term, term]))
  z <- beta / se
  c(
    coefficient = beta,
    standardError = se,
    hazardRatio = exp(beta),
    ciLow = exp(beta - qnorm(0.975) * se),
    ciHigh = exp(beta + qnorm(0.975) * se),
    z = z,
    p = 2 * pnorm(abs(z), lower.tail = FALSE)
  )
}

ph_rows <- list()
schoenfeld_rows <- list()
time_varying_rows <- list()
piecewise_rows <- list()
rmst_rows <- list()
primary_tumour_rows <- list()
assay_discordance_rows <- list()
study_rows <- list()
meta_rows <- list()
loo_rows <- list()
audit_rows <- list()

major_cancer_contexts <- c(
  "KEAP1–STK11 (LUAD)",
  "KRAS–TP53 (PAAD)",
  "PIK3CA–TP53 (BRCA)",
  "STK11–TP53 (LUAD)",
  "KRAS–TP53 (LUAD)"
)

piecewise_bands <- data.frame(
  interval = c("0–12 months", "12–36 months", ">36 months"),
  lowerMonths = c(0, 12, 36),
  upperMonths = c(12, 36, Inf),
  stringsAsFactors = FALSE
)

meta_analysis <- function(beta, se) {
  keep <- is.finite(beta) & is.finite(se) & se > 0
  beta <- beta[keep]
  se <- se[keep]
  k <- length(beta)
  if (k < 2L) return(NULL)
  variance <- se^2
  fixed_weights <- 1 / variance
  fixed_beta <- sum(fixed_weights * beta) / sum(fixed_weights)
  fixed_se <- sqrt(1 / sum(fixed_weights))
  q <- sum(fixed_weights * (beta - fixed_beta)^2)
  q_df <- k - 1L
  c_value <- sum(fixed_weights) - sum(fixed_weights^2) / sum(fixed_weights)
  tau2 <- if (c_value > 0) max(0, (q - q_df) / c_value) else 0
  random_weights <- 1 / (variance + tau2)
  random_beta <- sum(random_weights * beta) / sum(random_weights)
  random_se <- sqrt(1 / sum(random_weights))
  i2 <- if (q > 0) max(0, (q - q_df) / q) * 100 else 0
  c(
    nInformativeStudies = k,
    fixedCoefficient = fixed_beta,
    fixedStandardError = fixed_se,
    fixedHazardRatio = exp(fixed_beta),
    fixedCiLow = exp(fixed_beta - qnorm(0.975) * fixed_se),
    fixedCiHigh = exp(fixed_beta + qnorm(0.975) * fixed_se),
    fixedP = 2 * pnorm(abs(fixed_beta / fixed_se), lower.tail = FALSE),
    randomCoefficient = random_beta,
    randomStandardError = random_se,
    randomHazardRatio = exp(random_beta),
    randomCiLow = exp(random_beta - qnorm(0.975) * random_se),
    randomCiHigh = exp(random_beta + qnorm(0.975) * random_se),
    randomP = 2 * pnorm(abs(random_beta / random_se), lower.tail = FALSE),
    tauSquared = tau2,
    cochranQ = q,
    cochranQDf = q_df,
    cochranQP = pchisq(q, df = q_df, lower.tail = FALSE),
    iSquaredPercent = i2,
    predictionLow = exp(random_beta - qnorm(0.975) * sqrt(random_se^2 + tau2)),
    predictionHigh = exp(random_beta + qnorm(0.975) * sqrt(random_se^2 + tau2))
  )
}

for (i in seq_len(nrow(model_specs))) {
  spec <- model_specs[i, , drop = FALSE]
  ids <- context_columns(spec)
  d <- analysis_data[analysis_data$datasetId == spec$datasetId, , drop = FALSE]
  d <- d[
    is.finite(d$months) & d$months > 0 & d$event %in% c(0, 1),
    , drop = FALSE
  ]
  primary <- fit_primary(d)
  fit <- primary$fit
  both_estimate <- estimate_row(fit, "Both")

  audit_rows[[length(audit_rows) + 1L]] <- data.frame(
    ids,
    nPatients = nrow(d),
    nEvents = sum(d$event),
    nStudies = length(unique(d$studyId)),
    nStrata = length(unique(d$stratumId)),
    nPrimaryTumours = sum(d$sampleTypeGroup == "Primary", na.rm = TRUE),
    primaryTumourPercentage = 100 * mean(d$sampleTypeGroup == "Primary", na.rm = TRUE),
    primaryModelHazardRatio = both_estimate[["hazardRatio"]],
    primaryFitStatus = primary$status,
    endpoint = "Overall survival as reported by contributing studies",
    tiesMethod = "Efron",
    timeOriginSensitivity = paste(
      "Not estimable: a harmonised diagnosis-to-overall-survival origin",
      "was not available across studies"
    ),
    stringsAsFactors = FALSE
  )

  # Coefficient-level and global proportional-hazards diagnostics. cox.zph$y
  # contains the scaled Schoenfeld residual estimate of beta(t), which is also
  # used by survival's plot.cox.zph method.
  zph <- if (!is.null(fit)) {
    tryCatch(cox.zph(fit, transform = "rank", terms = FALSE), error = function(e) NULL)
  } else NULL
  if (!is.null(zph)) {
    ph_table <- zph$table
    for (term in rownames(ph_table)) {
      ph_rows[[length(ph_rows) + 1L]] <- data.frame(
        ids,
        term = term,
        chiSquare = unname(ph_table[term, "chisq"]),
        degreesFreedom = unname(ph_table[term, "df"]),
        phTestP = unname(ph_table[term, "p"]),
        transform = "rank",
        diagnostic = "cox.zph scaled Schoenfeld-residual test",
        tiesMethod = "Efron",
        stringsAsFactors = FALSE
      )
    }
    if ("Both" %in% colnames(zph$y)) {
      residual_frame <- data.frame(
        ids,
        eventTimeMonths = zph$time,
        transformedTime = zph$x,
        scaledSchoenfeldBeta = zph$y[, "Both"],
        scaledSchoenfeldHazardRatio = exp(zph$y[, "Both"]),
        stringsAsFactors = FALSE
      )
      residual_frame <- residual_frame[is.finite(residual_frame$scaledSchoenfeldBeta), ]
      schoenfeld_rows[[length(schoenfeld_rows) + 1L]] <- residual_frame

      smooth <- tryCatch(
        plot(zph, var = "Both", resid = FALSE, se = TRUE, df = 4, nsmo = 100, plot = FALSE),
        error = function(e) NULL
      )
      if (!is.null(smooth)) {
        time_map <- aggregate(zph$time, by = list(transformedTime = zph$x), FUN = median)
        time_map <- time_map[order(time_map$transformedTime), ]
        raw_time <- approx(
          time_map$transformedTime,
          time_map$x,
          xout = smooth$x,
          rule = 2,
          ties = mean
        )$y
        time_varying_rows[[length(time_varying_rows) + 1L]] <- data.frame(
          ids,
          eventTimeMonths = raw_time,
          transformedTime = smooth$x,
          coefficient = smooth$y[, 1L],
          ciHighCoefficient = smooth$y[, 2L],
          ciLowCoefficient = smooth$y[, 3L],
          hazardRatio = exp(smooth$y[, 1L]),
          ciLow = exp(smooth$y[, 3L]),
          ciHigh = exp(smooth$y[, 2L]),
          splineDegreesFreedom = 4,
          diagnostic = "natural-spline smooth of scaled Schoenfeld beta(t)",
          stringsAsFactors = FALSE
        )
      }
    }
  }

  # Conditional interval-specific Cox estimates. Patients contribute to an
  # interval only if still under observation at its lower boundary.
  for (band_index in seq_len(nrow(piecewise_bands))) {
    band <- piecewise_bands[band_index, ]
    lower <- band$lowerMonths
    upper <- band$upperMonths
    interval_data <- d[d$months > lower, , drop = FALSE]
    interval_data$intervalTime <- if (is.finite(upper)) {
      pmin(interval_data$months, upper) - lower
    } else interval_data$months - lower
    interval_data$intervalEvent <- as.integer(
      interval_data$event == 1 &
        interval_data$months > lower &
        (!is.finite(upper) | interval_data$months <= upper)
    )
    interval_fit <- NULL
    interval_status <- "estimated"
    if (nrow(interval_data) < 80L || sum(interval_data$intervalEvent) < 20L) {
      interval_status <- "insufficient sample/event count"
    } else if (any(vapply(
      interval_data[, c("A_only", "B_only", "Both"), drop = FALSE],
      function(x) length(unique(x)) < 2L,
      logical(1L)
    ))) {
      interval_status <- "non-varying genotype covariate"
    } else {
      interval_fit <- tryCatch(
        coxph(
          Surv(intervalTime, intervalEvent) ~ A_only + B_only + Both + strata(stratumId),
          data = interval_data,
          ties = "efron",
          singular.ok = FALSE,
          model = TRUE,
          x = TRUE,
          y = TRUE
        ),
        error = function(e) e
      )
      if (inherits(interval_fit, "error")) {
        interval_status <- paste("fit failed:", conditionMessage(interval_fit))
        interval_fit <- NULL
      }
    }
    interval_estimate <- estimate_row(interval_fit, "Both")
    piecewise_rows[[length(piecewise_rows) + 1L]] <- data.frame(
      ids,
      interval = band$interval,
      lowerMonths = lower,
      upperMonths = if (is.finite(upper)) upper else NA_real_,
      nAtRiskAtIntervalStart = nrow(interval_data),
      nEventsInInterval = sum(interval_data$intervalEvent),
      nBothAtRiskAtIntervalStart = sum(interval_data$Both == 1),
      nDoubleNegativeAtRiskAtIntervalStart = sum(
        interval_data$A_only == 0 & interval_data$B_only == 0 & interval_data$Both == 0
      ),
      coefficient = interval_estimate[["coefficient"]],
      standardError = interval_estimate[["standardError"]],
      hazardRatio = interval_estimate[["hazardRatio"]],
      ciLow = interval_estimate[["ciLow"]],
      ciHigh = interval_estimate[["ciHigh"]],
      p = interval_estimate[["p"]],
      contrast = "A+B versus A−/B−",
      model = "conditional interval-specific four-group Cox model",
      baselineHazards = as.character(spec$baselineHazards),
      tiesMethod = "Efron",
      fitStatus = interval_status,
      stringsAsFactors = FALSE
    )
  }

  # Non-parametric restricted mean survival time contrasts at 36 and 60 months.
  # The contrast is intentionally separate from the Cox interaction term.
  rmst_data <- d[
    d$Both == 1 | (d$A_only == 0 & d$B_only == 0 & d$Both == 0),
    , drop = FALSE
  ]
  rmst_data$rmstGroup <- factor(
    ifelse(rmst_data$Both == 1, "A+B", "A−/B−"),
    levels = c("A−/B−", "A+B")
  )
  for (tau in c(36, 60)) {
    rmst_status <- "estimated"
    rmst_values <- rep(NA_real_, 4L)
    names(rmst_values) <- c("reference", "referenceSe", "both", "bothSe")
    if (length(unique(rmst_data$rmstGroup)) < 2L) {
      rmst_status <- "both contrast groups not represented"
    } else {
      sf <- tryCatch(
        survfit(Surv(months, event) ~ rmstGroup, data = rmst_data),
        error = function(e) e
      )
      if (inherits(sf, "error")) {
        rmst_status <- paste("fit failed:", conditionMessage(sf))
      } else {
        sf_table <- summary(sf, rmean = tau, extend = TRUE)$table
        if (is.null(dim(sf_table)) || nrow(sf_table) != 2L) {
          rmst_status <- "RMST groups not estimable"
        } else {
          reference_row <- grep("A−/B−", rownames(sf_table), fixed = TRUE)
          both_row <- grep("A+B", rownames(sf_table), fixed = TRUE)
          if (length(reference_row) == 1L && length(both_row) == 1L) {
            rmst_values <- c(
              reference = sf_table[reference_row, "rmean"],
              referenceSe = sf_table[reference_row, "se(rmean)"],
              both = sf_table[both_row, "rmean"],
              bothSe = sf_table[both_row, "se(rmean)"]
            )
          } else rmst_status <- "RMST group labels not recovered"
        }
      }
    }
    difference <- rmst_values[["both"]] - rmst_values[["reference"]]
    difference_se <- sqrt(rmst_values[["bothSe"]]^2 + rmst_values[["referenceSe"]]^2)
    difference_z <- difference / difference_se
    rmst_rows[[length(rmst_rows) + 1L]] <- data.frame(
      ids,
      horizonMonths = tau,
      nDoubleNegative = sum(rmst_data$rmstGroup == "A−/B−"),
      nBoth = sum(rmst_data$rmstGroup == "A+B"),
      nDoubleNegativeAtHorizon = sum(
        rmst_data$rmstGroup == "A−/B−" & rmst_data$months >= tau
      ),
      nBothAtHorizon = sum(rmst_data$rmstGroup == "A+B" & rmst_data$months >= tau),
      rmstDoubleNegativeMonths = rmst_values[["reference"]],
      rmstDoubleNegativeSe = rmst_values[["referenceSe"]],
      rmstBothMonths = rmst_values[["both"]],
      rmstBothSe = rmst_values[["bothSe"]],
      rmstDifferenceMonths = difference,
      standardError = difference_se,
      ciLow = difference - qnorm(0.975) * difference_se,
      ciHigh = difference + qnorm(0.975) * difference_se,
      p = 2 * pnorm(abs(difference_z), lower.tail = FALSE),
      contrast = "A+B minus A−/B− restricted mean overall survival",
      estimator = "Kaplan–Meier area under the survival curve",
      adjustment = "descriptive unadjusted genotype-state contrast",
      fitStatus = rmst_status,
      stringsAsFactors = FALSE
    )
  }

  # Restrict the frozen primary population to explicitly annotated primary
  # tumours. Unknown specimen types are not reclassified.
  primary_tumour_data <- d[d$sampleTypeGroup == "Primary", , drop = FALSE]
  primary_tumour_fit <- fit_primary(primary_tumour_data)
  primary_tumour_estimate <- estimate_row(primary_tumour_fit$fit, "Both")
  primary_tumour_rows[[length(primary_tumour_rows) + 1L]] <- data.frame(
    ids,
    nPatients = nrow(primary_tumour_data),
    nEvents = sum(primary_tumour_data$event),
    nStudies = length(unique(primary_tumour_data$studyId)),
    nStrata = length(unique(primary_tumour_data$stratumId)),
    coefficient = primary_tumour_estimate[["coefficient"]],
    standardError = primary_tumour_estimate[["standardError"]],
    hazardRatio = primary_tumour_estimate[["hazardRatio"]],
    ciLow = primary_tumour_estimate[["ciLow"]],
    ciHigh = primary_tumour_estimate[["ciHigh"]],
    p = primary_tumour_estimate[["p"]],
    contrast = "A+B versus A−/B−",
    sampleRestriction = "sampleTypeGroup exactly equal to Primary",
    unclassifiedSpecimensRetained = FALSE,
    baselineHazards = as.character(spec$baselineHazards),
    tiesMethod = "Efron",
    fitStatus = primary_tumour_fit$status,
    stringsAsFactors = FALSE
  )

  # Metadata sensitivity: remove every specimen carrying any observed mutation
  # outside its documented assay scope. Off-panel positives remain excluded and
  # are never used to reconstruct negative assay denominators.
  n_assay_conflict <- sum(d$hasAssayScopeConflict, na.rm = TRUE)
  assay_sensitivity_data <- d[!d$hasAssayScopeConflict, , drop = FALSE]
  assay_sensitivity_fit <- fit_primary(assay_sensitivity_data)
  assay_sensitivity_estimate <- estimate_row(assay_sensitivity_fit$fit, "Both")
  assay_status <- if (n_assay_conflict == 0L) {
    "structural no-overlap; estimate identical to strict primary"
  } else assay_sensitivity_fit$status
  assay_discordance_rows[[length(assay_discordance_rows) + 1L]] <- data.frame(
    ids,
    nPrimaryPatients = nrow(d),
    nExcludedConflictSpecimens = n_assay_conflict,
    nPatients = nrow(assay_sensitivity_data),
    nEvents = sum(assay_sensitivity_data$event),
    nStudies = length(unique(assay_sensitivity_data$studyId)),
    nStrata = length(unique(assay_sensitivity_data$stratumId)),
    coefficient = assay_sensitivity_estimate[["coefficient"]],
    standardError = assay_sensitivity_estimate[["standardError"]],
    hazardRatio = assay_sensitivity_estimate[["hazardRatio"]],
    ciLow = assay_sensitivity_estimate[["ciLow"]],
    ciHigh = assay_sensitivity_estimate[["ciHigh"]],
    p = assay_sensitivity_estimate[["p"]],
    contrast = "A+B versus A−/B−",
    sensitivityDefinition = paste(
      "exclude each specimen with any off-panel assay-metadata conflict;",
      "do not reclassify off-panel positives as assayed"
    ),
    baselineHazards = as.character(spec$baselineHazards),
    tiesMethod = "Efron",
    fitStatus = assay_status,
    stringsAsFactors = FALSE
  )

  # Study-specific binary A+B versus double-negative Cox estimates. Other two
  # genotype groups are omitted here so every study contributes the same direct
  # clinical contrast; the pooled primary four-group model remains authoritative.
  study_estimates <- list()
  for (study in sort(unique(d$studyId))) {
    study_data <- d[d$studyId == study, , drop = FALSE]
    study_data <- study_data[
      study_data$Both == 1 |
        (study_data$A_only == 0 & study_data$B_only == 0 & study_data$Both == 0),
      , drop = FALSE
    ]
    study_data$jointState <- study_data$Both
    status <- "estimated"
    study_fit <- NULL
    if (
      nrow(study_data) < 20L || sum(study_data$event) < 5L ||
      sum(study_data$jointState == 1L) < 3L ||
      sum(study_data$jointState == 0L) < 8L
    ) {
      status <- "insufficient direct-contrast sample/event count"
    } else if (length(unique(study_data$jointState)) < 2L) {
      status <- "joint genotype contrast not represented"
    } else {
      study_formula <- if (as.character(spec$scope) == "pan-cancer") {
        Surv(months, event) ~ jointState + strata(broadCancerCode)
      } else Surv(months, event) ~ jointState
      study_fit <- tryCatch(
        coxph(
          study_formula,
          data = study_data,
          ties = "efron",
          singular.ok = FALSE,
          model = TRUE,
          x = TRUE,
          y = TRUE
        ),
        error = function(e) e
      )
      if (inherits(study_fit, "error")) {
        status <- paste("fit failed:", conditionMessage(study_fit))
        study_fit <- NULL
      }
    }
    study_estimate <- estimate_row(study_fit, "jointState")
    study_frame <- data.frame(
      ids,
      studyId = study,
      nPatients = nrow(study_data),
      nEvents = sum(study_data$event),
      nBoth = sum(study_data$jointState == 1L),
      nDoubleNegative = sum(study_data$jointState == 0L),
      nCancerGroups = length(unique(study_data$broadCancerCode)),
      coefficient = study_estimate[["coefficient"]],
      standardError = study_estimate[["standardError"]],
      hazardRatio = study_estimate[["hazardRatio"]],
      ciLow = study_estimate[["ciLow"]],
      ciHigh = study_estimate[["ciHigh"]],
      p = study_estimate[["p"]],
      contrast = "A+B versus A−/B−",
      model = if (as.character(spec$scope) == "pan-cancer") {
        "within-study Cox model with cancer-group baseline strata"
      } else "within-study Cox model",
      tiesMethod = "Efron",
      fitStatus = status,
      stringsAsFactors = FALSE
    )
    study_estimates[[length(study_estimates) + 1L]] <- study_frame
    study_rows[[length(study_rows) + 1L]] <- study_frame
  }

  study_frame <- bind_rows(study_estimates)
  informative <- study_frame[
    is.finite(study_frame$coefficient) &
      is.finite(study_frame$standardError) &
      study_frame$standardError > 0,
    , drop = FALSE
  ]
  meta <- meta_analysis(informative$coefficient, informative$standardError)
  if (!is.null(meta)) {
    meta_rows[[length(meta_rows) + 1L]] <- data.frame(
      ids,
      as.list(meta),
      contrast = "A+B versus A−/B−",
      studyEstimator = "within-study Cox model",
      poolingMethod = "DerSimonian–Laird random-effects and inverse-variance fixed-effect",
      stringsAsFactors = FALSE
    )
    if (nrow(informative) >= 3L) {
      for (excluded_study in informative$studyId) {
        retained <- informative[informative$studyId != excluded_study, , drop = FALSE]
        loo_meta <- meta_analysis(retained$coefficient, retained$standardError)
        if (is.null(loo_meta)) next
        loo_rows[[length(loo_rows) + 1L]] <- data.frame(
          ids,
          excludedStudyId = excluded_study,
          nPatientsExcludedStudy = informative$nPatients[
            informative$studyId == excluded_study
          ][1L],
          nEventsExcludedStudy = informative$nEvents[
            informative$studyId == excluded_study
          ][1L],
          nStudiesRetained = loo_meta[["nInformativeStudies"]],
          coefficient = loo_meta[["randomCoefficient"]],
          standardError = loo_meta[["randomStandardError"]],
          hazardRatio = loo_meta[["randomHazardRatio"]],
          ciLow = loo_meta[["randomCiLow"]],
          ciHigh = loo_meta[["randomCiHigh"]],
          p = loo_meta[["randomP"]],
          tauSquared = loo_meta[["tauSquared"]],
          iSquaredPercent = loo_meta[["iSquaredPercent"]],
          contrast = "A+B versus A−/B−",
          leaveOneOutMethod = "leave-one-study-out random-effects meta-analysis",
          fitStatus = "estimated",
          stringsAsFactors = FALSE
        )
      }
    }
  }

  # For the five principal cancer-specific contexts, also refit the exact pooled
  # stratified four-group Cox model after removing each study in turn.
  if (as.character(spec$context) %in% major_cancer_contexts) {
    for (excluded_study in sort(unique(d$studyId))) {
      retained <- d[d$studyId != excluded_study, , drop = FALSE]
      loo_fit <- fit_primary(retained)
      loo_estimate <- estimate_row(loo_fit$fit, "Both")
      excluded <- d[d$studyId == excluded_study, , drop = FALSE]
      loo_rows[[length(loo_rows) + 1L]] <- data.frame(
        ids,
        excludedStudyId = excluded_study,
        nPatientsExcludedStudy = nrow(excluded),
        nEventsExcludedStudy = sum(excluded$event),
        nStudiesRetained = length(unique(retained$studyId)),
        coefficient = loo_estimate[["coefficient"]],
        standardError = loo_estimate[["standardError"]],
        hazardRatio = loo_estimate[["hazardRatio"]],
        ciLow = loo_estimate[["ciLow"]],
        ciHigh = loo_estimate[["ciHigh"]],
        p = loo_estimate[["p"]],
        tauSquared = NA_real_,
        iSquaredPercent = NA_real_,
        contrast = "A+B versus A−/B−",
        leaveOneOutMethod = "exact pooled stratified four-group Cox refit",
        fitStatus = loo_fit$status,
        stringsAsFactors = FALSE
      )
    }
  }
}

write_result(bind_rows(ph_rows), "survival_ph_diagnostics.csv")
write_result(bind_rows(schoenfeld_rows), "survival_scaled_schoenfeld_residuals.csv")
write_result(bind_rows(time_varying_rows), "survival_time_varying_hazard_ratios.csv")
write_result(bind_rows(piecewise_rows), "survival_piecewise_hazard_ratios.csv")
write_result(bind_rows(rmst_rows), "survival_rmst_differences.csv")
write_result(bind_rows(primary_tumour_rows), "survival_primary_tumour_sensitivity.csv")
write_result(
  bind_rows(assay_discordance_rows),
  "survival_assay_discordance_exclusion_sensitivity.csv"
)
write_result(bind_rows(study_rows), "survival_study_specific_hazard_ratios.csv")
write_result(bind_rows(meta_rows), "survival_study_meta_analysis.csv")
write_result(bind_rows(loo_rows), "survival_leave_one_study_out.csv")
write_result(bind_rows(audit_rows), "survival_extended_diagnostic_audit.csv")

software <- data.frame(
  component = c("R", "survival"),
  version = c(R.version.string, as.character(packageVersion("survival"))),
  role = c(
    "extended survival diagnostic runtime",
    "Efron Cox models, cox.zph, Kaplan–Meier and RMST calculations"
  ),
  stringsAsFactors = FALSE
)
write_result(software, "survival_software_environment.csv")
