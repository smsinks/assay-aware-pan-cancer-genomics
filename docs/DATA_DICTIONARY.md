# Data dictionary

## Core analytical definitions

| Term | Definition |
|---|---|
| Selected tumour | One eligible tissue tumour per patient key after cross-study specimen reconciliation. |
| Assay-covered gene (operationally callable) | A gene within documented WES/WGS scope or present on the sample's documented targeted panel. Per-locus depth and quality metrics were not uniformly available. |
| Unassayed gene | A gene outside the documented assay scope for that tumour. |
| Mutation-positive | At least one retained protein-altering mutation in an assay-covered gene. |
| Assay-covered mutation-negative | An assay-covered gene without a retained protein-altering mutation. |
| Assay-aware prevalence | Mutation-positive tumours divided by assay-covered tumours for the gene. |
| Primary conditioned odds ratio | Mantel–Haenszel common odds ratio across strata defined by detailed cancer code, study and exact assay or panel, without mutation-burden conditioning. |
| Leave-two-out burden sensitivity | Primary strata additionally divided by quintiles of total retained mutation burden after subtracting the two tested gene indicators. |
| Total-burden diagnostic | Specification using quintiles of total retained mutation burden, including the two tested genes. |
| Cross-assay concordant | Sensitivity criterion requiring the same effect direction, WES/WGS FDR < 0.10 and targeted-panel nominal P < 0.05 under the stated model. |
| Hotspot-negative model | A cell-line model without a hotspot call for the tested gene. |
| Three-group functional comparator | Canonical hotspot mutant, other retained alteration in the same gene and no retained alteration in the gene. |
| Adjusted CRISPR effect | Hotspot-mutant minus hotspot-negative Chronos gene effect after cancer-lineage fixed-effect adjustment. |
| Adjusted sensitisation | Hotspot-negative minus hotspot-mutant PRISM AUC after cancer-lineage fixed-effect adjustment. |
| A−/B− | Jointly callable survival reference group without a retained mutation in either tested gene. |
| Joint genotype-state survival contrast | A+B versus A−/B− from a saturated four-genotype-state Cox model. |
| Multiplicative survival interaction | A×B product term conditional on the individual mutation A and mutation B main effects. |

Column names containing `callable` are retained for compatibility with the frozen
analysis state; throughout this release they implement the assay-scope definition above.

## Prepared result tables

| File | Rows represent | Principal quantities |
|---|---|---|
| `cohort_summary.csv` | Study-wide metrics | Cohort stages, assay counts and tumour–gene observation partitions |
| `cohort_by_cancer.csv` | Cancer families | Tumours, assay composition and callable fraction |
| `gene_prevalence.csv` | Recurrent genes | Callable denominator, mutation count, prevalence and Wilson interval |
| `mutation_evidence.csv` | Genes | CMC tier counts and tier-matched fraction |
| `hotspot_by_cancer.csv` | Gene–amino-acid–cancer contexts | Distinct tumour count |
| `landscape_heatmap.csv` | Cancer-family–gene cells | Callable, unassayed and mutated counts with prevalence |
| `gene_pair_contexts.csv` | Selected reference cancer–gene-pair contexts | Primary, leave-two-out and total-burden estimates with stability flags |
| `gene_pair_three_specifications.csv` | All 74,582 tested cancer–gene-pair contexts | Three CMH specifications, assay-stratified estimates, FDR, concordance and stability |
| `gene_pair_screen_summary.csv` | Prespecified robustness criteria | Pair counts retained at each stability and assay-concordance step |
| `gene_pair_study_heterogeneity.csv` | Principal and displayed pair contexts | Study-specific range, leave-one-study-out range, heterogeneity and random-effects estimates |
| `gene_pair_leave_one_study_out.csv` | Context–omitted-study combinations | Leave-one-study-out primary CMH estimates and direction stability |
| `gene_pair_off_panel_sensitivity.csv` | Principal and displayed pair contexts | Strict exclusion of specimens with assay-metadata discordance |
| `assay_discordance_by_study.csv` | Study–panel groups | Off-panel record, specimen, sample–gene-pair and gene counts |
| `pathway_by_cancer.csv` | Cancer–pathway combinations | WES/WGS denominator and mutation-only pathway frequency |
| `functional_crispr.csv` | Recurrent hotspot genotypes | Lineage-adjusted CRISPR effects and matched functional summaries |
| `functional_prism_selected.csv` | Selected genotype–compound contexts | Adjusted sensitisation, interval and FDR |
| `functional_cross_layer.csv` | Genes | CRISPR and PRISM evidence on separate effect scales |
| `network_contexts.csv` | Cancer-specific network contexts | Primary and sensitivity effects, stability, assay concordance and annotation group |
| `network_composition.csv` | Cancer-by-direction groups | Number of displayed contexts |
| `interactome_degree_matched_null.csv` | Display-network–connectivity metrics | Observed connectivity, matched-null expectation and empirical permutation P value |
| `survival_medians.csv` | Genotype groups within selected contexts | Patients, events and median overall survival |
| `survival_complete_screen.csv` | 2,492 cancer-specific and 120 pan-cancer contexts | Separate joint-state and formal multiplicative-interaction Cox estimates, global FDR and mutation-frequency thresholds |
| `survival_screen_eligibility.csv` | 5,254 mutation-frequency-qualified candidate contexts | Endpoint, study, stratum and four-state group counts with explicit model-eligibility state |
| `survival_pan_cancer.csv` | 13 pan-cancer gene pairs | Joint-state and multiplicative-interaction estimates with PH diagnostics |
| `survival_cancer_specific.csv` | 16 cancer-specific gene pairs | Joint-state and multiplicative-interaction estimates with PH diagnostics |
| `survival_sensitivity.csv` | Survival sensitivity specifications | Hazard ratios under each analysis specification |
| `survival_joint_state_interaction.csv` | 29 expanded diagnostic contexts | Separate joint-state and multiplicative-interaction estimates used for time-resolved diagnostics |
| `survival_ph_diagnostics.csv` | Model terms and global tests | Schoenfeld-residual proportional-hazards tests and FDR |
| `survival_piecewise_hazard_ratios.csv` | Context–follow-up-interval combinations | Hazard ratios for 0–12, 12–36 and >36 months |
| `survival_rmst_differences.csv` | Context–time-horizon combinations | A+B minus A−/B− restricted mean survival time at 36 and 60 months |
| `survival_time_varying_hazard_ratios.csv` | Context–follow-up-time points | Smoothed time-varying joint-state hazard ratios and intervals |
| `survival_primary_tumour_sensitivity.csv` | Primary-tumour-restricted contexts | Joint-state hazard ratios under specimen-type restriction |
| `survival_off_panel_sensitivity.csv` | Strict assay-discordance-exclusion contexts | Joint-state hazard ratios after excluding affected specimens |
| `survival_off_panel_specimen_audit.csv` | Study-level assay-discordance summaries | Excluded records and specimens used by the sensitivity |
| `survival_study_specific_hazard_ratios.csv` | Context–study combinations | Study-specific joint-state hazard ratios |
| `survival_study_meta_analysis.csv` | Meta-analysable survival contexts | Fixed- and random-effects study-level summaries and heterogeneity |
| `survival_leave_one_study_out.csv` | Principal context–omitted-study combinations | Leave-one-study-out random-effects meta-analysis estimates |
| `survival_leave_one_study_out_summary.csv` | Survival contexts | Minimum and maximum leave-one-study-out estimates and direction stability |
| `hallmark_by_cancer.csv` | Cancer–hallmark combinations | WES/WGS denominator, mutation count and Wilson interval |
| `pathway_gene_representation.csv` | Published pathway templates | Eligible genes, represented genes and representation percentage |
| `functional_three_group_comparator.csv` | Gene or genotype–compound contexts | Canonical hotspot, other-alteration and no-retained-alteration comparisons |
| `functional_leave_one_lineage_out.csv` | Functional context–omitted-lineage combinations | Influence estimates after excluding each informative lineage |
| `functional_leave_one_lineage_out_summary.csv` | Functional contexts | Leave-one-lineage-out range, median and direction stability |

## Figure inventory

`results/figures/` contains `figure1` through `figure9`. The supplementary inventory
contains Figures S1–S7. Every deposited image is a 600-dpi PNG and is displayed by the
executable notebook.
