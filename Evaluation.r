# Load required packages
library(scoringutils)  # Forecast scoring utilities
library(ggplot2)       # Plotting
library(dplyr)         # Data manipulation
library(data.table)    # Efficient data operations
library(tidyr)         # Data tidying

# ============================================================
# CONFIGURATION - Modify these paths according to your setup
# ============================================================
OUTPUT_DIR    <- "/scratch/user/s4921048/summer_research/output_darts_transformer_filled/"  # Directory containing forecast CSVs
TRAINING_DATA <- "/scratch/user/s4921048/summer_research/data/training_data.csv"            # Observed data CSV
RESULTS_DIR   <- "/scratch/user/s4921048/summer_research/evaluation_result/"                # Directory to save results

# Set a model name (needed because your scoringutils defaults to by='model')
MODEL_NAME <- "darts_transformer_filled"

# Country code mapping
COUNTRY_CODES <- c(
  'Belgium'='BE', 'Czech Republic'='CZ', 'France'='FR', 'Poland'='PL',
  'Romania'='RO', 'Austria'='AT', 'Germany'='DE', 'Italy'='IT',
  'Spain'='ES', 'Netherlands'='NL', 'Portugal'='PT', 'Sweden'='SE',
  'Norway'='NO', 'Denmark'='DK', 'Finland'='FI', 'Ireland'='IE',
  'United Kingdom'='GB', 'Switzerland'='CH', 'Hungary'='HU', 'Slovakia'='SK',
  'Slovenia'='SI', 'Croatia'='HR', 'Bulgaria'='BG', 'Greece'='GR',
  'Estonia'='EE', 'Latvia'='LV', 'Lithuania'='LT', 'Luxembourg'='LU',
  'Malta'='MT', 'Cyprus'='CY'
)

# Create results directory if needed
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# HELPER: Interval coverage plotting using your get_coverage() columns
# ============================================================
plot_interval_coverage_gg <- function(df, facet_vars = NULL, title = "", subtitle = "") {
  # df must contain: interval_range, interval_coverage
  p <- ggplot(df, aes(x = interval_range, y = interval_coverage)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey50") +
    geom_point(size = 1.8) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "Nominal Coverage (interval_range)",
      y = "Empirical Coverage (interval_coverage)"
    ) +
    theme_minimal()

  # facet_vars: NULL, or length 1 (facet_wrap), or length 2 (facet_grid)
  if (!is.null(facet_vars)) {
    if (length(facet_vars) == 1) {
      p <- p + facet_wrap(as.formula(paste("~", facet_vars[1])))
    } else if (length(facet_vars) == 2) {
      p <- p + facet_grid(as.formula(paste(facet_vars[1], "~", facet_vars[2])))
    }
  }
  p
}

# ============================================================
# STEP 1: Load and Prepare Data
# ============================================================
cat("Loading forecast files...\n")

# Get all forecast output files
files <- list.files(OUTPUT_DIR, pattern = "forecast_output_.*\\.csv$", full.names = TRUE)
if (length(files) == 0) stop("ERROR: No forecast files found in ", OUTPUT_DIR)

cat(sprintf("Found %d forecast files\n", length(files)))

# Combine all forecast files into one data.table
forecasts <- rbindlist(lapply(files, fread))

cat("Loading training data (actual observations)...\n")

# Load actual observation data
actuals <- fread(TRAINING_DATA)
actuals[, date := as.Date(date)]                      # Convert date
actuals[, location := COUNTRY_CODES[country_name]]    # Country -> code
actuals <- actuals[!is.na(location)]                  # Remove unmapped

# Convert forecast date columns
forecasts[, origin_date := as.Date(origin_date)]           # Forecast issue date
forecasts[, target_end_date := as.Date(target_end_date)]   # Target date

# Filter to quantile predictions only
quant_fc <- forecasts[output_type == "quantile"]
quant_fc[, quantile_level := as.numeric(output_type_id)]   # Quantile level numeric

# Prepare actuals for merging
actuals_for_merge <- actuals[, .(
  target_end_date = date,
  target,
  location,
  observed = value
)]

# Merge forecasts with actual observations
merged <- merge(quant_fc, actuals_for_merge, by = c("target_end_date", "target", "location"))
merged <- merged[!is.na(observed)]  # Keep only rows with observed values

cat(sprintf("Merged data: %d rows\n", nrow(merged)))

# Create forecast data frame for scoringutils
forecast_data <- merged[, .(
  model         = MODEL_NAME,       
  forecast_date = origin_date,
  target_end_date,
  target,
  location,
  horizon,
  observed,
  predicted     = value,
  quantile_level
)]

# Convert to scoringutils quantile forecast format
fc_obj <- as_forecast_quantile(
  forecast_data,
  forecast_unit  = c("model", "forecast_date", "target_end_date", "target", "location", "horizon"),
  observed       = "observed",
  predicted      = "predicted",
  quantile_level = "quantile_level"
)

cat("Forecast object created successfully.\n\n")

# ============================================================
# TASK 1: Interval Coverage Plot (Overall / by target / by horizon / by location / horizon x location)
# ============================================================
cat("========== TASK 1: Interval Coverage Plot ==========\n")

# ---------- Overall coverage (by model) ----------
# Using by="model" is safest for your scoringutils version
cov_overall <- as.data.table(get_coverage(fc_obj, by = "model"))

# Keep only interval-related columns needed for plotting
cov_overall_plot <- cov_overall[, .(model, interval_range, interval_coverage)]

p_cov_overall <- plot_interval_coverage_gg(
  cov_overall_plot,
  facet_vars = NULL,
  title = "Interval Coverage: Overall",
  subtitle = "Empirical vs nominal coverage (diagonal = perfect calibration)"
)

ggsave(file.path(RESULTS_DIR, "1_interval_coverage_overall.png"), p_cov_overall,
       width = 10, height = 7, dpi = 150)
cat("  Saved: 1_interval_coverage_overall.png\n")

# ---------- Coverage by target ----------
cov_by_target <- as.data.table(get_coverage(fc_obj, by = c("model", "target")))
cov_by_target_plot <- cov_by_target[, .(model, target, interval_range, interval_coverage)]

p_cov_target <- plot_interval_coverage_gg(
  cov_by_target_plot,
  facet_vars = c("target"),
  title = "Interval Coverage by Target",
  subtitle = "Faceted by target"
)

ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_target.png"), p_cov_target,
       width = 12, height = 8, dpi = 150)
cat("  Saved: 1_interval_coverage_by_target.png\n")

# ---------- Coverage by horizon ----------
cov_by_horizon <- as.data.table(get_coverage(fc_obj, by = c("model", "horizon")))
cov_by_horizon_plot <- cov_by_horizon[, .(model, horizon, interval_range, interval_coverage)]

p_cov_horizon <- plot_interval_coverage_gg(
  cov_by_horizon_plot,
  facet_vars = c("horizon"),
  title = "Interval Coverage by Horizon",
  subtitle = "Faceted by horizon"
)

ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_horizon.png"), p_cov_horizon,
       width = 12, height = 8, dpi = 150)
cat("  Saved: 1_interval_coverage_by_horizon.png\n")

# ---------- Coverage by location ----------
cov_by_location <- as.data.table(get_coverage(fc_obj, by = c("model", "location")))
cov_by_location_plot <- cov_by_location[, .(model, location, interval_range, interval_coverage)]

p_cov_location <- plot_interval_coverage_gg(
  cov_by_location_plot,
  facet_vars = c("location"),
  title = "Interval Coverage by Location",
  subtitle = "Faceted by location (may be crowded)"
)

ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_location.png"), p_cov_location,
       width = 16, height = 10, dpi = 150)
cat("  Saved: 1_interval_coverage_by_location.png\n")

# ---------- Coverage by horizon x location (requested grouping) ----------
cov_hxloc <- as.data.table(get_coverage(fc_obj, by = c("model", "horizon", "location")))
cov_hxloc_plot <- cov_hxloc[, .(model, horizon, location, interval_range, interval_coverage)]

p_cov_hxloc <- plot_interval_coverage_gg(
  cov_hxloc_plot,
  facet_vars = c("horizon", "location"),
  title = "Interval Coverage by Horizon and Location",
  subtitle = "Facet grid: rows=horizon, cols=location (can be very wide)"
)

ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_horizon_location.png"), p_cov_hxloc,
       width = 26, height = 10, dpi = 150)
cat("  Saved: 1_interval_coverage_by_horizon_location.png\n")

# Optional: If too many locations, plot only Top N locations for readability
TOP_N_LOC <- 12
loc_counts <- cov_hxloc_plot[, .N, by = location][order(-N)]
top_locs <- loc_counts$location[1:min(TOP_N_LOC, nrow(loc_counts))]
cov_hxloc_top <- cov_hxloc_plot[location %in% top_locs]

p_cov_hxloc_top <- plot_interval_coverage_gg(
  cov_hxloc_top,
  facet_vars = c("horizon", "location"),
  title = paste0("Interval Coverage by Horizon and Location (Top ", TOP_N_LOC, " locations)"),
  subtitle = "Reduced locations for readability"
)

ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_horizon_location_topN.png"), p_cov_hxloc_top,
       width = 20, height = 10, dpi = 150)
cat("  Saved: 1_interval_coverage_by_horizon_location_topN.png\n")

# ============================================================
# Compute Scores
# ============================================================
cat("\nComputing scores...\n")
scores <- score(fc_obj)
cat(sprintf("Computed scores for %d forecast instances.\n\n", nrow(scores)))

# ============================================================
# TASK 2: Score Summary by Forecast Date
# ============================================================
cat("========== TASK 2: Summary by Forecast Date ==========\n")

# Summarize by forecast date and target
summary_by_date <- summarise_scores(scores, by = c("forecast_date", "target"))
df_date <- as.data.frame(summary_by_date)
df_date$forecast_date <- as.Date(df_date$forecast_date)

# Save CSV
fwrite(df_date, file.path(RESULTS_DIR, "2_scores_by_forecast_date.csv"))
cat("  Saved: 2_scores_by_forecast_date.csv\n")

# Plot WIS over time
p_date <- ggplot(df_date, aes(x = forecast_date, y = wis, color = target)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  labs(
    title = "Weighted Interval Score (WIS) by Forecast Date",
    subtitle = "Lower WIS indicates better forecast accuracy",
    x = "Forecast Date",
    y = "WIS (lower is better)",
    color = "Target"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_x_date(date_labels = "%Y-%m-%d", date_breaks = "4 weeks") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(RESULTS_DIR, "2_wis_by_forecast_date.png"), p_date,
       width = 14, height = 7, dpi = 150)
cat("  Saved: 2_wis_by_forecast_date.png\n")

# Faceted trend with LOESS
p_date_facet <- ggplot(df_date, aes(x = forecast_date, y = wis)) +
  geom_line(color = "#2E86AB", linewidth = 0.6) +
  geom_point(color = "#2E86AB", size = 1) +
  geom_smooth(method = "loess", se = TRUE, color = "#E94F37", linewidth = 0.8) +
  facet_wrap(~target, scales = "free_y") +
  labs(
    title = "WIS Trend by Target (with LOESS smoothing)",
    subtitle = "Red line: smoothed trend; Gray area: confidence interval",
    x = "Forecast Date",
    y = "WIS"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

ggsave(file.path(RESULTS_DIR, "2_wis_trend_faceted.png"), p_date_facet,
       width = 14, height = 10, dpi = 150)
cat("  Saved: 2_wis_trend_faceted.png\n")

cat("\n  Summary statistics by forecast date:\n")
print(summary_by_date)

# ============================================================
# TASK 3: Score Summary by Horizon
# ============================================================
cat("\n========== TASK 3: Summary by Horizon ==========\n")

# Summarize by horizon and target
summary_by_horizon <- summarise_scores(scores, by = c("horizon", "target"))
df_horizon <- as.data.frame(summary_by_horizon)

# Save CSV
fwrite(df_horizon, file.path(RESULTS_DIR, "3_scores_by_horizon.csv"))
cat("  Saved: 3_scores_by_horizon.csv\n")

# Bar plot of WIS by horizon
p_horizon_bar <- ggplot(df_horizon, aes(x = factor(horizon), y = wis, fill = target)) +
  geom_col(position = "dodge", alpha = 0.85) +
  labs(
    title = "Weighted Interval Score (WIS) by Forecast Horizon",
    subtitle = "Comparing performance at different lead times",
    x = "Horizon (weeks ahead)",
    y = "WIS (lower is better)",
    fill = "Target"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave(file.path(RESULTS_DIR, "3_wis_by_horizon_bar.png"), p_horizon_bar,
       width = 10, height = 6, dpi = 150)
cat("  Saved: 3_wis_by_horizon_bar.png\n")

# Line plot showing skill decay
p_horizon_line <- ggplot(df_horizon, aes(x = horizon, y = wis, color = target, group = target)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 4) +
  labs(
    title = "WIS vs Forecast Horizon",
    subtitle = "Shows forecast skill degradation over time",
    x = "Horizon (weeks ahead)",
    y = "WIS (lower is better)",
    color = "Target"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_x_continuous(breaks = unique(df_horizon$horizon))

ggsave(file.path(RESULTS_DIR, "3_wis_by_horizon_line.png"), p_horizon_line,
       width = 10, height = 6, dpi = 150)
cat("  Saved: 3_wis_by_horizon_line.png\n")

# WIS decomposition plot
df_decomp <- df_horizon %>%
  select(horizon, target, overprediction, underprediction, dispersion) %>%
  pivot_longer(cols = c(overprediction, underprediction, dispersion),
               names_to = "component", values_to = "score")

p_decomp <- ggplot(df_decomp, aes(x = factor(horizon), y = score, fill = component)) +
  geom_col(position = "stack") +
  facet_wrap(~target) +
  scale_fill_manual(values = c(
    "overprediction"  = "#E94F37",
    "underprediction" = "#2E86AB",
    "dispersion"      = "#44AF69"
  )) +
  labs(
    title = "WIS Decomposition by Horizon",
    subtitle = "Red=Overprediction, Blue=Underprediction, Green=Dispersion",
    x = "Horizon",
    y = "Score Component",
    fill = "Component"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave(file.path(RESULTS_DIR, "3_wis_decomposition.png"), p_decomp,
       width = 12, height = 8, dpi = 150)
cat("  Saved: 3_wis_decomposition.png\n")

cat("\n  Summary statistics by horizon:\n")
print(summary_by_horizon)

# ============================================================
# BONUS: Overall Summary
# ============================================================
cat("\n========== BONUS: Overall Summary ==========\n")

overall <- summarise_scores(scores)
cat("\n  Overall performance:\n")
print(overall)

fwrite(as.data.frame(overall), file.path(RESULTS_DIR, "0_overall_summary.csv"))
cat("\n  Saved: 0_overall_summary.csv\n")