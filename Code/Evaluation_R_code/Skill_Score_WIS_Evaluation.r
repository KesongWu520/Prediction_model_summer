# ============================================================
# Respiratory Disease Forecast Analysis Script
# Adapted for transformer_forecast.py output format
#
# Reads Transformer output from all_forecasts.csv
# Reads ARIMA output from all_forecasts.csv (separate directory)
#
# Main Outputs:
#   1) Interval Coverage Plots
#   2) Score summary by forecast date
#   3) Score summary by horizon
#   4) Skill Score Heatmap: Transformer vs ARIMA  (NEW)
#   0) Overall summary
# ============================================================

library(scoringutils)
library(ggplot2)
library(dplyr)
library(data.table)
library(tidyr)

# ============================================================
# CONFIGURATION
# ============================================================
TRANSFORMER_DIR <- "/scratch/user/s4921048/summer_research/output_enhanced_forecast/"
ARIMA_DIR       <- "/scratch/user/s4921048/summer_research/output_darts_arima_benchmark/"
TRAINING_DATA   <- "/scratch/user/s4921048/summer_research/data/new_train_data.csv"
RESULTS_DIR     <- "/scratch/user/s4921048/summer_research/evaluation_result_enhanced_v2_0219/"

MODEL_NAME <- "Transformer"

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

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# HELPER: Interval coverage plot
# ============================================================
plot_interval_coverage_gg <- function(df, facet_vars = NULL, title = "", subtitle = "") {
  p <- ggplot(df, aes(x = interval_range, y = interval_coverage)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey50") +
    geom_point(size = 1.8) +
    labs(
      title = title, subtitle = subtitle,
      x = "Nominal Coverage (interval_range)",
      y = "Empirical Coverage (interval_coverage)"
    ) +
    theme_minimal()
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
# HELPER: Load forecast CSVs from a directory
# ============================================================
load_forecasts <- function(output_dir) {
  consolidated_file <- file.path(output_dir, "all_forecasts.csv")
  if (file.exists(consolidated_file)) {
    cat(sprintf("  Reading: %s\n", consolidated_file))
    fc <- fread(consolidated_file)
  } else {
    files <- list.files(output_dir, pattern = "^forecast_.*\\.csv$", full.names = TRUE)
    files <- files[!grepl("all_forecasts\\.csv$", files)]
    if (length(files) == 0) stop("No forecast files found in ", output_dir)
    cat(sprintf("  Reading %d individual files from %s\n", length(files), output_dir))
    fc <- rbindlist(lapply(files, fread))
  }
  fc[, origin_date     := as.Date(origin_date)]
  fc[, target_end_date := as.Date(target_end_date)]
  return(fc)
}

# ============================================================
# STEP 1: Load observations
# ============================================================
cat("Loading training data (observations)...\n")
actuals <- fread(TRAINING_DATA)
actuals[, date := as.Date(date)]
actuals[, location := COUNTRY_CODES[country_name]]
actuals <- actuals[!is.na(location)]

actuals_for_merge <- unique(
  actuals[, .(target_end_date = date, target, location, observed = value)],
  by = c("target_end_date", "target", "location")
)

# ============================================================
# STEP 2: Load & prepare Transformer forecasts
# ============================================================
cat("\nLoading Transformer forecasts...\n")
tf_raw <- load_forecasts(TRANSFORMER_DIR)
cat(sprintf("  Total Transformer rows: %d\n", nrow(tf_raw)))

quant_tf <- tf_raw[output_type == "quantile"]
quant_tf[, quantile_level := as.numeric(output_type_id)]

merged_tf <- merge(quant_tf, actuals_for_merge,
                   by = c("target_end_date", "target", "location"))
merged_tf <- merged_tf[!is.na(observed)]
cat(sprintf("  Merged Transformer rows: %d\n", nrow(merged_tf)))

forecast_tf <- merged_tf[, .(
  model           = MODEL_NAME,
  forecast_date   = origin_date,
  target_end_date, target, location, horizon,
  observed, predicted = value, quantile_level
)]

fc_tf <- as_forecast_quantile(
  forecast_tf,
  forecast_unit  = c("model", "forecast_date", "target_end_date",
                      "target", "location", "horizon"),
  observed       = "observed",
  predicted      = "predicted",
  quantile_level = "quantile_level"
)
cat("Transformer forecast object created.\n")

# ============================================================
# TASK 1: Interval Coverage (Transformer)
# ============================================================
cat("\n========== TASK 1: Interval Coverage Plot ==========\n")

cov_overall <- as.data.table(get_coverage(fc_tf, by = "model"))
p <- plot_interval_coverage_gg(cov_overall, title = "Interval Coverage: Overall")
ggsave(file.path(RESULTS_DIR, "1_interval_coverage_overall.png"), p, width=10, height=7, dpi=150)

cov_target <- as.data.table(get_coverage(fc_tf, by = c("model", "target")))
p <- plot_interval_coverage_gg(cov_target, facet_vars="target", title="Coverage by Target")
ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_target.png"), p, width=12, height=8, dpi=150)

cov_horizon <- as.data.table(get_coverage(fc_tf, by = c("model", "horizon")))
p <- plot_interval_coverage_gg(cov_horizon, facet_vars="horizon", title="Coverage by Horizon")
ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_horizon.png"), p, width=12, height=8, dpi=150)

cov_loc <- as.data.table(get_coverage(fc_tf, by = c("model", "location")))
p <- plot_interval_coverage_gg(cov_loc, facet_vars="location", title="Coverage by Location")
ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_location.png"), p, width=16, height=10, dpi=150)

cov_hxloc <- as.data.table(get_coverage(fc_tf, by = c("model", "horizon", "location")))
p <- plot_interval_coverage_gg(cov_hxloc, facet_vars=c("horizon","location"),
                                title="Coverage by Horizon Ã Location")
ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_horizon_location.png"), p, width=26, height=10, dpi=150)

TOP_N_LOC <- 12
loc_counts <- cov_hxloc[, .N, by = location][order(-N)]
top_locs <- loc_counts$location[1:min(TOP_N_LOC, nrow(loc_counts))]
p <- plot_interval_coverage_gg(cov_hxloc[location %in% top_locs],
                                facet_vars=c("horizon","location"),
                                title=paste0("Coverage by Horizon Ã Location (Top ", TOP_N_LOC, ")"))
ggsave(file.path(RESULTS_DIR, "1_interval_coverage_by_horizon_location_topN.png"), p, width=20, height=10, dpi=150)

cat("  All coverage plots saved.\n")

# ============================================================
# Compute Transformer Scores
# ============================================================
cat("\nComputing Transformer scores...\n")
scores_tf <- score(fc_tf)
cat(sprintf("  Scored %d forecast instances.\n", nrow(scores_tf)))

# ============================================================
# TASK 2: Score Summary by Forecast Date
# ============================================================
cat("\n========== TASK 2: Summary by Forecast Date ==========\n")

summary_by_date <- summarise_scores(scores_tf, by = c("forecast_date", "target"))
df_date <- as.data.frame(summary_by_date)
df_date$forecast_date <- as.Date(df_date$forecast_date)
fwrite(df_date, file.path(RESULTS_DIR, "2_scores_by_forecast_date.csv"))

p <- ggplot(df_date, aes(x=forecast_date, y=wis, color=target)) +
  geom_line(linewidth=0.8) + geom_point(size=2) +
  labs(title="WIS by Forecast Date", x="Forecast Date", y="WIS", color="Target") +
  theme_minimal() + theme(legend.position="bottom",
                           axis.text.x=element_text(angle=45, hjust=1)) +
  scale_x_date(date_labels="%Y-%m-%d", date_breaks="4 weeks")
ggsave(file.path(RESULTS_DIR, "2_wis_by_forecast_date.png"), p, width=14, height=7, dpi=150)

p <- ggplot(df_date, aes(x=forecast_date, y=wis)) +
  geom_line(color="#2E86AB", linewidth=0.6) + geom_point(color="#2E86AB", size=1) +
  geom_smooth(method="loess", se=TRUE, color="#E94F37", linewidth=0.8) +
  facet_wrap(~target, scales="free_y") +
  labs(title="WIS Trend (LOESS)", x="Forecast Date", y="WIS") +
  theme_minimal() + theme(axis.text.x=element_text(angle=45, hjust=1, size=8))
ggsave(file.path(RESULTS_DIR, "2_wis_trend_faceted.png"), p, width=14, height=10, dpi=150)
cat("  Task 2 saved.\n")

# ============================================================
# TASK 3: Score Summary by Horizon
# ============================================================
cat("\n========== TASK 3: Summary by Horizon ==========\n")

summary_by_horizon <- summarise_scores(scores_tf, by = c("horizon", "target"))
df_horizon <- as.data.frame(summary_by_horizon)
fwrite(df_horizon, file.path(RESULTS_DIR, "3_scores_by_horizon.csv"))

p <- ggplot(df_horizon, aes(x=factor(horizon), y=wis, fill=target)) +
  geom_col(position="dodge", alpha=0.85) +
  labs(title="WIS by Horizon", x="Horizon (weeks)", y="WIS", fill="Target") +
  theme_minimal() + theme(legend.position="bottom")
ggsave(file.path(RESULTS_DIR, "3_wis_by_horizon_bar.png"), p, width=10, height=6, dpi=150)

p <- ggplot(df_horizon, aes(x=horizon, y=wis, color=target, group=target)) +
  geom_line(linewidth=1.2) + geom_point(size=4) +
  labs(title="WIS vs Horizon", x="Horizon (weeks)", y="WIS", color="Target") +
  theme_minimal() + theme(legend.position="bottom") +
  scale_x_continuous(breaks=unique(df_horizon$horizon))
ggsave(file.path(RESULTS_DIR, "3_wis_by_horizon_line.png"), p, width=10, height=6, dpi=150)

df_decomp <- df_horizon %>%
  select(horizon, target, overprediction, underprediction, dispersion) %>%
  pivot_longer(cols=c(overprediction, underprediction, dispersion),
               names_to="component", values_to="score")
p <- ggplot(df_decomp, aes(x=factor(horizon), y=score, fill=component)) +
  geom_col(position="stack") + facet_wrap(~target) +
  scale_fill_manual(values=c("overprediction"="#E94F37",
                              "underprediction"="#2E86AB",
                              "dispersion"="#44AF69")) +
  labs(title="WIS Decomposition", x="Horizon", y="Score", fill="Component") +
  theme_minimal() + theme(legend.position="bottom")
ggsave(file.path(RESULTS_DIR, "3_wis_decomposition.png"), p, width=12, height=8, dpi=150)
cat("  Task 3 saved.\n")

# ============================================================
# Overall Summary
# ============================================================
overall <- summarise_scores(scores_tf)
fwrite(as.data.frame(overall), file.path(RESULTS_DIR, "0_overall_summary.csv"))
cat("\nOverall Transformer summary:\n")
print(overall)

# ============================================================
# TASK 4: Skill Score Heatmap vs ARIMA
# ============================================================
#
# Skill Score = 1 - WIS_transformer / WIS_arima
#   > 0 (green): Transformer better
#   = 0 (yellow): same
#   < 0 (red):   ARIMA better
#
cat("\n========== TASK 4: Skill Score vs ARIMA ==========\n")

# --- 4a: Load ARIMA forecasts ---
cat("Loading ARIMA forecasts...\n")
arima_raw <- load_forecasts(ARIMA_DIR)
cat(sprintf("  Total ARIMA rows: %d\n", nrow(arima_raw)))

quant_arima <- arima_raw[output_type == "quantile"]
quant_arima[, quantile_level := as.numeric(output_type_id)]

merged_arima <- merge(quant_arima, actuals_for_merge,
                      by = c("target_end_date", "target", "location"))
merged_arima <- merged_arima[!is.na(observed)]
cat(sprintf("  Merged ARIMA rows: %d\n", nrow(merged_arima)))

forecast_arima <- merged_arima[, .(
  model           = "ARIMA",
  forecast_date   = origin_date,
  target_end_date, target, location, horizon,
  observed, predicted = value, quantile_level
)]

fc_arima <- as_forecast_quantile(
  forecast_arima,
  forecast_unit  = c("model", "forecast_date", "target_end_date",
                      "target", "location", "horizon"),
  observed       = "observed",
  predicted      = "predicted",
  quantile_level = "quantile_level"
)

# --- 4b: Score ARIMA ---
cat("Scoring ARIMA...\n")
scores_arima <- score(fc_arima)

# --- 4c: Compute WIS per (forecast_date, target, location) for both ---
wis_tf <- as.data.table(
  summarise_scores(scores_tf, by = c("forecast_date", "target", "location"))
)
wis_tf[, forecast_date := as.Date(forecast_date)]

wis_arima <- as.data.table(
  summarise_scores(scores_arima, by = c("forecast_date", "target", "location"))
)
wis_arima[, forecast_date := as.Date(forecast_date)]

# --- 4d: Merge and compute skill score ---
merged_wis <- merge(
  wis_tf[,    .(forecast_date, target, location, wis_tf    = wis)],
  wis_arima[, .(forecast_date, target, location, wis_arima = wis)],
  by = c("forecast_date", "target", "location")
)
merged_wis[, skill_score := 1 - wis_tf / wis_arima]

fwrite(merged_wis, file.path(RESULTS_DIR, "4_skill_scores.csv"))
cat(sprintf("  Skill scores: %d rows\n", nrow(merged_wis)))
cat(sprintf("  Transformer wins: %.1f%% of forecasts\n",
            100 * mean(merged_wis$skill_score > 0, na.rm = TRUE)))
cat(sprintf("  Mean skill score: %.4f\n", mean(merged_wis$skill_score, na.rm = TRUE)))

# --- 4e: Per-target heatmap (location Ã forecast_date) ---
targets <- unique(merged_wis$target)

for (tgt in targets) {
  sub <- merged_wis[target == tgt]
  if (nrow(sub) == 0) next

  vmax <- max(0.5, max(abs(sub$skill_score), na.rm = TRUE))

  # X-axis breaks: every ~4 weeks
  unique_dates <- sort(unique(sub$forecast_date))
  n_dates <- length(unique_dates)
  step <- max(1, n_dates %/% 15)
  break_dates <- unique_dates[seq(1, n_dates, by = step)]

  p <- ggplot(sub, aes(x = forecast_date, y = location, fill = skill_score)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_gradient2(
      low = "#D32F2F", mid = "#FFFFCC", high = "#388E3C",
      midpoint = 0, limits = c(-vmax, vmax),
      name = "Skill Score\n(1 - WIS_tf / WIS_arima)"
    ) +
    scale_x_date(breaks = break_dates, date_labels = "%m-%d") +
    labs(
      title = paste0("Skill Score vs ARIMA - ", tgt),
      subtitle = "Green = Transformer better | Red = ARIMA better",
      x = "Forecast Origin Date", y = "Location"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      axis.text.y = element_text(size = 10),
      panel.grid  = element_blank()
    )

  n_loc <- length(unique(sub$location))
  fname <- paste0("4_skill_heatmap_", gsub(" ", "_", tgt), ".png")
  ggsave(file.path(RESULTS_DIR, fname), p,
         width = max(12, n_dates * 0.28),
         height = max(4, n_loc * 0.7 + 2), dpi = 150)

  cat(sprintf("  %s: mean skill = %.3f  ->  %s\n",
              tgt, mean(sub$skill_score, na.rm = TRUE), fname))
}

# --- 4f: Summary heatmap (location Ã target) ---
summary_skill <- merged_wis[, .(mean_skill = mean(skill_score, na.rm = TRUE)),
                             by = .(location, target)]

vmax_s <- max(0.3, max(abs(summary_skill$mean_skill), na.rm = TRUE))

p_summary <- ggplot(summary_skill, aes(x = target, y = location, fill = mean_skill)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.3f", mean_skill)),
            size = 4, fontface = "bold") +
  scale_fill_gradient2(
    low = "#D32F2F", mid = "#FFFFCC", high = "#388E3C",
    midpoint = 0, limits = c(-vmax_s, vmax_s),
    name = "Mean Skill Score"
  ) +
  labs(
    title = "Mean Skill Score vs ARIMA (Location * Target)",
    subtitle = "Green = Transformer better | Red = ARIMA better",
    x = "Target", y = "Location"
  ) +
  theme_minimal() +
  theme(panel.grid = element_blank(), axis.text = element_text(size = 11))

ggsave(file.path(RESULTS_DIR, "4_skill_heatmap_summary.png"), p_summary,
       width = max(6, length(targets) * 2.5),
       height = max(4, length(unique(summary_skill$location)) * 0.7 + 2), dpi = 150)
cat("  Summary heatmap saved.\n")

# --- 4g: Skill score over time (line plot) ---
skill_time <- merged_wis[, .(mean_skill = mean(skill_score, na.rm = TRUE)),
                          by = .(forecast_date, target)]

p_time <- ggplot(skill_time, aes(x = forecast_date, y = mean_skill, color = target)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.15, linewidth = 0.6) +
  labs(
    title = "Skill Score vs ARIMA Over Time",
    subtitle = "Above 0 = Transformer better | Below 0 = ARIMA better",
    x = "Forecast Date", y = "Mean Skill Score", color = "Target"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_x_date(date_labels = "%Y-%m-%d", date_breaks = "4 weeks")

ggsave(file.path(RESULTS_DIR, "4_skill_over_time.png"), p_time,
       width = 14, height = 7, dpi = 150)
cat("  Skill over time plot saved.\n")

# --- 4h: Overall model skill score heatmap (target Ã time, all locations aggregated) ---
# This shows the WHOLE model's skill at each forecast date, not split by location.
skill_overall <- merged_wis[, .(mean_skill = mean(skill_score, na.rm = TRUE)),
                             by = .(forecast_date, target)]

vmax_o <- max(0.5, max(abs(skill_overall$mean_skill), na.rm = TRUE))
unique_dates_o <- sort(unique(skill_overall$forecast_date))
n_dates_o <- length(unique_dates_o)
step_o <- max(1, n_dates_o %/% 20)
break_dates_o <- unique_dates_o[seq(1, n_dates_o, by = step_o)]

p_overall_heat <- ggplot(skill_overall,
                          aes(x = forecast_date, y = target, fill = mean_skill)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_gradient2(
    low = "#D32F2F", mid = "#FFFFCC", high = "#388E3C",
    midpoint = 0, limits = c(-vmax_o, vmax_o),
    name = "Skill Score\n(1 - WIS_tf / WIS_arima)"
  ) +
  scale_x_date(breaks = break_dates_o, date_labels = "%m-%d") +
  labs(
    title = "Overall Model Skill Score vs ARIMA Over Time",
    subtitle = "Aggregated across all locations | Green = Transformer better | Red = ARIMA better",
    x = "Forecast Origin Date",
    y = "Target"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 12),
    panel.grid  = element_blank()
  )

ggsave(file.path(RESULTS_DIR, "4_skill_heatmap_overall_time.png"), p_overall_heat,
       width = max(14, n_dates_o * 0.28),
       height = max(3, length(unique(skill_overall$target)) * 1.2 + 2), dpi = 150)
cat("  Overall model skill heatmap saved.\n")

# --- 4i: Overall ARIMA summary for comparison ---
overall_arima <- summarise_scores(scores_arima)
cat("\nOverall ARIMA summary:\n")
print(overall_arima)

cat("\n  WIS comparison:\n")
cat(sprintf("    Transformer mean WIS: %.4f\n", as.data.frame(overall)$wis))
cat(sprintf("    ARIMA mean WIS:       %.4f\n", as.data.frame(overall_arima)$wis))
cat(sprintf("    Overall skill score:  %.4f\n",
            1 - as.data.frame(overall)$wis / as.data.frame(overall_arima)$wis))

# ============================================================
# COMPLETE
# ============================================================
cat("\n")
cat(paste(rep("=", 60), collapse = ""))
cat("\n  Analysis Complete!\n")
cat(paste(rep("=", 60), collapse = ""))
cat(sprintf("\n\nAll results saved to: %s\n", RESULTS_DIR))
cat("\nGenerated files:\n")
cat("  - 0_overall_summary.csv\n")
cat("  - 1_interval_coverage_*.png              (Transformer calibration)\n")
cat("  - 2_scores_by_forecast_date.csv / .png   (WIS over time)\n")
cat("  - 3_scores_by_horizon.csv / .png         (WIS by horizon)\n")
cat("  - 4_skill_scores.csv                     (raw skill scores)\n")
cat("  - 4_skill_heatmap_{target}.png           (per-target: location Ã date)\n")
cat("  - 4_skill_heatmap_overall_time.png       (overall: target Ã date, all locations)\n")
cat("  - 4_skill_heatmap_summary.png            (summary: location Ã target)\n")
cat("  - 4_skill_over_time.png                  (skill trend line)\n")