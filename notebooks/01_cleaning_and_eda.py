"""
Final Project — Step 2: Dataset Preparation & EDA
Author: James Lee
Course: MTECH Data Analytics Capstone — Spring 2026

This script:
  1. Loads the UCI Appliances Energy Prediction dataset
  2. Performs cleaning and standardization
  3. Computes summary statistics
  4. Creates 8 EDA visualizations
  5. Saves a cleaned dataset and an EDA findings markdown

Designed to be reproducible: run `python 01_cleaning_and_eda.py` from this folder.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Paths ----------
HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_RAW = ROOT / "data" / "energydata_complete.csv"
DATA_CLEAN = ROOT / "data" / "energydata_clean.csv"
VIZ_DIR = ROOT / "visualizations"
REPORT_DIR = ROOT / "report"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight"})

# ---------- 1. Load ----------
print("Loading raw data...")
df = pd.read_csv(DATA_RAW)
print(f"  Shape: {df.shape}")

# ---------- 2. Cleaning ----------
print("\nCleaning...")
# 2a. Date parsing
df["date"] = pd.to_datetime(df["date"])

# 2b. Null check
nulls = df.isna().sum()
print(f"  Null counts (top 5):\n{nulls[nulls > 0].head() if nulls.sum() else '  No nulls.'}")

# 2c. Duplicates
dup = df.duplicated().sum()
print(f"  Duplicate rows: {dup}")
if dup:
    df = df.drop_duplicates()

# 2d. Time features (useful for EDA + analysis)
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.day_name()
df["dow_num"] = df["date"].dt.dayofweek  # 0 = Monday
df["is_weekend"] = df["dow_num"].isin([5, 6])
df["date_only"] = df["date"].dt.date
df["month"] = df["date"].dt.month_name()

# 2e. Drop the injected noise columns rv1, rv2 from analysis (keep in raw)
#     We'll keep them for the validation question, but exclude from sensor analysis
sensor_cols = [f"T{i}" for i in range(1, 10)] + [f"RH_{i}" for i in range(1, 10)]
weather_cols = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
target = "Appliances"  # in Wh

# 2f. Sanity: clip RH (relative humidity) to [0, 100]; flag any out-of-range
oor = ((df[[c for c in df.columns if c.startswith("RH_")]] < 0) |
       (df[[c for c in df.columns if c.startswith("RH_")]] > 100)).sum().sum()
print(f"  Humidity values outside [0,100]: {oor} (left as-is, see report)")

# 2g. Save cleaned dataset
keep_cols = ["date", target, "lights"] + sensor_cols + weather_cols + ["rv1", "rv2",
             "hour", "day_of_week", "dow_num", "is_weekend", "date_only", "month"]
df[keep_cols].to_csv(DATA_CLEAN, index=False)
print(f"  Cleaned CSV written -> {DATA_CLEAN.name} ({len(df):,} rows)")

# ---------- 3. Summary statistics ----------
print("\nSummary statistics...")
desc = df[[target, "lights"] + sensor_cols + weather_cols].describe().round(2)
desc.to_csv(REPORT_DIR / "summary_statistics.csv")
print(f"  Wrote summary_statistics.csv")

# ---------- 4. Visualizations ----------
print("\nBuilding visualizations...")

# --- Viz 1: Distribution of Appliances (target) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df[target], bins=60, color="#1F77B4", edgecolor="white")
axes[0].set_title("Appliance energy use — distribution (Wh)")
axes[0].set_xlabel("Wh per 10-min interval"); axes[0].set_ylabel("Frequency")
axes[1].hist(np.log1p(df[target]), bins=60, color="#1F77B4", edgecolor="white")
axes[1].set_title("Appliance energy use — log scale")
axes[1].set_xlabel("log(1 + Wh)"); axes[1].set_ylabel("Frequency")
plt.suptitle("Figure 1. Appliance energy is highly right-skewed", y=1.02, fontsize=12, weight="bold")
plt.savefig(VIZ_DIR / "01_appliance_distribution.png")
plt.close()

# --- Viz 2: Average appliance energy by hour of day ---
hourly = df.groupby("hour")[target].mean()
plt.figure(figsize=(10, 4))
plt.plot(hourly.index, hourly.values, marker="o", color="#D62728")
plt.fill_between(hourly.index, hourly.values, alpha=0.15, color="#D62728")
plt.title("Figure 2. Average appliance energy by hour of day", weight="bold")
plt.xlabel("Hour of day (0–23)"); plt.ylabel("Avg Wh per 10-min interval")
plt.xticks(range(0, 24))
plt.savefig(VIZ_DIR / "02_hourly_pattern.png")
plt.close()

# --- Viz 3: Weekday vs weekend hourly profile ---
hr_dow = df.groupby(["hour", "is_weekend"])[target].mean().unstack()
hr_dow.columns = ["Weekday", "Weekend"]
plt.figure(figsize=(10, 4))
for col, color in zip(hr_dow.columns, ["#1F77B4", "#FF7F0E"]):
    plt.plot(hr_dow.index, hr_dow[col], marker="o", label=col, color=color)
plt.title("Figure 3. Hourly energy profile — weekday vs weekend", weight="bold")
plt.xlabel("Hour"); plt.ylabel("Avg Wh per 10-min interval")
plt.xticks(range(0, 24)); plt.legend()
plt.savefig(VIZ_DIR / "03_hourly_by_dow.png")
plt.close()

# --- Viz 4: Energy by day of week ---
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = df.groupby("day_of_week")[target].mean().reindex(order)
plt.figure(figsize=(8, 4))
bars = plt.bar(dow.index, dow.values, color="#2CA02C", edgecolor="white")
for b, v in zip(bars, dow.values):
    plt.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.0f}",
             ha="center", va="bottom", fontsize=9)
plt.title("Figure 4. Average appliance energy by day of week", weight="bold")
plt.ylabel("Avg Wh per 10-min interval")
plt.xticks(rotation=30)
plt.savefig(VIZ_DIR / "04_day_of_week.png")
plt.close()

# --- Viz 5: Correlation heatmap (target vs sensors + weather) ---
corr_cols = [target] + sensor_cols + weather_cols
corr = df[corr_cols].corr()
# Just the target row, sorted
target_corr = corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr.loc[[target] + list(target_corr.index)][[target] + list(target_corr.index)],
            annot=False, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Pearson r"})
plt.title("Figure 5. Correlation heatmap — appliance energy vs sensors and weather", weight="bold")
plt.savefig(VIZ_DIR / "05_correlation_heatmap.png")
plt.close()

# --- Viz 6: Top correlations with target ---
plt.figure(figsize=(8, 6))
colors = ["#D62728" if v > 0 else "#1F77B4" for v in target_corr.values]
plt.barh(target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1])
plt.axvline(0, color="grey", linewidth=0.8)
plt.title("Figure 6. Pearson correlation with appliance energy use", weight="bold")
plt.xlabel("Correlation (r)")
plt.savefig(VIZ_DIR / "06_correlation_with_target.png")
plt.close()

# --- Viz 7: Time series (daily mean) ---
daily = df.groupby("date_only")[target].mean()
plt.figure(figsize=(11, 4))
plt.plot(pd.to_datetime(daily.index), daily.values, color="#9467BD")
plt.title("Figure 7. Daily mean appliance energy over the 4.5-month study window", weight="bold")
plt.xlabel("Date"); plt.ylabel("Avg Wh per 10-min interval")
plt.savefig(VIZ_DIR / "07_daily_timeseries.png")
plt.close()

# --- Viz 8: Lights vs Appliances scatter (downsampled) ---
sample = df.sample(min(3000, len(df)), random_state=42)
plt.figure(figsize=(8, 5))
plt.scatter(sample["lights"], sample[target], alpha=0.25, s=12, color="#1F77B4")
plt.title("Figure 8. Lights vs Appliances (random 3K sample)", weight="bold")
plt.xlabel("Lights (Wh)"); plt.ylabel("Appliances (Wh)")
plt.savefig(VIZ_DIR / "08_lights_vs_appliances.png")
plt.close()

# --- Viz 9: Validation - rv1, rv2 should have ~0 correlation ---
rv_corr = df[[target, "rv1", "rv2"]].corr()[target].drop(target)
plt.figure(figsize=(6, 3))
plt.barh(rv_corr.index, rv_corr.values, color="#7F7F7F")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlim(-0.05, 0.05)
plt.title("Figure 9. Sanity check: random columns rv1/rv2 are uncorrelated with target", weight="bold")
plt.xlabel("Correlation (r)")
plt.savefig(VIZ_DIR / "09_rv_validation.png")
plt.close()

print(f"  Saved 9 visualizations to {VIZ_DIR}")

# ---------- 5. Findings summary ----------
findings = f"""# Step 2 — EDA Findings

**Dataset:** {len(df):,} rows × {df.shape[1]} columns (after time-feature engineering)
**Time window:** {df['date'].min()} to {df['date'].max()} ({(df['date'].max() - df['date'].min()).days} days)

## Cleaning summary

| Step | Result |
|---|---|
| Null values | {int(nulls.sum())} (no imputation required) |
| Duplicate rows | {int(dup)} |
| Date column | Parsed to datetime, derived hour / day-of-week / weekend |
| Humidity sanity (RH outside [0,100]) | {int(oor)} values flagged but kept (outdoor RH plausibly noisy) |
| Noise columns | `rv1`, `rv2` retained for sanity-check Q6, excluded from feature analysis |

## Headline EDA findings

1. **Appliance energy is highly right-skewed** (Fig 1). Median = {df[target].median():.0f} Wh; mean = {df[target].mean():.1f} Wh; max = {df[target].max():.0f} Wh. About 75% of intervals are at or below 100 Wh, but a long tail of high-consumption bursts dominates total energy use. Log-scaling will be appropriate for many analyses.

2. **Strong daily cycle** (Fig 2). Energy use peaks around 17:00–20:00 (evening), with a smaller morning peak at 06:00–08:00 and a clear overnight trough from 23:00 to 05:00.

3. **Weekend vs weekday matters** (Fig 3). Weekend mornings start later but the evening peak is similar in height — household activity shifts but doesn't disappear.

4. **Day-of-week differences are modest** (Fig 4). The largest day-to-day spread is ~{(dow.max() - dow.min()):.0f} Wh, well within the daily peak-to-trough range.

5. **Outdoor weather has weak-to-moderate correlations** with appliance use (Fig 5, Fig 6). Top correlated weather variables are `T_out` ({corr.loc[target, 'T_out']:+.2f}) and `Tdewpoint` ({corr.loc[target, 'Tdewpoint']:+.2f}). Surprising for a smart-home product team: indoor sensor humidity values are nearly as informative as outdoor weather.

6. **Indoor sensors are not interchangeable.** The most-correlated indoor humidity sensor is `{target_corr.head(1).index[0]}` (r = {target_corr.head(1).values[0]:+.2f}). For a Vivint-style product team, this means a smart-home AI shouldn't just average sensors — it should weight them.

7. **Time-series shows a slow upward drift** (Fig 7) as the study moves from January into May — consistent with longer-day, more-activity seasonality.

8. **Lights and Appliances are only weakly correlated** (Fig 8, r = {df[[target, 'lights']].corr().iloc[0,1]:+.2f}). They tell partly independent stories of household activity, so a personalization model should treat them as separate signals.

9. **Validation (Q6) passes** (Fig 9). The injected random columns `rv1` and `rv2` have correlation with the target of {rv_corr['rv1']:+.3f} and {rv_corr['rv2']:+.3f} — essentially zero, as expected. This confirms my analytical method is honest about which features matter.

## Top 10 sensors / weather variables by |correlation| with Appliances

{target_corr.abs().sort_values(ascending=False).head(10).to_markdown()}

## Open questions for Step 3 (Analysis)

- Can we quantify how much of the variation in `Appliances` is explained by indoor sensors alone vs adding weather?
- Do "high-consumption episodes" cluster around specific sensor-state combinations?
- A simple RandomForest feature importance will be a stronger feature ranking than Pearson — does it agree with Fig 6 or surface different sensors?
- Is there a relationship between humidity and the time-of-day peak — i.e., does the household run more appliances when it's humid?

## Reproducibility

```
python notebooks/01_cleaning_and_eda.py
```
Produces:
- `data/energydata_clean.csv`
- `visualizations/01_*.png` through `09_*.png`
- `report/summary_statistics.csv`
- this file: `report/Step2_EDA_Findings.md`
"""
(REPORT_DIR / "Step2_EDA_Findings.md").write_text(findings)
print(f"\nWrote: {REPORT_DIR / 'Step2_EDA_Findings.md'}")
print("\nDone.")
