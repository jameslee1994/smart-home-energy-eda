# Smart-Home Appliance Energy Analytics

> What signals best predict appliance energy use in a connected home — and what should a smart-home product team do with that information?

## Overview

This project analyzes 4.5 months of 10-minute smart-home sensor data (19,735 records) from the UCI Appliances Energy Prediction dataset. The goal is to identify which signals — indoor temperature/humidity sensors, outdoor weather, and time-of-day context — most strongly drive appliance energy consumption, and to translate the findings into concrete product recommendations a smart-home platform (think Vivint, Ecobee, Google Nest) could ship as personalization features.

## Tools used

- **Python** (pandas, numpy, scikit-learn)
- **Visualization:** matplotlib, seaborn
- **Notebooks:** plain `.py` scripts for reproducibility
- **Git/GitHub** for version control and portfolio hosting

## Data source

- **Dataset:** [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) — UCI Machine Learning Repository
- **Citation:** Candanedo, Feldheim & Deramaix (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 140, 81–97.
- **Size:** 19,735 rows × 29 columns (~12 MB CSV)
- **Time range:** January 11 – May 27, 2016, sampled every 10 minutes
- **Coverage:** appliance energy use (target), lights, nine indoor temperature/humidity sensor pairs, six outdoor weather variables, and two intentionally injected random noise columns

## Key findings

**1. Two daily peaks — evening dominates.** A smaller morning peak at 06:00–08:00 and a much larger evening peak at 17:00–20:00. Personalization features should target the evening band.

![Average appliance energy by hour of day](visualizations/02_hourly_pattern.png)

**2. Time-of-day is the single most powerful predictor.** RandomForest models built on indoor sensors only, weather only, and the full feature set all land within R² = 0.52–0.54 — because *all three* include hour-of-day.

**3. The most-correlated sensor (Pearson) is not the most-important sensor (RandomForest).** Ranking the sensor mesh by simple correlation will mis-prioritize it.

![RandomForest feature importance](visualizations/10_feature_importance.png)

**4. Indoor sensors and outdoor weather carry overlapping signal**, not strictly complementary signal. Both should be used, but the team shouldn't expect combined-features to outperform either subset by a large margin.

**5. High-consumption episodes** (top 5% by Wh) cluster in the 17:00–20:00 evening band and are slightly more common on weekdays. A "high-use window approaching" prediction is plausible from sensor state alone.

![Hour × day-of-week heatmap](visualizations/11_hour_dow_heatmap.png)

**6. Validation passes:** the two intentionally injected random columns (`rv1`, `rv2`) correctly rank near the bottom of feature importance — the analysis pipeline is honest about what matters.

Full chart-by-chart write-up in [`report/report.md`](report/report.md).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── energydata_complete.csv     # raw (UCI download)
│   └── energydata_clean.csv        # cleaned + time features
├── notebooks/
│   ├── 01_cleaning_and_eda.py      # Step 2
│   └── 02_analysis.py              # Step 3
├── visualizations/                 # 13 PNG charts (Figures 1–13)
└── report/
    └── report.md                   # full write-up
```

## How to reproduce

```bash
git clone https://github.com/jameslee1994/smart-home-energy-eda.git
cd smart-home-energy-eda
pip install -r requirements.txt
python notebooks/01_cleaning_and_eda.py
python notebooks/02_analysis.py
```

## Product recommendations (for a smart-home product team)

The headline of the analysis isn't a model accuracy number; it's six concrete recommendations a product team can act on:

1. **Time-of-day context is non-negotiable** — start every personalization model with hour + day-of-week.
2. **Build features for the 17:00–20:00 evening band** — that's where the energy and the high-consumption episodes are.
3. **Weight sensors by RandomForest importance, not Pearson correlation** — they disagree, and the disagreement matters.
4. **Pull weather in, but don't over-invest** — it overlaps with indoor sensor signal.
5. **Treat lights and appliances as separate signals** — they're only weakly correlated.
6. **"High-consumption window approaching" is a buildable feature** today — strong basis for proactive alerts, automations, and tailored recommendations.

Full details in [`report/report.md`](report/report.md).

## Author

**James Lee** — Senior Product Manager, AI
[LinkedIn](https://www.linkedin.com/in/jim-lee-781602) · jimabout@gmail.com
