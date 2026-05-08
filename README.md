# Forecasting Short-Term Port Congestion Severity

MSc Business Administration and Data Science Thesis

Authors:
- [Maxwell Bernard](https://www.linkedin.com/in/maxwell-bernard/)
- [Johan Schommartz](https://www.linkedin.com/in/johan-henri-schommartz-90b445200/)

Supervisor: [Günter Prockl](https://www.cbs.dk/en/research/departments/department-digitalisation/gunter-prockl)

End-to-end open-source pipeline that forecasts **severe port congestion** (the weekly 95th-percentile of vessel anchorage waiting time) at four US ports (Los Angeles/Long Beach, New York/New Jeey, Houston, Port of Virginia) over 2022–2025. Continuous severity forecasts are then calibrated into severe-congestion risk probabilities, with thresholds anchored in twoexpert practitioner interviews.

The framework combines **62 AIS-derived port-behaviour features** with **six macroeconomic indicators**, evaluated across XGBoost, GRU, and SARIMAX model families, split by vessel type (container vs bulk carrier).

## Pipeline

Data flows top to bottom. Each stage's output is the next stage's input.

| Stage | Code | Purpose |
|---|---|---|
| 1. Port bounding boxes | `bbox_creation/port_bbox.py` | Build per-port bboxes from NOAA ENC + OSM features; drives the AIS download filter. |
| 2. Raw AIS | `ais_data/filtered_parquet/{2022..2025}/` | MarineCadastre AIS pings, pre-filtered to the four port bboxes. |
| 3. Clean | `notebooks/cleaning_pipeline.py` (`run_cleaning.py`) | Merge yearly parquets, normalise schema, restrict to the four ports and to container / bulk-carrier vessels, attach GISIS vessel particulars. |
| 4. Standardise | `notebooks/standardization_pipeline.py` (`run_standardization.py`) | Assign trajectory segment IDs (one continuous observation period per vessel × port). |
| 5. Feature engineering | `notebooks/feature_engineering_pipeline.py` (`run_feature_engineering.py`) | Stop detection, zone classification (anchorage / berth / approach), visit-lifecycle reconstruction, waiting-time computation, weekly p95 target, and 62 AIS features over a 7-day rolling window. |
| 6. Macro join | `src/macro_join_pipeline.py` | Join six macro indicators with publication-lag offsets (prevents look-ahead leakage). |
| 7. Modelling | `notebooks_models/` | Forecasting experiments (see below). |

Vessel-particulars helpers live in `src/gisis_lookup.py` and `src/gisis_cache_stats.py`.

## Final modelling tables

`final_dataframes/` holds the model inputs (one row per port × week):

- `container.csv` — container vessels
- `bulk_carrier.csv` — bulk carriers
- `all_cargo.csv` — pooled, used by the general model

Targets: `target_p95_waiting_{1,2}w` (forecast horizons of 1 and 2 weeks; some
scripts also include 4w).

## Models

In `notebooks_models/`, organised as a generalisation → specialisation
progression:

| File | Model | Scope |
|---|---|---|
| `00_xgboost_combined.py` | XGBoost | General — both vessel types pooled |
| `01_sarimax_container_vessel.py` | SARIMAX | Container, per port |
| `02_xgboost_container_vessel.py` | XGBoost | **Main container model** — global panel, all ports |
| `03_gru_container_vessel.py` | GRU | Container, global panel |
| `04_xgboost_bulk_carrier.py` | XGBoost | **Main bulk model** — global panel |
| `06_sarimax_bulk_carrier.py` | SARIMAX | Bulk, per port |
| `08_gru_bulk_carrier.py` | GRU | Bulk, global panel |
| `10_xgboost_perport_container_vessel.py` | XGBoost | Container, per port |
| `11_xgboost_perport_bulk_carrier.py` | XGBoost | Bulk, per port |
| `13_risk_probabilities.py` | Platt-scaling calibration | Continuous severity → severe-congestion risk probability |
| `14_xgboost_houston_bulk_specialized.py` | XGBoost | Houston bulk only — port-specialised |

Each model script runs five feature specifications: M0 seasonal-naïve baseline,
M1 lag-only, M2 macro-only, M3 AIS-only, M4 AIS + macro.

## Headline results

- **Container** — adding macro indicators raised global XGBoost R² from 0.605
  to 0.655 at 1 week and 0.571 to 0.633 at 2 weeks.
- **Bulk** — XGBoost performed best on AIS features alone (R² 0.488 at 2 weeks);
  macro indicators degraded accuracy.
- **Feature importance** differs sharply by vessel type: container forecasts
  depend most on the ratio of waiting vessels to vessels at berth; bulk
  forecasts depend most on the port's two-month average severe-congestion level.

## Other directories

- `raw_macro_data/` — source CSVs for the macro indicators (Bloomberg, US
  Census Bureau).
- `zone_maps/` — port zone visualisations on satellite basemaps.
- `images/raw_bbox/` — port bbox visualisations.
- `appendix/` — supplementary figures (cross-port correlation, etc.).

NOTE: 
Pipeline scripts are written in jupytext `# %%` cell format — they run as
plain Python and also open as notebooks in Jupyter / VS Code.
