# Pressure Resistance Score (PRS)

A research-grade Bayesian hierarchical framework for measuring football player composure under pressure, leveraging StatsBomb 360 freeze-frame data.

## Overview

Traditional metrics evaluate passes under pressure but fail to account for spatial context (distance to opponents, angular coverage, passing lane geometry). The **Pressure Resistance Score (PRS)** uses a **Hierarchical Beta Hurdle Model** to simultaneously evaluate *Ball Security* (possession retention) and *Value Retention* (Expected Threat preserved) during tight-pressure duels. Player scores are adjusted for opponent quality, competition context, and positional role — isolating each player's true intrinsic composure trait.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hurdle Model** (two-component) | Separates the binary "did they keep it?" from the continuous "how dangerous was the action?" — avoids the safe-pass bias of single metrics |
| **Non-centered parameterisation** | All random effects sampled as `θ_raw ~ N(0,1)` then scaled, ensuring HMC sampler navigates hierarchical funnels without divergences |
| **Opponent + Competition random effects** | Removes environmental noise so θ_player reflects skill, not schedule strength |
| **Tight-pressure filter (≤5 yards)** | Eliminates token pressure events; model only sees genuine close-quarters duels |
| **One row per carrier action** | Collapses multiple defender `Pressure` logs linked to the same action, because freeze-frame geometry already captures pressure density |
| **Raw `bc_x`, `bc_y` coordinates** | Replaces an ordinal zone integer that falsely implied a linear relationship across pitch thirds |
| **Parallelised data build** | `ThreadPoolExecutor` across matches; StatsBomb API calls (I/O) and NumPy/Shapely operations natively release the GIL for true parallelism without process-serialization overhead |
| **Out-of-sample residual correlation** | Strong validation check for whether PRS is stable and transferable rather than in-sample noise |

---

## Architecture

```
pressure_resistance/
├── config.py                        # All constants, paths, MCMC settings
├── run_kfold.py                     # 4-fold cross-validation orchestrator
├── modal/
│   └── modal_run.py                 # Cloud GPU runner via Modal (gitignored)
├── data/
│   ├── xt_grid.json                 # Karun Singh xT grid (loaded dynamically)
│   ├── raw/                         # Cached StatsBomb data
│   └── processed/                   # Built datasets
└── src/
    ├── data/
    │   ├── loader.py                # StatsBomb API caching + I/O
    │   ├── pairing.py               # Links Pressure → ball-carrier events + 360 frames
    │   ├── labels.py                # Defines success for Pass/Carry/Dribble
    │   ├── builder.py               # Full pipeline: game state, parallel feature extraction
    │   └── validation.py            # Explicit data validation contracts
    ├── features/
    │   ├── geometry.py              # Gaussian pitch control, xT grid, Voronoi, angular span
    │   └── spatial.py               # Freeze-frame → 21-feature vector
    ├── models/
    │   ├── bayesian.py              # Hurdle model definition + MCMC sampling
    │   ├── inference.py             # Posterior → leaderboard + scenario analysis
    │   └── validation.py            # Out-of-sample residual correlation
    └── visualization/
        ├── interpretability.py      # Variance decomposition, ICE curves, marginal effects
        └── plots.py                 # Publication-ready figures
```

### Data Validation

The pipeline includes explicit validation contracts to ensure data integrity at each stage:

| Function | Purpose |
|----------|---------|
| `validate_statsbomb_events()` | Checks required columns, nulls, duplicates in raw event data |
| `validate_statsbomb_frames()` | Validates 360 freeze-frame data integrity |
| `validate_model_dataset()` | Ensures processed dataset has required features + valid labels |

Raises `DataValidationError` with actionable messages if contracts are violated.

---

## Feature Vector (21 features)

| Feature | Description |
|---------|-------------|
| `dist_nearest_opp` | Euclidean distance to nearest opponent (yards) |
| `dist_2nd_nearest_opp` | Distance to 2nd nearest opponent |
| `opps_within_1yd` | Count of opponents within 1 yard |
| `opps_within_2yd` | Count of opponents within 2 yards |
| `opps_within_4yd` | Count of opponents within 4 yards |
| `angle_nearest_opp` | Angle of nearest opponent relative to goal direction |
| `coverage_arc` | Total angular span blocked by opponents within 3 yds (trigonometric, not hardcoded) |
| `voronoi_area` | Shapely Voronoi cell area for ball carrier (sq yds), clipped to pitch |
| `n_free_teammates` | Teammates with no opponent within `clear_pass_distance` (2 yds) |
| `max_free_triangle_area` | Largest triangle area formed by ball carrier + 2 free teammates |
| `dist_nearest_free_teammate` | Distance to nearest free teammate |
| `angle_nearest_free_teammate` | Angle to nearest free teammate relative to goal |
| `pitch_control` | Gaussian influence ratio at ball-carrier location ∈ [-1, 1] |
| `opp_density_5yd` | Count of opponents within 5 yards |
| `has_progressive_option` | Binary: ∃ free teammate in higher-xT zone with unblocked lane |
| `xt_value` | Karun Singh xT at ball-carrier location |
| `bc_x`, `bc_y` | Raw pitch coordinates (replaces misleading ordinal zone integer) |
| `game_state_diff` | Score differential (carrier team − opponent) at event time |
| `minutes_elapsed` | Match minute |
| `match_period` | StatsBomb period number (including extra time/shootout periods when present) |

---

## Model Specification

### Ball Security (Logistic / Bernoulli)
$$\text{logit}(p_i) = \alpha_{succ} + X_i\beta_{succ} + \gamma_{pos,succ} + \theta_{player,succ} + \delta_{opp,succ} + \zeta_{comp,succ}$$
$$Y_{success} \sim \text{Bernoulli}(p_i)$$

### Value Retention (Beta Regression, successes only)
$$\text{logit}(\mu_i) = \alpha_{val} + X_i\beta_{val} + \gamma_{pos,val} + \theta_{player,val} + \delta_{opp,val} + \zeta_{comp,val}$$
$$V_{scaled} \sim \text{Beta}(\mu_i \cdot \kappa,\ (1-\mu_i) \cdot \kappa), \quad \kappa \sim \text{Exponential}(0.1)$$

### Priors (all non-centered)
| Parameter | Prior | Role |
|-----------|-------|------|
| `alpha_succ/val` | `Normal(0, 1.5)` | Global intercept |
| `beta_succ/val` | `Normal(0, 1.0)` | Feature coefficients (spatial + contextual) |
| `gamma_pos` | `Normal(0, 1.0)` | Position group fixed effect |
| `sigma_theta` | `Exponential(1.0)` | Player skill spread |
| `theta_player` | `theta_raw * sigma_theta` | Individual player composure |
| `sigma_opp/comp` | `Exponential(1.0)` | Environment noise spread |
| `delta_opp / zeta_comp` | `raw * sigma` | Opponent / competition effects |
| `kappa` | `Exponential(0.1)` | Beta concentration (mean=10) |

---

## Setup & Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

For hardware-accelerated MCMC (strongly recommended on Apple Silicon):
```bash
pip install -r requirements-accelerated.txt
pip install jax-metal        # Mac GPU backend
```

---

## Running the Pipeline

### Option A — Full 4-Fold Cross-Validation (recommended)
```bash
# macOS / Linux
python3 run_kfold.py

# Windows
python run_kfold.py
```
This sequentially rotates through all 4 competitions as holdout and reports aggregate Pearson correlation and AUC.

### Option B — Single Run
```bash
# 1. Download, filter, and build feature dataset (parallelised)
# macOS / Linux: python3 -m src.data.builder
# Windows:      python -m src.data.builder

# 2. Fit hierarchical Hurdle model (MCMC)
# macOS / Linux: python3 -m src.models.bayesian
# Windows:      python -m src.models.bayesian

# 3. Extract leaderboard + scenario profiles
# macOS / Linux: python3 -m src.models.inference
# Windows:      python -m src.models.inference

# 4. Validate via out-of-sample residual correlation
# macOS / Linux: python3 -m src.models.validation
# Windows:      python -m src.models.validation

# 5. Variance decomposition + marginal effects
# macOS / Linux: python3 -m src.visualization.interpretability
# Windows:      python -m src.visualization.interpretability

# 6. Generate publication figures
# macOS / Linux: python3 -m src.visualization.plots
# Windows:      python -m src.visualization.plots
```

### Holdout Competition
Controlled via the `PRS_HOLDOUT` environment variable (default: `Euro_2020`). Must be set before running the full pipeline:
```bash
export PRS_HOLDOUT=World_Cup_2022
# macOS / Linux
python3 -m src.data.builder
python3 -m src.models.bayesian
python3 -m src.models.inference
python3 -m src.models.validation

# Windows (PowerShell)
$env:PRS_HOLDOUT="World_Cup_2022"
python -m src.data.builder
python -m src.models.bayesian
python -m src.models.inference
python -m src.models.validation
```

---

## Competitions (StatsBomb 360 Open Data)

| Key | Competition | Season | comp_id | season_id |
|-----|------------|--------|---------|-----------|
| `Euro_2020` | UEFA Euro | 2020 | 55 | 43 |
| `Euro_2024` | UEFA Euro | 2024 | 55 | 282 |
| `World_Cup_2022` | FIFA World Cup | 2022 | 43 | 106 |
| `Bundesliga_2024` | Bundesliga | 2023/24 | 9 | 281 |

---

## Output Artifacts

| Path | Description |
|------|-------------|
| `outputs/tables/prs_leaderboard.csv` | Player Ball Security, Value Retention, PRS, HDI, best scenario |
| `outputs/tables/feature_importance.csv` | Standardized β coefficients with 90% HDI for both sub-models |
| `outputs/tables/variance_decomposition.csv` | Variance split: Player Skill / Opp Quality / Competition / Spatial Features |
| `outputs/tables/holdout_correlation_data.csv` | Per-player training PRS vs holdout residuals |
| `outputs/tables/holdout_metrics.csv` | Pearson r, p-value, Spearman r (logged only), AUC, n_players |
| `outputs/tables/kfold_results.csv` | Aggregate 4-fold validation results |
| `outputs/tables/marginal_dist.csv` | *(Optional)* Population marginal expected value by opponent distance |
| `outputs/tables/marginal_arc.csv` | *(Optional)* Population marginal expected value by opponent coverage arc |
| `outputs/tables/ice_curves.csv` | *(Optional)* Individual Conditional Expectation curves |
| `outputs/figures/1_leaderboard_2D.png` | Ball Security vs Value Retention scatter (top 30) |
| `outputs/figures/2_feature_importance.png` | Feature coefficient bar chart (both sub-models) |
| `outputs/figures/3_marginal_dist.png` | *(Optional)* Population marginal effect — nearest opponent distance |
| `outputs/figures/3_marginal_arc.png` | *(Optional)* Population marginal effect — coverage arc |
| `outputs/figures/7_stability_scatter.png` | Training PRS vs holdout residuals |
| `outputs/model_traces/pooled_trace.nc` | Full MCMC posterior (NetCDF) |
