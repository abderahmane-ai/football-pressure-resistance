# Pressure Resistance Score (PRS)

A research-grade Bayesian hierarchical framework for measuring football player composure under pressure, leveraging StatsBomb 360 freeze-frame data.

## Overview

Traditional metrics evaluate passes under pressure but fail to account for spatial context (distance to opponents, angular coverage, passing lane geometry). The **Pressure Resistance Score (PRS)** uses a **Hierarchical Beta Hurdle Model** to simultaneously evaluate *Ball Security* (possession retention) and *Value Retention* (Expected Value preserved) during tight-pressure duels. Player scores are adjusted for opponent quality, team style, competition context, and fine-grained positional roles.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hurdle Model** (two-component) | Separates the binary "did they keep it?" from the continuous "how dangerous was the action?" — avoids the safe-pass bias of single metrics |
| **VAEP Valuation Model** | Uses LightGBM scoring and conceding classifiers trained on all match events to replace discrete xT with a net action value accounting for both offensive threat and defensive turnover risk |
| **Correlated Player Effects** | Models ball security ($\theta_{succ}$) and value retention ($\theta_{val}$) jointly using a multivariate LKJ Cholesky covariance prior to capture player profile correlations |
| **Team + Opponent + Comp Effects** | Removes environmental and tactical system noise (e.g. Pep's possession structures) so $\theta_{player}$ reflects raw composure, not schedule strength or team bias |
| **B-Spline Coordinates** | Expands raw locations and opponent distance into non-linear spline bases (degree 3, 5 knots) to map non-linear threat gradients |
| **6-Group Positions (CB/FB/DM/CM/W/CF)** | Distinguishes positional demands to avoid comparing central defenders with strikers |
| **Non-centered parameterisation** | All random effects sampled as `θ_raw ~ N(0,1)` then scaled, ensuring HMC sampler navigates hierarchical funnels without divergences |
| **Parallelised data build** | `ThreadPoolExecutor` across matches; StatsBomb API calls (I/O) and NumPy/Shapely operations natively release the GIL for true parallelism without process-serialization overhead |
| **Out-of-sample calibration & ECE** | Assesses probability calibration using Expected Calibration Error (ECE) and out-of-sample correlation metrics |

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
    ├── common.py                    # Shared is_valid_loc(), event type string constants
    ├── paths.py                     # ModelPaths dataclass — single source of truth for artifact paths
    ├── data/
    │   ├── loader.py                # StatsBomb API caching + I/O
    │   ├── pairing.py               # Links Pressure → ball-carrier events + 360 frames
    │   ├── lineups.py               # Position groups and goalkeeper identifier filters
    │   ├── events.py                # Pipeline match worker + score diff + xT calculators
    │   ├── writer.py                # Parquet file IO and data hashing/provenance
    │   ├── builder.py               # Full pipeline orchestrator and facade interface
    │   └── validation.py            # Explicit data validation contracts
    ├── features/
    │   ├── geometry.py              # Gaussian pitch control, xT grid, Voronoi, angular span
    │   ├── spatial.py               # Freeze-frame -> 41-feature vector (with B-splines)
    │   ├── vaep.py                  # VAEP model: state extraction, labels, predictions
    │   └── train_vaep.py            # LightGBM training script for scoring & conceding classifiers
    ├── models/
    │   ├── posterior.py             # PosteriorContext — shared access for all analysis modules
    │   ├── bayesian.py              # Joint Hurdle model (LKJ prior) + MCMC sampling
    │   ├── inference.py             # Posterior → leaderboard + scenario analysis
    │   └── validation.py            # Calibration metrics + ECE + residual correlations
    └── visualization/
        ├── interpretability.py      # Variance decomposition, ICE curves, marginal effects, SNR
        └── plots.py                 # Publication-ready figures (leaderboard, calibration)
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

## Feature Vector (41 features)

| Feature | Description |
|---------|-------------|
| `dist_nearest_opp_spline_0..5` | Non-linear B-spline expansion of distance to nearest opponent (5 knots) |
| `bc_x_spline_0..5` | Non-linear B-spline expansion of ball carrier X coordinate |
| `bc_y_spline_0..5` | Non-linear B-spline expansion of ball carrier Y coordinate |
| `dist_2nd_nearest_opp` | Distance to 2nd nearest opponent |
| `opps_within_1yd` | Count of opponents within 1 yard |
| `opps_within_2yd` | Count of opponents within 2 yards |
| `opps_within_4yd` | Count of opponents within 4 yards |
| `angle_nearest_opp` | Angle of nearest opponent relative to goal direction |
| `coverage_arc` | Total angular span blocked by opponents within 3 yds (trigonometric) |
| `voronoi_area` | Shapely Voronoi cell area for ball carrier (sq yds), clipped to pitch |
| `n_free_teammates` | Teammates with no opponent within `clear_pass_distance` (2 yds) |
| `max_free_triangle_area` | Largest triangle area formed by ball carrier + 2 free teammates |
| `dist_nearest_free_teammate` | Distance to nearest free teammate |
| `angle_nearest_free_teammate` | Angle to nearest free teammate relative to goal |
| `pitch_control` | Gaussian influence ratio at ball-carrier location ∈ [-1, 1] |
| `opp_density_5yd` | Count of opponents within 5 yards |
| `has_progressive_option` | Binary: ∃ free teammate in higher-xT zone with unblocked lane |
| `xt_value` | Karun Singh xT (fallback value before VAEP lookup) |
| `game_state_diff` | Score differential (carrier team − opponent) at event time |
| `minutes_elapsed` | Match minute |
| `match_period` | StatsBomb period number |
| `recent_pressures` | Count of pressured touches in player's last 5 carrier events |
| `counter_press` | Binary: 1.0 if the linked pressure was a counter-press event |
| `pass_height_ground` | Binary: 1.0 if Pass height category is Ground |
| `pass_height_low` | Binary: 1.0 if Pass height category is Low |
| `pass_height_high` | Binary: 1.0 if Pass height category is High |

---

## Model Specification

### Ball Security (Logistic / Bernoulli)
$$\text{logit}(p_i) = \alpha_{succ} + X_{i,\text{global}}\beta_{\text{global},succ} + X_{i,\text{pos\_spec}}\beta_{\text{pos}[i],succ} + \gamma_{pos,succ} + \theta_{player,succ} + \delta_{opp,succ} + \zeta_{comp,succ} + \eta_{team,succ}$$
$$Y_{success} \sim \text{Bernoulli}(p_i)$$

### Value Retention (Beta Regression, successes only)
$$\text{logit}(\mu_i) = \alpha_{val} + X_{i,\text{global}}\beta_{\text{global},val} + X_{i,\text{pos\_spec}}\beta_{\text{pos}[i],val} + \gamma_{pos,val} + \theta_{player,val} + \delta_{opp,val} + \zeta_{comp,val} + \eta_{team,val}$$
$$V_{scaled} \sim \text{Beta}(\mu_i \cdot \kappa,\ (1-\mu_i) \cdot \kappa), \quad \kappa \sim \text{Exponential}(0.1)$$

### Correlated Player Effects ($\theta_{player,succ}$, $\theta_{player,val}$)
To capture joint dependencies between player security and value retention traits, we specify a multivariate normal prior:
$$\begin{pmatrix} \theta_{player,succ} \\ \theta_{player,val} \end{pmatrix} \sim \text{MvNormal}\left( \mathbf{0},\ \Sigma_{\theta} \right), \quad \Sigma_{\theta} = \mathbf{D} \mathbf{R} \mathbf{D}$$
Where $\mathbf{R} \sim \text{LKJCholeskyCov}(\eta=2.0)$ and $\mathbf{D}$ is the diagonal scale matrix.

### Priors (all non-centered)
| Parameter | Prior | Role |
|-----------|-------|------|
| `alpha_succ/val` | `Normal(0, 1.5)` | Global intercept |
| `beta_succ/val` | `Normal(0, 1.0)` | Feature coefficients (B-spline + spatial + contextual) |
| `gamma_pos` | `Normal(0, 1.0)` | Positional group fixed effect (CB/FB/DM/CM/W/CF) |
| `theta_chol` | `LKJCholeskyCov(eta=2.0, sd_dist=Exponential(1.0))` | Cholesky covariance factor for player effects |
| `sigma_opp/comp/team` | `Exponential(1.0)` | Spread for opponent, competition, and team random effects |
| `delta_opp / zeta_comp / eta_team` | `raw * sigma` | Opponent, competition, and team-level random effects |
| `kappa` | `Exponential(0.1)` | Beta concentration (mean=10) |

---

## Setup & Installation

Requires Python 3.10+.

```bash
pip install -r requirements.lock
```

For hardware-accelerated MCMC (strongly recommended on Apple Silicon):
```bash
pip install -e ".[accelerated]"
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
This runs the full pipeline (including VAEP model training, dataset building, Bayesian modeling, leaderboard extraction, calibration checking, and residual correlations) sequentially across folds.

### Option B — Single Run
```bash
# 1. Train VAEP scoring and conceding LightGBM classifiers
python3 -m src.features.train_vaep

# 2. Download, filter, and build feature dataset (parallelised)
python3 -m src.data.builder

# 3. Fit hierarchical Hurdle model (MCMC)
python3 -m src.models.bayesian

# 4. Extract leaderboard + scenario profiles
python3 -m src.models.inference

# 5. Validate via out-of-sample residual correlation and ECE calibration
python3 -m src.models.validation

# 6. Variance decomposition + marginal effects + SNR
python3 -m src.visualization.interpretability

# 7. Generate publication figures (Calibration, Leaderboard, Coefficient spans)
python3 -m src.visualization.plots
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRS_HOLDOUT` | `Euro_2020` | Competition held out during cross-validation |
| `PRS_FORCE_RETRAIN` | (unset) | Set to `1` to force retrain VAEP models and MCMC chains |
| `PRS_FORCE_REBUILD_DATA` | (unset) | Set to `1` to force rebuild all feature datasets |
| `PRS_ALLOW_CPU` | (unset) | Set to `1` to run MCMC on CPU (warning: very slow) |

Holdout example:
```bash
export PRS_HOLDOUT=World_Cup_2022
python3 -m src.features.train_vaep
python3 -m src.data.builder
python3 -m src.models.bayesian
python3 -m src.models.inference
python3 -m src.models.validation
```

---

## Competitions (StatsBomb 360 Open Data)

| Key | Competition | Season | comp_id | season_id |
|-----|------------|--------|---------|-----------|
| `Euro_2020` | UEFA Euro | 2020 | 55 | 43 |
| `Euro_2024` | UEFA Euro | 2024 | 55 | 282 |
| `World_Cup_2022` | FIFA World Cup | 2022 | 43 | 106 |
| `Bundesliga_2024` | Bundesliga | 2023/24 | 9 | 281 |
| `MLS_2023` | Major League Soccer | 2023 | 44 | 107 |

---

## Output Artifacts

| Path | Description |
|------|-------------|
| `outputs/tables/prs_leaderboard_{holdout}.csv` | Player Ball Security, Value Retention, PRS, HDI, best scenario |
| `outputs/tables/feature_importance.csv` | Standardized β coefficients with 90% HDI for both sub-models |
| `outputs/tables/variance_decomposition.csv` | Variance split: Player Skill / Team Style / Opp Quality / Competition / Spatial Features |
| `outputs/tables/holdout_correlation_data_{holdout}.csv` | Per-player training PRS vs holdout residuals |
| `outputs/tables/holdout_metrics_{holdout}.csv` | Pearson r, p-value, Spearman r, AUC, ECE calibration error, n_players |
| `outputs/tables/calibration_curve_{holdout}.csv` | Observed vs predicted probabilities for calibration validation |
| `outputs/tables/snr_decomposition.csv` | Signal-to-Noise Ratio (SNR) for individual skill features |
| `outputs/tables/kfold_results.csv` | Aggregate 4-fold validation results |
| `outputs/tables/marginal_dist.csv` | Population marginal expected value by opponent distance (evaluated via B-splines) |
| `outputs/tables/marginal_arc.csv` | Population marginal expected value by opponent coverage arc |
| `outputs/tables/ice_curves.csv` | Individual Conditional Expectation (ICE) curves |
| `outputs/figures/1_leaderboard_2D.png` | Ball Security vs Value Retention scatter (top 30) |
| `outputs/figures/2_feature_importance.png` | Feature coefficient bar chart (both sub-models) |
| `outputs/figures/3_marginal_dist.png` | Population marginal effect — nearest opponent distance (spline basis) |
| `outputs/figures/3_marginal_arc.png` | Population marginal effect — coverage arc |
| `outputs/figures/4_calibration.png` | Reliability diagram (Perfect vs Observed success probability) |
| `outputs/figures/7_stability_scatter.png` | Training PRS vs holdout residuals |
| `outputs/model_traces/pooled_trace_{holdout}.nc` | Full MCMC posterior (NetCDF) |
| `outputs/model_traces/pooled_scaler_{holdout}.pkl` | Fitted StandardScaler + feature list + spline transformers + scaling constants |
| `outputs/model_traces/pooled_mappings_{holdout}.pkl` | Player / position / competition / team index↔label mappings |
