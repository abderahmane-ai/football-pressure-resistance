# Pressure Resistance Score (PRS)

A production-grade Bayesian hierarchical framework for evaluating football players' composure under pressure, leveraging StatsBomb 360 freeze-frame data.

## Overview
Traditional metrics evaluate passes under pressure but fail to account for the spatial context (e.g., distance to opponents, density, passing angles). The **Pressure Resistance Score (PRS)** uses a **Zero-Inflated Bayesian Hurdle Model** to predict both the *Turnover Risk* (Ball Security) and the *Value Retention* (Expected Threat preserved) by a player during high-pressure duals. 

### Key Innovations:
1. **Dual-Axis Valuation (Hurdle Model):** We separated the binary outcome (Did they keep the ball?) from the continuous value (How dangerous was the action?). The model evaluates "Ball Security" independently from "Value Retention", identifying players who are safe possession retainers versus high-risk, high-reward progressors.
2. **Domain-Accurate Spatial Engineering:** Features include Gaussian Pitch Control, which models probabilistic area-ownership using continuous distance-decay functions, and Lane-Aware Progressive Options, ensuring teammates are only considered viable options if the passing lane is unblocked.
3. **Tight-Pressure Filtering:** Only events where the nearest opponent is within a defined **tight pressure radius** (e.g., 5 yards, driven by spatial configuration) are considered, isolating genuine high-pressure duals and removing "token" pressure.
4. **Role-Adjusted Fixed Effects:** The model incorporates Position-Group fixed effects (Defender, Midfielder, Forward) dynamically assigned via spatial clustering. PRS therefore measures "Above-Replacement" composure *within a player's specific tactical role*.
5. **Principled Cross-Validation:** Player performance is correlated not just on raw scores, but on their out-of-sample *expected value residuals* across different tournaments, proving PRS is a highly stable, scoutable trait.

## Architecture

* **Data Engineering (`src/data/builder.py`):** Extracts spatial features (Voronoi area, coverage arcs, Gaussian Pitch Control, unblocked passing lanes) from 360° freeze-frames, assigns intended xT values, and maps dynamic position groups.
* **Bayesian Model (`src/models/bayesian.py`):** A pooled hierarchical Zero-Inflated Beta (Hurdle) regression.
  * **Turnover Risk (Logistic):** `Bernoulli(p)`
  * **Value Retention (Beta):** `Beta(mu * kappa, (1-mu) * kappa)`
  * **Random Effects:** Player Skill ($\theta$), Competition context, and Opponent Team defensive quality estimated for both model components.
* **Interpretability (`src/visualization/interpretability.py`):** Generates Individual Conditional Expectation (ICE) curves, Population Marginal Effects, and Covariance-Aware Variance Decomposition.
* **Validation (`src/models/validation.py`):** Calculates out-of-sample stability correlations on a held-out tournament dataset using fully vectorized tensor operations.

## Setup & Installation

Ensure you have Python 3.10+ installed. Install the necessary dependencies:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn pymc arviz statsbombpy tqdm shapely
```
*(If you are on macOS or Linux, ensure you have a C-compiler installed for PyMC's backend, though `numpyro` is highly recommended for faster sampling).*

## Running the Pipeline

Execute the pipeline sequentially from the project root directory:

```bash
# 1. Download data, apply tight-pressure filters, and engineer features
python3 -m src.data.builder

# 2. Fit the hierarchical Zero-Inflated Beta regression model (MCMC sampling)
python3 -m src.models.bayesian

# 3. Extract player dual metrics, calculate HDI, and categorize 'Best Under' scenarios
python3 -m src.models.inference

# 4. Validate model via residual-based cross-tournament correlation
python3 -m src.models.validation

# 5. Extract variance decomposition and population marginal effects
python3 -m src.visualization.interpretability

# 6. Generate publication-ready visualizations
python3 -m src.visualization.plots
```

## Output Artifacts
* **`outputs/tables/prs_leaderboard.csv`**: The primary output containing each player's Ball Security score, Value Retention score, Combined PRS, and specialized composure profile.
* **`outputs/figures/`**: Contains visual insights including 2D scatter plots (Turnover Risk vs Value Retention), marginal threat curves, feature importance bar charts, and the stability scatter plot.
* **`outputs/model_traces/`**: Contains the raw NetCDF trace files from the PyMC sampler for custom downstream analysis.
