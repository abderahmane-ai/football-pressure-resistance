# Pressure Resistance Score (PRS)

A production-grade Bayesian hierarchical framework for evaluating football players' composure under pressure, leveraging StatsBomb 360 freeze-frame data.

## Overview
Traditional metrics evaluate passes under pressure but fail to account for the spatial context (e.g., distance to opponents, density, passing angles). The **Pressure Resistance Score (PRS)** uses a **Bayesian Beta Regression Model** to predict the *Expected Threat (xT) preserved* by a player during high-pressure duals. 

### Key Innovations:
1. **Value-Weighted Outcomes (xT):** We moved from binary success metrics to continuous expected value (Expected Threat). The model evaluates how much offensive value a player preserves when pressured, penalizing safe back-passes that lose ground and rewarding progressive escapes.
2. **Tight-Pressure Filtering:** Only events where the nearest opponent is within **5 yards** are considered, isolating genuine high-pressure duals and removing "token" pressure.
3. **Role-Adjusted Fixed Effects:** The model incorporates Position-Group fixed effects (Defender, Midfielder, Forward). PRS therefore measures "Above-Replacement" composure *within a player's specific tactical role*, ensuring midfielders are compared to midfielders.
4. **Principled Cross-Validation:** Player performance is correlated not just on raw scores, but on their out-of-sample *value-preserved residuals* across different tournaments, proving PRS is a highly stable, scoutable trait.

## Architecture

* **Data Engineering (`src/build_dataset.py`):** Extracts spatial features (Voronoi area, coverage arcs, nearest opponents) from 360° freeze-frames, assigns xT values, excludes goalkeepers, and maps position groups.
* **Bayesian Model (`src/bayesian_model.py`):** A pooled hierarchical Beta regression.
  * **Likelihood:** `Beta(mu * kappa, (1-mu) * kappa)`
  * **Fixed Effects:** Spatial features (Beta) and Position Groups (Gamma).
  * **Random Effects:** Player Skill ($\theta$, the PRS), Competition context, and Opponent Team defensive quality.
* **Interpretability (`src/interpretability.py`):** Generates Individual Conditional Expectation (ICE) curves, Marginal Effects, Counterfactuals, and Bayesian Feature Importance. 
* **Validation (`src/cross_validation.py`):** Calculates out-of-sample stability correlations on a held-out tournament dataset.

## Setup & Installation

Ensure you have Python 3.10+ installed. Install the necessary dependencies:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn pymc arviz statsbombpy tqdm
```
*(If you are on macOS or Linux, ensure you have a C-compiler installed for PyMC's backend, though `numpyro` is highly recommended for faster sampling).*

## Running the Pipeline

Execute the pipeline sequentially from the project root directory:

```bash
# 1. Download data, apply tight-pressure filters, and engineer features
python3 -m src.data.builder

# 2. Fit the hierarchical Beta regression model (MCMC sampling)
python3 -m src.models.bayesian

# 3. Extract player PRS, calculate HDI, and categorize 'Best Under' scenarios
python3 -m src.models.inference

# 4. Validate model via residual-based cross-tournament correlation
python3 -m src.models.validation

# 5. Extract variance decomposition and population marginal effects
python3 -m src.visualization.interpretability

# 6. Generate publication-ready visualizations
python3 -m src.visualization.plots
```

## Output Artifacts
* **`outputs/tables/prs_leaderboard.csv`**: The primary output containing each player's PRS, 90% HDI, and specialized composure profile (e.g., "Best Under: Front_Tight").
* **`outputs/figures/`**: Contains visual insights including marginal threat curves, feature importance bar charts, the stability scatter plot, and the top 20 player leaderboard.
* **`outputs/model_traces/`**: Contains the raw NetCDF trace files from the PyMC sampler for custom downstream analysis or extending the model without retraining.
