# Methodology: Pressure Resistance Score (PRS)

This document serves as the formal, comprehensive whitepaper detailing the mathematical, conceptual, and analytical framework underpinning the **Pressure Resistance Score (PRS)**. It outlines the paradigm shift from naive success metrics to a dual-axis value framework, the derivation of domain-accurate geometric features from 360-degree freeze-frame data, the architecture of the Bayesian Hierarchical Beta Hurdle Model, and the rigorous out-of-sample validation strategy utilized to evaluate its efficacy.

---

## 1. Theoretical Framework & Motivation

Historically, football analytics has evaluated a player's "pressure resistance" or "composure" by calculating their passing completion percentage while under pressure. However, this binary approach suffers from severe ecological fallacies that limit its utility for elite scouting and tactical analysis:

1. **The Safe-Pass Bias (Survival Bias):** Binary completion metrics treat a 2-yard backward pass to an unmarked center-back exactly the same as a 20-yard line-breaking pass through an opponent's midfield block. Both are logged as a "success" ($Y = 1$). Consequently, players who are highly risk-averse—those who immediately recycle possession backwards the moment they feel pressure—are artificially rewarded. This masks a player's inability to actually progress the ball when constrained.
2. **The Zero-Inflation Problem:** A player under pressure faces two distinct, sequential cognitive challenges: 
   * *Phase 1: Can I physically keep the ball?* (Ball Security / possession retention)
   * *Phase 2: If I keep it, can I execute a dangerous action?* (Value Generation / Offensive Output)
   Standard continuous models collapse these two distinct processes into a single metric, failing to handle the massive spike of exact zeroes (turnovers) in the data. A failed action creates an artificial floor that breaks the assumptions of standard linear regression.
3. **The Spatial Vacuum:** Traditional models largely ignore the surrounding pitch geometry. A player's ability to complete a pass depends heavily on unblocked passing lanes, the angular span of closing opponents, and the dynamic shifting of pitch control. A model without geometric context is evaluating decisions in a vacuum.

The PRS framework resolves these fundamental flaws by migrating to a **Hurdle framework**. This allows us to evaluate *Ball Security* independently from *Expected Threat (xT) Retained*, conditioning both values on the exact spatial geometry of the duel, and extracting the player's true underlying skill via a Bayesian hierarchical model.

---

## 2. Cohort Definition and Data Preprocessing

To ensure the model isolates genuine cognitive and technical challenges, the raw StatsBomb 360 dataset undergoes strict filtering and cohort stratification before entering the modeling pipeline.

### 2.1 Goalkeeper Exclusion
Goalkeepers face fundamentally different pressure geometries than outfield players. They typically receive the ball facing forward with the entire pitch ahead of them, and have the tactical license to play low-risk lateral passes or launch long clearances. In a baseline model, their high completion rate of these low-value passes artificially inflates their apparent composure. Therefore, goalkeepers are entirely excluded from the training and evaluation cohorts.

### 2.2 The Tight-Pressure Filter
Utilizing the continuous $x,y$ coordinates from the 360° freeze-frames, we calculate the Euclidean distance to the nearest opponent, denoted as $d_{min}$, at the exact fraction of a second the ball-carrier receives the ball. 

To remove the statistical noise of "token pressure"—such as a striker casually jogging toward a defender from 8 yards away—we strictly filter the dataset to only include events where:
$$ d_{min} \le 5.0 \text{ yards} $$
This forces the model to evaluate genuine, close-quarters duels, ensuring that a high PRS score translates directly to performance in the most heavily contested areas of the pitch.

### 2.3 Dynamic Positional Stratification
Players are mapped to one of six fine-grained tactical roles derived directly from StatsBomb position IDs: **CB** (Centre-Back), **FB** (Full-Back / Wing-Back), **DM** (Defensive Midfielder), **CM** (Central Midfielder), **W** (Wide Midfielder / Winger), and **CF** (Centre-Forward / Striker). This six-group taxonomy replaces the coarser three-bracket scheme and allows the hierarchical model to set separate baseline priors for roles with genuinely different spatial profiles — a CB in a deep defensive block faces wholly different passing geometry from a CF on the shoulder of the last defender.

For players absent from the formal lineup data (a common edge case in sprawling multi-competition datasets), the system falls back to a coordinate-based imputation: it computes the average $X$ position of all open-play touches (passes, carries, dribbles, shots) and classifies the player based on which third of the pitch they habitually occupied, with a laterality check ($|y - 40| > 16$ yards) to distinguish wide roles from central ones. This ensures every player is evaluated against an *Above-Replacement* baseline anchored to their true positional peers.

---

## 3. Geometric Feature Engineering

For every filtered carrier action $i$ under pressure, we construct a rich, highly non-linear vector of spatial features $X_i$ from the 360° freeze-frame. Multiple defender `Pressure` events linked to the same carrier action are collapsed to one modeled observation, because the freeze-frame features already encode how many opponents are close. Let the ball-carrier's location be $C = (x_c, y_c)$, the set of teammates $T$, and the set of opponents $O$.

### 3.1 Gaussian Pitch Control
We discard naive additive distance formulas in favor of a continuous **Gaussian Influence Model** (inspired by Fernandez and Bornn's wide-scale pitch control models). It calculates the probabilistic ownership of the ball-carrier's immediate location based on the density and proximity of teammates versus opponents. 

The influence of any set of players at a specific point on the pitch is defined as:
$$ \text{Influence}(P) = \sum_{d \le r_{max}} \exp\left( - \frac{d^2}{2\sigma^2} \right) $$
Where $d$ is the Euclidean distance from the player to the point, $\sigma = 4.2$ yards (Fernandez/Bornn), and $r_{max} = 15$ yards is the influence cutoff beyond which players have negligible effect. The final Pitch Control metric is the ratio of teammate influence to total influence, mapped to the interval $[-1, 1]$. A negative value indicates severe opponent isolation; a positive value indicates strong teammate support.

### 3.2 Lane-Aware Progressive Options
A teammate is no longer considered a "progressive option" merely by standing geographically closer to the opponent's goal. The framework imposes two strict physical constraints:
1. **Value Constraint:** The teammate must occupy a zone with higher xT than the ball-carrier, or be strictly more than 5 yards further up the pitch (`tm_x > bc_x + 5.0`).
2. **Physics Constraint:** The system utilises a `LineString` geometric intersection check to verify that the direct passing lane between the ball-carrier and the teammate is physically unblocked by the coverage radii of defending opponents (clearance radius: 1.5 yards).

### 3.3 Angular Coverage Arc ($\Phi$)
This feature mathematically calculates the "visual wall" presented by opponents within a parameterized coverage radius (e.g., 3.0 yards) of the ball-carrier. Instead of assuming standard block widths, it uses a principled trigonometric formula (`2 * np.arctan((player_width / 2) / distance)`) to derive the exact angular span an opponent's physical body blocks in the passing lane. We calculate the angle of each closing opponent, sort them, and identify the largest angular gap (representing the widest available escape or passing lane). The coverage arc is defined as the remainder of the $2\pi$ circle. A larger $\Phi$ indicates the ball-carrier is heavily surrounded, severely limiting physical escape vectors.

### 3.4 Relative Orientation ($\psi$)
Captures where the pressure is originating from relative to the opponent's goal (the ultimate objective). Angle mapping utilizes proper trigonometric projections (`np.arctan2(y, x)`), orienting relative pressure to the strict $X$-axis of the attacking pitch. This allows the model to differentiate the psychological difficulty of a defender pressing from the blindside (back pressure) versus head-on confrontation.

### 3.5 Voronoi Area ($A_{vor}$)
We compute the Voronoi tessellation of all players on the pitch, clipping the resultant polygons to the configured pitch boundaries (e.g., 120x80 yards). If edge cases occur (e.g., perfect collinearity, fewer than 4 players in frame), the system falls back to a statistically rigorous grid-based approximation. $A_{vor}$ is the area of the cell containing the ball-carrier — the raw square footage of grass the ball-carrier controls outright before facing an immediate tackle.

### 3.6 B-Spline Pitch Coordinates ($bc_x$, $bc_y$, $d_{min}$)
The ball-carrier's pitch coordinates and distance to the nearest opponent are mapped through cubic B-spline basis functions before entering the model. Earlier implementations encoded location as an ordinal zone integer, which imposed a false linear ordering. The raw coordinate was later used directly, but a linear scaler still cannot capture the highly non-linear value gradients on a football pitch (e.g., the steep xT gradient in front of goal, or the qualitatively different pressure dynamics at the touchline).

Using scikit-learn's `SplineTransformer` with $n_{\text{knots}} = 5$, $\text{degree} = 3$, and `extrapolation="constant"`, each of the three raw features ($bc_x$, $bc_y$, $d_{\text{min}}$) is expanded into **6 B-spline basis columns** ($(n_{\text{knots}} - 1) + \text{degree} - 1$ basis functions with `include_bias=False`). The transformers are fitted on the training split, saved alongside the StandardScaler, and applied identically to the holdout set to prevent data leakage. This yields 18 additional model columns that flexibly capture non-linear pitch geography without requiring the practitioner to hand-engineer interaction terms.

### 3.7 Match Context Features
Six non-spatial contextual features are appended to the feature vector from the match event stream:

| Feature | Source | Semantics |
|---------|---------|-----------|
| `game_state_diff` | Goal tracking (Shot + Own Goal events) | Score differential at event time from ball-carrier's team perspective. Captures clutch-pressure effects and game management behaviour. |
| `minutes_elapsed` | Event `minute` field | Enables the model to learn fatigue and time-pressure effects. |
| `match_period` | Event `period` field | Accounts for structural differences between match periods, including extra time when present. |
| `counter_press` | Pressure event `counterpress` flag | Binary indicator whether the pressure was part of an organised counter-press sequence. Counter-pressing events create qualitatively higher cognitive load and time constraints. |
| `pass_height_ground` | Pass event `pass.height.id == 1` | Binary indicator for ground-level passes (StatsBomb ID 1). |
| `pass_height_low` / `pass_height_high` | `pass.height.id == 2 / 3` | Binary indicators for low (lofted) and high (aerial) pass trajectories. Pass height strongly mediates the technical execution difficulty under pressure. |

`game_state_diff` is computed by scanning events chronologically and tracking goals via `type == 'Shot'` with `shot_outcome == 'Goal'` and `type == 'Own Goal For'` events. The differential is recorded *before* each event is processed, ensuring no lookahead leakage. The three pass-height columns form a one-hot encoding of the pass height category; non-pass events receive all-zero encodings.

---

## 4. The Dual Target Variables: Success & Value

To solve the Zero-Inflation problem and the Safe-Pass bias simultaneously, we split the evaluation into two distinct targets:

1. **$Y_{success}$:** A binary variable ($1$ if possession is retained, $0$ if dispossessed). The logic traces up to exactly 5 subsequent actions (`carry_lookahead_events`) to verify true possession retention — recognising a `Foul Committed` event by the *opposing* team (i.e. a foul won by the carrier) as success, and correctly attributing opponent actions (`Pass`, `Carry`, `Shot`, `Clearance`, `Interception`, `Dispossessed`) as failures. StatsBomb does not emit a `Foul Won` event type; the corresponding signal is `Foul Committed` logged under the opposing team's `team_id`.
2. **$V_{intended}$:** The value of the *intended* action, stored as the `value_preserved` column in the processed dataset. The framework supports two interchangeable value signals:
   - **VAEP (primary):** When pre-trained LightGBM classifiers are available, the value is computed as $\text{VAEP} = (P_{\text{score,after}} - P_{\text{score,before}}) - (P_{\text{concede,after}} - P_{\text{concede,before}})$, implementing Decroos et al. (2019). Two classifiers — one for $P(\text{score in next } n \text{ actions})$ and one for $P(\text{concede in next } n \text{ actions})$ — are trained on all available StatsBomb events with a lookahead window of $n = 10$ actions. VAEP accounts for both the offensive threat created and the defensive risk introduced by the action, resolving a key limitation of unidirectional xT.
   - **xT (fallback):** When VAEP models are not available, the framework falls back to the Karun Singh 8×12 Expected Threat grid. For passes, it evaluates xT at the destination coordinate; for carries and dribbles, at the location of the next action in the sequence.

For the Beta distribution component of the model, $V_{intended}$ is scaled to the open interval $(0, 1)$ using a theoretical maximum value $V_{max}$ and a smoothing factor $\epsilon = 10^{-6}$:
$$ V_{scaled} = \left( \frac{V_{intended}}{V_{max}} \right) (1 - 2\epsilon) + \epsilon $$

---

## 5. Bayesian Hierarchical Beta Hurdle Model

To extract the underlying, unobservable cognitive traits of the players, we construct a joint Hurdle Model using PyMC and the NUTS (No-U-Turn Sampler) algorithm.

### 5.1 The Hurdle Architecture
The model strictly separates the prediction space into two sequential hurdles:
* **Hurdle 1 (Ball Security):** Modeled via Logistic Regression. Predicts the probability $p$ that $Y_{success} = 1$.
   $$ Y_{success} \sim \text{Bernoulli}(p) $$
* **Hurdle 2 (Value Retention):** Modeled via Beta Regression. Evaluated *only* on the subset of data where $Y_{success} = 1$. It predicts the expected value generated $\mu$, given that the ball was not lost.
   $$ V_{scaled} \sim \text{Beta}(\alpha=\mu \kappa, \beta=(1 - \mu) \kappa) $$
   The concentration parameter $\kappa \sim \text{Exponential}(0.1)$ gives a prior mean of 10, appropriate for Beta regression on bounded xT values. A mean of 1 (the default Exponential(1.0)) would force extreme bimodal distributions and cause numerical instability.

### 5.2 The Linear Predictors
Both sub-models utilise a parallel hierarchical linear structure linked via the inverse-logit function:
$$ \text{logit}(p_i) = \alpha_{succ} + X_i \beta_{succ} + \gamma_{pos, succ} + \theta_{player, succ} + \delta_{opp, succ} + \zeta_{comp, succ} + \eta_{team, succ} $$
$$ \text{logit}(\mu_i) = \alpha_{val} + X_i \beta_{val} + \gamma_{pos, val} + \theta_{player, val} + \delta_{opp, val} + \zeta_{comp, val} + \eta_{team, val} $$

Where:
* **$\theta_{player, succ}$:** The player's intrinsic *Ball Security* trait (resistance to being tackled or forcing an error).
* **$\theta_{player, val}$:** The player's intrinsic *Value Retention* trait (ability to spot and execute dangerous passes despite pressure).
* $X_i \beta$: The dot product of the standardised feature vector (spatial + contextual + B-spline expansions), establishing the mathematical difficulty of the specific situation.
* $\gamma_{pos}$: Fixed effect for the player's position group (CB / FB / DM / CM / W / CF) — the replacement-level baseline within their tactical role.
* $\delta_{opp}$: Random effect capturing the specific defensive intensity of the opposing team.
* $\zeta_{comp}$: Random effect capturing the tactical ecosystem and competitive standard of the competition.
* **$\eta_{team}$:** Random effect for the ball-carrier's own attacking team, capturing systematic offensive style (e.g. high-press, possession-dominant teams generate a different distribution of pressure situations than reactive counter-attacking sides).

### 5.3 Priors and Non-Centered Parameterisation
Weakly informative Normal priors are enforced throughout to provide regularisation (shrinkage), preventing the model from overfitting players with small sample sizes (e.g., a player who succeeds in 2 out of 2 duels will be shrunk heavily toward the positional mean).

All random effects use **non-centered parameterisations** to allow the HMC sampler to navigate the complex funnel geometries that arise in hierarchical models. Rather than sampling $\theta$ directly from $\mathcal{N}(0, \sigma)$, we sample a raw standard normal offset and scale it:
$$ \tilde{\theta}_{player} \sim \mathcal{N}(0, 1) $$
$$ \sigma_\theta \sim \text{Exponential}(1.0) $$
$$ \theta_{player} = \tilde{\theta}_{player} \times \sigma_\theta $$

#### Correlated Player Traits — LKJ Cholesky Prior
Because *Ball Security* ($\theta_{succ}$) and *Value Retention* ($\theta_{val}$) are cognitively related skills, they are modelled jointly as a bivariate Gaussian with a learnt covariance structure rather than as two independent scalars. The prior is specified via the LKJ distribution over correlation matrices (Lewandowski, Kurowicka & Joe, 2009):

$$ \begin{pmatrix} \tilde{\theta}_{succ,i} \\ \tilde{\theta}_{val,i} \end{pmatrix} \sim \mathcal{N}(0, I_{2}) \quad \text{(raw offsets)} $$
$$ \mathbf{L} \sim \text{LKJCholeskyCov}(\eta = 2,\; \sigma \sim \text{Exponential}(1.0)) $$
$$ \begin{pmatrix} \theta_{succ,i} \\ \theta_{val,i} \end{pmatrix} = \mathbf{L}\, \tilde{\theta}_i $$

The LKJ($\eta = 2$) prior places mild regularising pressure toward near-zero off-diagonal correlations, avoiding the degenerate case where the sampler collapses to a rank-1 solution. The posterior correlation $\rho_{\theta}$ is saved as a named deterministic variable and provides a direct quantitative estimate of the degree to which composure (ball security) and creativity (value retention) co-vary across the population.

The sampler thereby explores a shared low-dimensional manifold for each player, yielding more efficient MCMC mixing and stronger regularisation for players with sparse observations.

---

## 6. Interpretability Framework

A black-box model is useless for elite football scouting. The PRS framework provides explicit, geometric interpretability.

### 6.1 Covariance-Aware Variance Decomposition
To understand which components drive outcome variance, we decompose the total variance of the linear predictor into four distinct sources for each sub-model:

| Component | Estimator |
|-----------|----------|
| **Player Skill** | $\mathbb{E}[\sigma_\theta^2]$ (posterior mean of squared player SD) |
| **Opponent Quality** | $\mathbb{E}[\sigma_{opp}^2]$ |
| **Competition Context** | $\mathbb{E}[\sigma_{comp}^2]$ |
| **Spatial Features** | $\mathbb{E}[\text{Var}(X\beta)]$ — true sample variance of linear predictor, accounting for multi-collinearity |

Rather than naively summing squared $\beta$ coefficients (which assumes zero correlation between features), the feature component computes the true variance of the full linear predictor matrix $X\beta$ across the dataset. This correctly handles the heavy multi-collinearity between spatial features like density and coverage arcs.

### 6.2 Counterfactuals and "Best Under" Scenarios
To identify a player's specialized composure profile, we define orthogonal geometric matrices representing specific, identifiable tactical situations (e.g., "Front_Tight", "Lateral_Loose", "Back_Tight"). 

For a given Scenario $S$, we compute the total Expected Value generated by the specific player under that specific condition:
$$ \mathbb{E}[V | S, player] = \text{logit}^{-1}(\eta_{succ, S}) \times \text{logit}^{-1}(\eta_{val, S}) \times V_{max} $$

We then subtract the positional population baseline (what an average player in that role would generate) to map explicit geometric profiles. This answers specific scouting questions: *Does this holding midfielder handle pressure from the blindside better than pressure from the front?*

---

## 7. Validation Strategy: Out-of-Sample Residual Correlation & Calibration

Evaluating performance via in-sample loss functions (like AUC or RMSE on the training set) is insufficient for assessing whether a metric represents a scoutable, intrinsic player trait rather than statistical noise. The PRS employs two complementary out-of-sample validation protocols.

### 7.1 Expected Value Residual Correlation

1. **Prediction Generation:** The Hurdle model is trained on a diverse set of competitions (e.g., Euro 2024, World Cup 2022, Copa América 2024).
2. **Holdout Evaluation:** A completely separate competition is withheld. The specific holdout is controlled by the `PRS_HOLDOUT` environment variable (default: `Euro_2020`). For every carrier action under pressure $j$ in the holdout set, we calculate the expected value generated by an *average* player in that exact spatial geometry using fully vectorised posterior predictive operations. Opponent and competition random effects are marginalised to zero (their prior mean under the non-centered parameterisation), which is the principled choice for unseen groups:
   $$ \hat{V}_j = \sigma(\hat{\eta}_{succ, j}) \times \sigma(\hat{\eta}_{val, j}) \times V_{max} $$
3. **Residual Aggregation:** We calculate the value generated *above expected* for each event:
   $$ r_j = V_{true, j} - \hat{V}_j $$
   A positive residual indicates the player achieved an outcome better than the mathematical difficulty of the situation.
4. **Correlation:** Residuals are averaged per player ($\bar{r}_{player}$) and correlated with their *training* PRS via both Pearson and Spearman statistics.

A significant positive Pearson correlation is evidence that the two $\theta$ traits extracted by the model capture stable, persistent cognitive and technical traits — composure under pressure is not merely random variance but a measurable, transferable skill.

### 7.2 Calibration Assessment (ECE)

Beyond rank-order correlation, we assess the absolute reliability of the Ball Security probability estimates using the **Expected Calibration Error (ECE)**:
$$ \text{ECE} = \frac{1}{B} \sum_{b=1}^{B} \left| \bar{y}_b - \bar{p}_b \right| $$
where events are partitioned into $B = 10$ equal-width bins by predicted probability, $\bar{p}_b$ is the mean predicted probability in bin $b$, and $\bar{y}_b$ is the empirical success rate. A reliability curve (predicted vs. observed probability) is generated for each holdout run and saved as `calibration_curve_{holdout}.csv`. A well-calibrated model has ECE close to zero and a reliability curve close to the diagonal.

The holdout AUC for binary ball-security prediction and ECE are both reported in `holdout_metrics_{holdout}.csv`.

---

## 8. Limitations

While the PRS framework represents a methodological advance over naive binary metrics, several assumptions and constraints should be acknowledged:

### 8.1 Sample Size and Generalisability
The model trains on 4 international tournaments from the StatsBomb Open Data catalogue (~600 matches, ~50 000 pressure events after filtering). This provides strong statistical power for estimating population-level feature effects, but the per-player random effect ($\theta_i$) for infrequently-observed players (< 30 events) has wide credible intervals. Furthermore, the model has only been validated on international competitions; club-level football may exhibit different pressure geometries, tactical pressing systems, and roster continuity effects. Extending the model to club data would require re-examining the competition random effect and the hierarchical variance priors.

### 8.2 Composure as a Stable Trait
The hierarchical model assumes that a player's latent composure ($\theta_i$) is a time-invariant trait — the same in minute 5 and minute 85, across a group-stage dead rubber and a semi-final. In reality, composure is likely form-dependent (injury, confidence) and context-dependent (home vs away, scoreline, tournament stage). The `game_state_diff` and `minutes_elapsed` features partially control for within-match variation, but they do not model time-varying player effects. A state-space extension (random walk on $\theta_i$ across matches) would address this at the cost of computational complexity.

### 8.3 Action-Value Resolution
When the VAEP models are available (the default for full training runs), the value signal accounts for the full sequence of subsequent actions up to a lookahead window of 10 events, resolving the static cell-boundary problem inherent to xT grids. The xT fallback uses the Karun Singh 8×12 discrete grid, giving each cell a coverage of $\approx 15 \times 7$ yards. Actions that start and end within the same cell receive $\Delta xT = 0$, undervaluing subtle positional improvements. When using the xT fallback, a higher-resolution surface (e.g., $16 \times 24$ or a continuous gradient-boosted surface) would reduce quantisation error at the cost of requiring additional training data to populate reliably.

### 8.4 Angular Coverage Approximation
The `coverage_arc` feature uses a body-width trigonometric projection to estimate how much of the carrier's angular field of view is blocked by opponents. For multi-opponent scenarios, the implementation assumes opponents are non-overlapping angular sources. When opponents are very close together (< 1 yard apart), their individual body-width arcs may overlap, leading to a slight overestimate of coverage. Additionally, the model does not account for opponent height, approach speed, or body orientation, all of which influence the perceived pressure in practice.

### 8.5 Freeze-Frame Temporal Resolution
StatsBomb 360 frames capture a single snapshot at the moment of the event. They do not encode player velocities, acceleration vectors, or body orientation. A player sprinting toward the carrier at 8 m/s from 4 yards away exerts substantially more pressure than a stationary player at the same distance, but both produce identical spatial features. Integrating tracking data (where available) to compute velocity-weighted features would substantially improve the model's ability to distinguish true pressure from spatial proximity.
