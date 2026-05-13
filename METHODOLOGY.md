# Methodology: Pressure Resistance Score (PRS)

This document serves as the formal, comprehensive whitepaper detailing the mathematical, conceptual, and analytical framework underpinning the **Pressure Resistance Score (PRS)**. It outlines the paradigm shift from naive success metrics to a dual-axis value framework, the derivation of domain-accurate geometric features from 360-degree freeze-frame data, the architecture of the Bayesian Hierarchical Zero-Inflated Beta Regression (Hurdle Model), and the rigorous out-of-sample validation strategy utilized to prove its efficacy.

---

## 1. Theoretical Framework & Motivation

Historically, football analytics has evaluated a player's "pressure resistance" or "composure" by calculating their passing completion percentage while under pressure. However, this binary approach suffers from severe ecological fallacies that limit its utility for elite scouting and tactical analysis:

1. **The Safe-Pass Bias (Survival Bias):** Binary completion metrics treat a 2-yard backward pass to an unmarked center-back exactly the same as a 20-yard line-breaking pass through an opponent's midfield block. Both are logged as a "success" ($Y = 1$). Consequently, players who are highly risk-averse—those who immediately recycle possession backwards the moment they feel pressure—are artificially rewarded. This masks a player's inability to actually progress the ball when constrained.
2. **The Zero-Inflation Problem:** A player under pressure faces two distinct, sequential cognitive challenges: 
   * *Phase 1: Can I physically keep the ball?* (Ball Security / Turnover Risk)
   * *Phase 2: If I keep it, can I execute a dangerous action?* (Value Generation / Offensive Output)
   Standard continuous models collapse these two distinct processes into a single metric, failing to handle the massive spike of exact zeroes (turnovers) in the data. A failed action creates an artificial floor that breaks the assumptions of standard linear regression.
3. **The Spatial Vacuum:** Traditional models largely ignore the surrounding pitch geometry. A player's ability to complete a pass depends heavily on unblocked passing lanes, the angular span of closing opponents, and the dynamic shifting of pitch control. A model without geometric context is evaluating decisions in a vacuum.

The PRS framework resolves these fundamental flaws by migrating to a **Zero-Inflated Hurdle framework**. This allows us to evaluate *Turnover Risk* independently from *Expected Threat (xT) Retained*, conditioning both values on the exact spatial geometry of the dual, and extracting the player's true underlying skill via a Bayesian hierarchical model.

---

## 2. Cohort Definition and Data Preprocessing

To ensure the model isolates genuine cognitive and technical challenges, the raw StatsBomb 360 dataset undergoes strict filtering and cohort stratification before entering the modeling pipeline.

### 2.1 Goalkeeper Exclusion
Goalkeepers face fundamentally different pressure geometries than outfield players. They typically receive the ball facing forward with the entire pitch ahead of them, and have the tactical license to play low-risk lateral passes or launch long clearances. In a baseline model, their high completion rate of these low-value passes artificially inflates their apparent composure. Therefore, goalkeepers are entirely excluded from the training and evaluation cohorts.

### 2.2 The Tight-Pressure Filter
Utilizing the continuous $x,y$ coordinates from the 360° freeze-frames, we calculate the Euclidean distance to the nearest opponent, denoted as $d_{min}$, at the exact fraction of a second the ball-carrier receives the ball. 

To remove the statistical noise of "token pressure"—such as a striker casually jogging toward a defender from 8 yards away—we strictly filter the dataset to only include events where:
$$ d_{min} \le 5.0 \text{ yards} $$
This forces the model to evaluate genuine, close-quarters duals, ensuring that a high PRS score translates directly to performance in the most heavily contested areas of the pitch.

### 2.3 Dynamic Positional Stratification
Players are grouped into broad tactical roles: Defender, Midfielder, and Forward. This allows the hierarchical model to estimate baseline spatial constraints and expected values specific to each role. A center-back naturally operates in deeper zones with lower baseline Expected Threat (xT) opportunities compared to a forward receiving the ball on the edge of the penalty box. 

To prevent the arbitrary misclassification of players who lack formal lineup data (a common issue in sprawling datasets), the system dynamically computes the spatial center of gravity (the average $X$ coordinate of all their recorded touches). This data-driven imputation ensures players are accurately assigned to their structural role, allowing them to be evaluated by an "Above-Replacement" standard strictly relative to their true positional peers.

---

## 3. Geometric Feature Engineering

For every filtered pressure event $i$, we construct a rich, highly non-linear vector of spatial features $X_i$ from the 360° freeze-frame. Let the ball-carrier's location be $C = (x_c, y_c)$, the set of teammates $T$, and the set of opponents $O$.

### 3.1 Gaussian Pitch Control
We discard naive additive distance formulas in favor of a continuous **Gaussian Influence Model** (inspired by Fernandez and Bornn's wide-scale pitch control models). It calculates the probabilistic ownership of the ball-carrier's immediate location based on the density and proximity of teammates versus opponents. 

The influence of any set of players at a specific point on the pitch is defined as:
$$ \text{Influence}(P) = \sum \exp\left( - \frac{d^2}{2\sigma^2} \right) $$
Where $d$ is the Euclidean distance from the player to the point, and $\sigma$ is the standard deviation of influence, standardized to 4.2 yards. The final Pitch Control metric is the ratio of teammate influence to total influence, mapped to the interval $[-1, 1]$. A negative value indicates severe opponent isolation; a positive value indicates strong teammate support.

### 3.2 Lane-Aware Progressive Options
A teammate is no longer considered a "progressive option" merely by standing geographically closer to the opponent's goal. The framework imposes two strict physical constraints:
1. **Value Constraint:** The teammate must reside in a zone with a higher Expected Threat (xT) than the ball-carrier, or be significantly advanced up the pitch.
2. **Physics Constraint:** The system utilizes a `LineString` geometric intersection check to verify that the direct passing lane between the ball-carrier and the teammate is physically unblocked by the coverage radii of any defending opponents.

### 3.3 Angular Coverage Arc ($\Phi$)
This feature mathematically calculates the "visual wall" presented by opponents within a parameterized coverage radius (e.g., 3.0 yards) of the ball-carrier. Instead of assuming standard block widths, it uses a principled trigonometric formula (`2 * np.arctan((player_width / 2) / distance)`) to derive the exact angular span an opponent's physical body blocks in the passing lane. We calculate the angle of each closing opponent, sort them, and identify the largest angular gap (representing the widest available escape or passing lane). The coverage arc is defined as the remainder of the $2\pi$ circle. A larger $\Phi$ indicates the ball-carrier is heavily surrounded, severely limiting physical escape vectors.

### 3.4 Relative Orientation ($\psi$)
Captures where the pressure is originating from relative to the opponent's goal (the ultimate objective). Angle mapping utilizes proper trigonometric projections (`np.arctan2(y, x)`), orienting relative pressure to the strict $X$-axis of the attacking pitch. This allows the model to differentiate the psychological difficulty of a defender pressing from the blindside (back pressure) versus head-on confrontation.

### 3.5 Voronoi Area ($A_{vor}$)
We compute the Voronoi tessellation of all players on the pitch, clipping the resultant polygons to the configured pitch boundaries (e.g., 120x80 yards). If edge cases occur (e.g., perfect collinearity), the system falls back to a statistically rigorous uniform geometric expectation rather than arbitrary grid approximations. $A_{vor}$ is the area of the cell containing the ball-carrier. It mathematically represents the raw square footage of grass the ball-carrier controls outright before facing an immediate tackle.

---

## 4. The Dual Target Variables: Success & Value

To solve the Zero-Inflation problem and the Safe-Pass bias simultaneously, we split the evaluation into two distinct targets:

1. **$Y_{success}$:** A binary variable ($1$ if possession is retained, $0$ if dispossessed). Crucially, the logic traces up to 5 subsequent actions to verify true possession retention (e.g., recognizing "Foul Won" as a success and correctly attributing immediate interceptions by opponents as failures).
2. **$V_{intended}$:** The Expected Threat (xT) value of the *intended* action, measured continuously. If the event is a pass, it measures the xT of the destination coordinate. If the event is a carry, it evaluates the final destination. 

For the Beta distribution component of our model, $V_{intended}$ is scaled to the open interval $(0, 1)$ using a theoretical maximum value $V_{max}$ and a microscopic smoothing factor $\epsilon = 10^{-6}$:
$$ V_{scaled} = \left( \frac{V_{intended}}{V_{max}} \right) (1 - 2\epsilon) + \epsilon $$

---

## 5. Bayesian Hierarchical Zero-Inflated Beta Regression (Hurdle Model)

To extract the underlying, unobservable cognitive traits of the players, we construct a joint Hurdle Model using PyMC and the NUTS (No-U-Turn Sampler) algorithm.

### 5.1 The Hurdle Architecture
The model strictly separates the prediction space into two sequential hurdles:
* **Hurdle 1 (Turnover Risk):** Modeled via Logistic Regression. Predicts the probability $p$ that $Y_{success} = 1$.
   $$ Y_{success} \sim \text{Bernoulli}(p) $$
* **Hurdle 2 (Value Retention):** Modeled via Beta Regression. Evaluated *only* on the subset of data where $Y_{success} = 1$. It predicts the expected value generated $\mu$, given that the ball was not lost.
   $$ V_{scaled} \sim \text{Beta}(\alpha=\mu \kappa, \beta=(1 - \mu) \kappa) $$

### 5.2 The Linear Predictors
Both sub-models utilize a parallel hierarchical linear structure linked via the inverse-logit function:
$$ \text{logit}(p_i) = \alpha_{succ} + X_i \beta_{succ} + \gamma_{pos, succ} + \theta_{player, succ} + \delta_{opp, succ} + \zeta_{comp, succ} $$
$$ \text{logit}(\mu_i) = \alpha_{val} + X_i \beta_{val} + \gamma_{pos, val} + \theta_{player, val} + \delta_{opp, val} + \zeta_{comp, val} $$

Where:
* **$\theta_{player, succ}$:** The player's intrinsic *Ball Security* trait (resistance to being tackled or forcing an error).
* **$\theta_{player, val}$:** The player's intrinsic *Value Retention* trait (ability to spot and execute dangerous passes despite pressure).
* $X_i \beta$: The dot product of the standardized geometric features, establishing the mathematical difficulty of the specific situation.
* $\gamma_{pos}$: The fixed effect for the player's position group (the replacement-level baseline).
* $\delta_{opp}$: A random effect capturing the specific defensive intensity of the opposing team.

### 5.3 Priors and Non-Centered Parameterization
We enforce weakly informative Normal priors to provide regularization (shrinkage), preventing the model from overfitting players with small sample sizes (e.g., a youth player who succeeds in 2 out of 2 duals will be shrunk heavily toward the mean). 

To optimize the Hamiltonian Monte Carlo sampler and navigate the complex geometric funnels of hierarchical models, we utilize **non-centered parameterizations** for all random effects. For example, rather than sampling $\theta$ directly from $\mathcal{N}(0, \sigma)$, we sample a raw standard normal and multiply it by the standard deviation:
$$ \tilde{\theta}_{player} \sim \mathcal{N}(0, 1) $$
$$ \sigma_\theta \sim \text{Exponential}(1.0) $$
$$ \theta_{player} = \tilde{\theta}_{player} \times \sigma_\theta $$
This allows the PyMC sampler to explore the posterior landscape with maximum efficiency, avoiding divergent transitions.

---

## 6. Interpretability Framework

A black-box model is useless for elite football scouting. The PRS framework provides explicit, geometric interpretability.

### 6.1 Covariance-Aware Variance Decomposition
To understand which components (Player Skill vs. Spatial Difficulty vs. Opponent Quality) drive outcomes, we decompose the variance of the predictions. Instead of naively summing squared $\beta$ coefficients—which assumes spatial features are completely uncorrelated—the script computes the true sample variance of the linear predictor matrix $\text{var}(X \beta)$. This properly accounts for the heavy multi-collinearity between spatial features like density and coverage arcs.

### 6.2 Counterfactuals and "Best Under" Scenarios
To identify a player's specialized composure profile, we define orthogonal geometric matrices representing specific, identifiable tactical situations (e.g., "Front_Tight", "Lateral_Loose", "Back_Tight"). 

For a given Scenario $S$, we compute the total Expected Value generated by the specific player under that specific condition:
$$ \mathbb{E}[V | S, player] = \text{logit}^{-1}(\eta_{succ, S}) \times \text{logit}^{-1}(\eta_{val, S}) \times V_{max} $$

We then subtract the positional population baseline (what an average player in that role would generate) to map explicit geometric profiles. This answers specific scouting questions: *Does this holding midfielder handle pressure from the blindside better than pressure from the front?*

---

## 7. Validation Strategy: Out-of-Sample Residual Correlation

Evaluating performance via in-sample loss functions (like AUC or RMSE on the training set) is insufficient for proving a metric represents a scoutable, intrinsic player trait rather than statistical noise. The PRS utilizes **Out-of-Sample Expected Value Residual Correlation**, the gold standard in sports analytics.

1. **Prediction Generation:** The Hurdle model is trained on a massive, diverse set of competitions (e.g., Euro 2024, World Cup 2022).
2. **Holdout Evaluation:** A completely separate competition with a distinct tactical ecosystem (e.g., Euro 2020) is withheld. For every pressure event $j$ in the holdout set, we calculate the expected value generated by an *average* player in that exact spatial geometry using fully vectorized tensor operations:
   $$ \hat{V}_j = p_{baseline, j} \times \mu_{baseline, j} \times V_{max} $$
3. **Residual Aggregation:** We calculate the true value generated *above expected* for that specific event:
   $$ r_j = V_{true, j} - \hat{V}_j $$
   *(A positive residual means the player achieved an outcome better than the mathematical difficulty of the situation suggested).*
4. **Correlation:** We average the residuals per player across their season in the holdout dataset ($\bar{r}_{player}$) and correlate this with their *training* PRS ($\theta_{player}$) derived from the entirely separate training dataset.

A significant positive Pearson correlation empirically proves that the dual $\theta$ traits extracted by the model accurately measure stable, persistent cognitive and technical traits. It proves that composure under pressure is not random variance, but a measurable skill that transfers across entirely different tactical environments, leagues, and tournaments.