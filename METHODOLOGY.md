# Methodology: Pressure Resistance Score (PRS)

This document serves as the formal whitepaper detailing the mathematical, conceptual, and analytical framework underpinning the **Pressure Resistance Score (PRS)**. It comprehensively outlines the shift from binary success metrics to continuous expected value models, the derivation of geometric features from 360-degree freeze-frame data, the architecture of the Bayesian Hierarchical Beta Regression, and the rigorous out-of-sample validation strategy used to prove the metric's stability.

---

## 1. Theoretical Framework & Motivation

Historically, football analytics has evaluated a player's "pressure resistance" by calculating their passing completion percentage while under pressure. However, this binary approach suffers from severe ecological fallacies that limit its utility for scouting and tactical analysis:

1. **The Safe-Pass Bias:** Binary metrics treat a 2-yard backward pass to an unmarked center-back exactly the same as a 20-yard line-breaking pass through an opponent's midfield block. Both are logged as a "success" ($Y = 1$). Consequently, players who are highly risk-averse are artificially rewarded, masking their inability to actually progress the ball when constrained.
2. **The "Token Pressure" Problem:** Not all pressure is created equal. A striker casually jogging toward a defender from 8 yards away registers as "Pressure" in event logs, just as a coordinated two-man high press within 1 yard does. Mixing these fundamentally different geometric states dilutes the statistical signal of true composure.
3. **The Spatial Vacuum:** Traditional models largely ignore the surrounding pitch geometry. A player's ability to complete a pass depends heavily on passing lane availability, the angular span of blocking opponents, and the density of teammates offering support.

The PRS framework resolves these fundamental flaws by migrating from *binary completion* to **Expected Threat (xT) preserved**, conditioning this value on the exact spatial geometry of the dual, adjusting for the player's tactical role, and extracting the player's true underlying skill via a Bayesian hierarchical model.

---

## 2. Cohort Definition and Data Preprocessing

To ensure the model isolates genuine cognitive and technical challenges, the raw StatsBomb 360 dataset undergoes strict filtering and cohort stratification before entering the modeling pipeline.

### 2.1 Goalkeeper Exclusion
Goalkeepers face fundamentally different pressure geometries than outfield players. They typically receive the ball facing forward with the entire pitch ahead of them and have the tactical license to play low-risk lateral passes to wide-open defenders or launch long clearances. In a baseline model, their completion rate of these low-value passes artificially inflates their apparent composure. Therefore, goalkeepers are entirely excluded from the training and evaluation cohorts.

### 2.2 The Tight-Pressure Filter
Utilizing the continuous $x,y$ coordinates from the 360° freeze-frames, we calculate the Euclidean distance to the nearest opponent $d_{min}$ at the exact moment the ball-carrier receives pressure. 
To remove the noise of "token pressure," we strictly filter the dataset to only include events where:
$$ d_{min} \le 5.0 \text{ yards} $$
This forces the model to evaluate genuine, close-quarters duals, ensuring that a high PRS score translates to performance in the most contested areas of the pitch (the "trenches").

### 2.3 Positional Stratification
Players are grouped into broad tactical roles $g \in \{\text{Defender}, \text{Midfielder}, \text{Forward}\}$. This grouping allows the hierarchical model to estimate baseline spatial constraints and expected values specific to each role. A center-back naturally operates in deeper zones with lower baseline xT opportunities compared to a forward receiving the ball on the edge of the penalty box. Stratification ensures that players are judged by an "Above-Replacement" standard *relative to their peers in similar roles*.

---

## 3. Geometric Feature Engineering

For every filtered pressure event $i$, we construct a rich vector of spatial features $X_i$ from the 360° freeze-frame. Let the ball-carrier's location be $C = (x_c, y_c)$, the set of teammates $T = \{T_1, T_2, \dots\}$, and the set of opponents $O = \{O_1, O_2, \dots\}$.

### 3.1 Proximity and Density Metrics
* **Distance to Nearest Opponents:** The Euclidean distance to the primary ($d_{min}$) and secondary ($d_{2nd}$) nearest opponents.
* **Density Counters:** The scalar count of opponents within defined radii of $C$ (1 yard, 2 yards, and 4 yards). This captures "swarming" behavior that a simple minimum distance metric might miss.

### 3.2 Angular Coverage Arc ($\Phi$)
This feature measures the "visual wall" presented by opponents. We consider only opponents within a 3.0-yard radius of $C$. We calculate the angle of each opponent relative to $C$, sort these angles, and identify the largest angular gap (the widest passing lane). The coverage arc is defined as the remainder of the $2\pi$ circle:
$$ \Phi = 2\pi - \max(\text{gaps}) $$
A larger $\Phi$ indicates the ball-carrier is heavily surrounded, limiting escape vectors.

### 3.3 Relative Orientation ($\psi$)
This captures where the pressure is coming from relative to the opponent's goal (the ultimate objective). Let the angle to the center of the opponent's goal be $\theta_{goal}$ and the angle to the nearest opponent be $\theta_{opp}$.
$$ \psi = (\theta_{opp} - \theta_{goal} + \pi) \pmod{2\pi} - \pi $$
This maps the relative angle to $[-\pi, \pi]$. 
* $\psi \approx 0$: The opponent is directly between the player and the goal (Front pressure).
* $\psi \approx \pm\pi/2$: The opponent is pressing from the side (Lateral pressure).
* $\psi \approx \pm\pi$: The opponent is pressing from behind (Back pressure).

### 3.4 Voronoi Area ($A_{vor}$)
We compute the Voronoi tessellation of all players on the pitch, clipping the polygons to the pitch boundaries. $A_{vor}$ is the area of the cell containing $C$. It mathematically represents the square footage of grass the ball-carrier controls outright. When under intense pressure, this area collapses rapidly.

### 3.5 Pitch Control ($PC$)
A localized additive spatial influence model evaluated at $C$. It captures the relative density of supporting teammates versus pressing opponents, using a distance-decay function:
$$ PC(C) = \sum_{t \in T} \frac{1}{1 + ||t - C||^2} - \sum_{o \in O} \frac{1}{1 + ||o - C||^2} $$
Bounded to $[-1, 1]$, a negative value indicates severe opponent isolation, while a positive value indicates strong teammate support.

---

## 4. The Target Variable: Expected Value Preserved ($V$)

To solve the Safe-Pass Bias, we redefine the target variable from binary completion to continuous value generation. We utilize an Expected Threat (xT) grid, where $xT(x,y)$ represents the probability of a team scoring a goal in the next several actions starting from coordinates $(x,y)$.

For a given pressure event ending at coordinates $E = (x_e, y_e)$:
$$ V_{raw} = \mathbb{I}_{success} \times xT(x_e, y_e) $$
Where $\mathbb{I}_{success}$ is $1$ if possession is retained (e.g., successful pass or dribble), and $0$ if dispossessed.

### 4.1 Scaling for Beta Regression
The Beta distribution is defined strictly on the open interval $(0, 1)$. It cannot process exact zeroes (turnovers) or values above 1. Therefore, we scale $V_{raw}$ relative to a theoretical maximum value $V_{max}$ and apply a vanishingly small smoothing factor $\epsilon = 10^{-6}$:
$$ V_i = \left( \frac{V_{raw, i}}{V_{max}} \right) (1 - 2\epsilon) + \epsilon $$
This transformation ensures all outcomes cleanly fit the Beta support without distorting the relative differences in value generation between elite and poor decisions.

---

## 5. Bayesian Hierarchical Beta Regression

To extract the true, underlying composure trait of each player, we construct a Hierarchical Beta Regression model using PyMC and NumPyro.

### 5.1 Why Beta Regression?
Linear regression assumes homoskedasticity and normally distributed residuals with infinite support $(-\infty, \infty)$. Our target $V_i$ is bounded. The Beta regression appropriately models the variance constraint near the boundaries (ceiling and floor effects) and naturally handles continuous probability-like variables.

### 5.2 Likelihood Function
We parameterize the Beta distribution using a mean $\mu_i$ and a global precision (dispersion) parameter $\kappa$:
$$ V_i \sim \text{Beta}(\alpha=\mu_i \kappa, \beta=(1 - \mu_i) \kappa) $$

### 5.3 The Linear Predictor and Link Function
The mean $\mu_i$ is mapped from the linear predictor $\eta_i$ via the inverse-logit link function:
$$ \mu_i = \text{logit}^{-1}(\eta_i) = \frac{1}{1 + e^{-\eta_i}} $$

The linear predictor $\eta_i$ structurally separates the spatial difficulty, tactical role, and individual skill:
$$ \eta_i = \alpha + X_i \beta + \gamma_{pos[i]} + \theta_{player[i]} + \delta_{opp[i]} + \zeta_{comp[i]} $$

Where:
* $\alpha$: The global intercept (baseline log-odds of value retention).
* $X_i \beta$: The dot product of the standardized geometric features and their fixed-effect coefficients.
* $\gamma_{pos[i]}$: The fixed effect for the player's position group.
* **$\theta_{player[i]}$: The Pressure Resistance Score (PRS).** A player-specific random effect measuring innate ability to preserve value above expectation.
* $\delta_{opp[i]}$: A random effect capturing the specific defensive quality/intensity of the opposing team.
* $\zeta_{comp[i]}$: A random effect capturing systemic variance between different tournaments (e.g., World Cup vs. MLS).

### 5.4 Priors and Non-Centered Parameterization
We enforce weakly informative priors to provide regularization (shrinkage), preventing the model from overfitting players with small sample sizes. To optimize the geometry for the Hamiltonian Monte Carlo (NUTS) sampler and prevent divergent transitions, we utilize non-centered parameterizations for the random effects:

**Fixed Effects:**
* $\alpha \sim \mathcal{N}(0, 1.5)$
* $\beta \sim \mathcal{N}(0, 1.0)$
* $\gamma_{pos} \sim \mathcal{N}(0, 1.0)$

**Random Effects (Non-Centered):**
* $\sigma_\theta \sim \text{Exponential}(1.0)$
* $\tilde{\theta}_{player} \sim \mathcal{N}(0, 1)$
* $\theta_{player} = \tilde{\theta}_{player} \times \sigma_\theta$

*(The same non-centered structure is applied to $\delta_{opp}$ and $\zeta_{comp}$.)*

**Dispersion:**
* $\kappa \sim \text{Exponential}(1.0)$

---

## 6. Interpretability Framework

By fully sampling the joint posterior distribution, the framework transitions from abstract math to tangible coaching insights.

### 6.1 Individual Conditional Expectation (ICE)
ICE curves map exactly how an individual player's expected success decays as pressure increases. We achieve this by holding all geometric features in $X$ at their population means, sweeping a single feature (e.g., $d_{min}$) across its domain, and pushing the vectors through the posterior link function alongside the specific player's $\theta$ and $\gamma_{pos}$.

### 6.2 Counterfactuals and "Best Under" Scenarios
To identify a player's specialized composure profile, we define orthogonal geometric matrices representing specific tactical situations (e.g., "Front_Tight", "Back_Loose"). 
For a given Scenario $S$, we calculate the player's expected value:
$$ \mathbb{E}[V | S, g, \theta] = \text{logit}^{-1}(\alpha + X_{S}\beta + \gamma_{g} + \theta) \times V_{max} $$
We subtract the positional population baseline ($\theta=0$) from this value. The scenario yielding the highest positive delta is assigned as the player's "Best Under" scenario, providing scouts with explicit geometric profiles of *where* a player thrives under pressure.

---

## 7. Validation Strategy

Evaluating performance via in-sample loss functions (like AUC or RMSE on the training set) is insufficient for proving a metric represents a scoutable, intrinsic player trait rather than statistical noise. The PRS utilizes **Out-of-Sample Residual Correlation**, which is considered the gold standard in sports analytics.

1. **Prediction Generation:** The model is trained on a diverse set of competitions (e.g., European top leagues, Euros, World Cup).
2. **Holdout Evaluation:** A completely separate competition with a distinct tactical ecosystem (e.g., MLS 2023) is withheld. For every pressure event $j$ in the MLS set, we calculate the expected value generated by an *average* player in that exact position and spatial geometry:
   $$ \hat{V}_j = \mathbb{E}[ \text{logit}^{-1}(\alpha + X_j \beta + \gamma_{pos[j]}) ] \times V_{max} $$
3. **Residual Aggregation:** We calculate the true value generated above expected for that event ($r_j = V_{true, j} - \hat{V}_j$).
4. **Correlation:** We average the residuals per player in the MLS holdout ($\bar{r}_{player}$) and correlate this with their training PRS ($\theta_{player}$) derived from the European/International datasets.

A significant positive correlation (Pearson $r$, $p < 0.01$) empirically proves that $\theta$ accurately measures a stable, persistent cognitive trait that transfers across entirely different tactical environments, leagues, and continents.