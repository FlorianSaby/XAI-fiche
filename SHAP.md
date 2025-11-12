# **Shapley Values (SHAP)**

---

## Purpose

SHapley Additive exPlanations (SHAP) is a *model-agnostic* explainable AI framework that attributes a model’s prediction to its input features based on principles from cooperative game theory.  
It answers the question:

> *“How much did each feature contribute to this particular prediction compared to a baseline?”*

Unlike gradient-based interpretability methods, Shapley values provide theoretically sound, fair, and consistent feature attributions by evaluating all possible combinations of features.

---

## Principle

Intuitively, SHAP measures how the inclusion of feature `i` influences the model's prediction across all possible subsets of features. The corresponding SHAP value `φ_i` is computed as the weighted average of these effects, where the weights depend on the cardinality of each subset.

Formally, the Shapley value for feature `i` is given by:

```math
φ_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! \, (|F|-|S|-1)!}{|F|!} \left[ f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S) \right]
```
Where:

- `φ_i`: Shapley value (attribution) for feature `i`  
- `F`: full set of features  
- `S`: any subset of features excluding `i`  
- `f_S(x_S)`: model output when only features in `S` are available, with others replaced by baseline or expected values  

SHAP values satisfy three fundamental axioms ensuring fairness and consistency:

1. **Efficiency**: The sum of all feature attributions equals the difference between the actual and expected model predictions:

```math
\sum_{i \in F} \phi_i = f(x) - \mathbb{E}[f(X)]
```
- **Symmetry**: Features that contribute identically in all cases receive identical attributions.  
- **Additivity**: For composite or ensemble models, attributions from individual components combine linearly.  

---

## Prerequisites

Before computing SHAP values, you need:

- A trained model (any architecture: trees, linear models, neural networks, etc.)  
- Ability to query the model with partial inputs (or use approximation)  
- A *background dataset* (or single reference point) defining "absent" feature values  

---

## Step-by-Step Procedure

### Step 1 — Define baseline / background

Select a representative background dataset (e.g., training data mean, median, or random sample) to impute missing features.

### Step 2 — Enumerate subsets

For each feature `i`, generate all coalitions `S ⊆ F \ {i}`.  
Evaluate the model:

- `f_S(x_S)`: prediction with features in `S` (others set to background)  
- `f_{S ∪ {i}}(x_{S ∪ {i}})`: prediction with `S` *plus* the true value of `i`
### Step 3 — Compute marginal contributions

```math
\Delta_i(S) = f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S)
```
### Step 4 — Average contributions (weighted)

```math
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! \, (|F|-|S|-1)!}{|F|!} \Delta_i(S)
```
### Step 5 — Interpret results

- `φ_i > 0`: feature *pushes prediction upward*  
- `φ_i < 0`: feature *pushes prediction downward*  
- `φ_i ≈ 0`: feature has *negligible impact*  

---

## Pseudocode

```python
# Simplified Conceptual SHAP Computation with Comments
from itertools import combinations
import math

# Generate all possible subsets (the powerset) of the given feature set
def powerset(features):
    return [list(s) for r in range(len(features)+1) for s in combinations(features, r)]

def shapley_values(model, x, background, features):
    n = len(features)
    shap = {f: 0.0 for f in features}  # Initialize SHAP values for all features

    # Iterate over each feature i to compute its Shapley value φ_i
    for i in features:
        others = [f for f in features if f != i]

        # Consider every subset S of the remaining features
        for S in powerset(others):

            # Compute Shapley weighting factor:
            # weight = |S|!(n - |S| - 1)! / n!
            weight = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)

            # --- Model predictions ---
            # 1. Prediction using only features in subset S
            x_S = background.mean(axis=0) # Approximate missing features with background mean
            pred_S = model.predict(x_S.reshape(1, -1))[0]

            # 2. Prediction using features in S plus feature i
            x_Si = x_S.copy()
            x_Si[i] = x[i] # Add feature i’s actual value
            pred_Si = model.predict(x_Si.reshape(1, -1))[0]

            # Accumulate weighted contribution of feature i
            shap[i] += weight * (pred_Si - pred_S)

    return shap

```
## Best Practices

- Use `TreeSHAP` for tree-based models (exact, polynomial time)  
- Apply `KernelSHAP` or `DeepSHAP` for black-box or neural models  
- Select background data that reflects typical input distribution  
- Generate *summary plots*, *dependence plots*, and *waterfall plots* for insight  
- Validate attributions against domain expertise and sensitivity checks  

---

## Example Summary

| Feature         | SHAP Value | Interpretation                     |
|-----------------|------------|-----------------------------------|
| Age             | +0.15      | Increases predicted risk by 0.15  |
| Cholesterol     | -0.08      | Decreases predicted risk by 0.08  |
| Blood Pressure  | +0.20      | Strongly increases predicted risk |
| Gender          | 0.00       | No measurable effect               |

---

## Advantages and Limitations

### Advantages

- Model-agnostic with rigorous theoretical foundation  
- Fairly distributes credit across all feature interactions  
- Local (per-instance) and global (summary) explanations  
- Additive across ensemble or stacked models  

### Limitations

- Exact computation is $O(2^{|F|})$ — infeasible beyond ~20 features.  

  **Solution:** Use approximation methods such as *KernelSHAP*, *Sampling-based SHAP*, or *TreeSHAP* for tree models, which drastically reduce computation while maintaining accuracy.  
- Sensitive to background dataset choice  
- May misattribute in presence of strong feature correlations (use SHAP interaction values to diagnose)  

---

## Summary

Shapley Values (SHAP) offer a *unified, game-theoretic framework* for fair feature attribution in machine learning. By marginalizing over all possible feature coalitions, SHAP captures both direct effects and complex interactions — making it an essential tool for building trust and transparency in high-stakes AI systems.
