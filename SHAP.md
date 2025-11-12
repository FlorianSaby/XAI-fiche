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
