# **Integrated Gradients**  

---

## **Purpose**

Integrated Gradients (IG) is an *explainable AI (XAI)* method designed to attribute a model’s prediction to its input features.  
It helps answer the question:

> *“Which parts of my input most influenced the model’s prediction?”*

---

## **Principle**

The key idea of Integrated Gradients is to measure how much each input feature contributes to the model’s output by accumulating gradients along a straight path from a **baseline input** to the **actual input**.

- The **baseline input** represents an absence of signal (e.g., a black image, a zero vector, or an empty text).  
- The **actual input** is the data point being explained.

The method computes the average gradient of the model output with respect to the input, as we gradually move from the baseline to the actual input.  
By integrating these gradients, we obtain smooth, stable, and interpretable feature attributions.

---

## **Prerequisites**

Before applying Integrated Gradients, ensure that:

- You have a differentiable model (e.g., a neural network).  
- You can compute the gradients of the model’s output with respect to its input.  
- You have a baseline input that represents “no information.”  
- You have a specific scalar output that you want to interpret (e.g., class index for classification).

---

## **Step-by-Step Procedure**

### **Step 1 — Choose a baseline**

Select a baseline input \( x' \), such as:

- A black image (all zeros) for image models.  
- A zero or neutral embedding for text models.  
- The mean or zero vector for tabular data.

---

### **Step 2 — Interpolate between baseline and input**

Create intermediate inputs between the baseline \( x' \) and the actual input \( x \):

```math
x^{(k)} = x' + \frac{k}{m}(x - x'), \quad \text{for } k = 0, 1, ..., m
```
## **Step 3 — Compute gradients**

For each intermediate input \( x^{(k)} \), compute the gradient of the model’s output (for the target class) with respect to the input:

```math
\nabla F(x^{(k)})
```
## **Step 4 — Integrate gradients**

Approximate the integral of the gradients along the path using the average of these gradients:

```math
IG(x) = (x - x') \times \frac{1}{m} \sum_{k=1}^{m} \nabla F(x^{(k)})
```
This yields a vector of attribution scores—one for each input feature.

### Step 5 — Interpret results

- Positive attribution → feature supports the prediction.
- Negative attribution → feature opposes the prediction.
- Zero (or near-zero) attribution → little or no effect on the prediction.

  ## Pseudocode

```python
# Inputs: model F, input x, baseline x', target_class, steps m

integrated_gradients = 0

for k in range(1, m + 1):
    alpha = k / m
    x_scaled = x' + alpha * (x - x')
    gradient = compute_gradient(F, x_scaled, target_class)
    integrated_gradients += gradient
integrated_gradients = (x - x') * (integrated_gradients / m)
return integrated_gradients

```



## Best Practices

- Select a baseline that represents the absence of meaningful input, such as a black image for visual models or the `[PAD]` token embedding for LLMs.
- Use sufficient interpolation steps (typically 20–300).
- Normalize or visualize attributions to enhance interpretability.
- Combine with other XAI methods (e.g., Grad-CAM, SHAP) for complementary insights.

## Example Summary

| Feature / Region | Attribution Value | Interpretation |
|-----------------|-----------------|----------------|
| Cat’s face region | +0.35 | Strongly supports “cat” prediction |
| Background | 0.00 | No influence |
| Dog nearby | -0.12 | Opposes “cat” prediction |

## Advantages and Limitations

### Advantages

- Theoretically sound (satisfies sensitivity and implementation invariance).
- Produces smooth, stable attributions compared to raw gradients.
- Works well with deep neural networks.
- Provides per-feature or per-pixel importance.

### Limitations

- Requires access to model gradients (not model-agnostic).
- Sensitive to baseline choice.
- Computationally intensive for high-dimensional data.
- Only applicable to differentiable models.

## Summary

Integrated Gradients is a principled and reliable method for interpreting deep learning models.  
By integrating gradients along a path from a baseline to the actual input, it quantifies how much each input feature contributes to the model’s prediction.  
This makes IG a powerful tool for understanding, debugging, and trusting neural network behavior.

