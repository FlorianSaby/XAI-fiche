# **Integrated Gradients (IG)**  
*Explainable AI (XAI) Method Summary*  

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
