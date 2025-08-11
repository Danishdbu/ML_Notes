# 📊 Ridge Regression

---

## **1. Introduction to Ridge Regression**

### **What is Ridge Regression?**

* Ridge Regression is a **regularized** version of **Linear Regression**.
* It adds a **penalty** (L2 regularization) to the cost function of Ordinary Least Squares (OLS).
* **Goal**: Prevent overfitting and improve model generalization.

> Ordinary Linear Regression minimizes the **sum of squared errors** (SSE).  
> Ridge Regression minimizes SSE **plus** a penalty proportional to the sum of squared coefficients.

---

### **Why Do We Need Ridge Regression?**

1. **Overfitting Problem**
   * In OLS, the model may fit the training data too closely, capturing noise.
   * Ridge controls model complexity by shrinking coefficients.

2. **Multicollinearity**
   * When features are highly correlated, OLS coefficients become unstable (large variance).
   * Ridge stabilizes them by adding a penalty term.

3. **Regularization**
   * Improves generalization by penalizing large weights.

---

### **When to Use Ridge Regression**

* Many correlated features.
* High-dimensional datasets prone to overfitting.
* Prefer smaller, stable coefficients (instead of sparse).

---

### **Comparison with Lasso & Elastic Net**

| Model           | Penalty Type | Effect                                                                     |
| --------------- | ------------ | -------------------------------------------------------------------------- |
| **Ridge**       | L2           | Shrinks coefficients but keeps all features.                               |
| **Lasso**       | L1           | Shrinks and can set some coefficients to exactly zero (feature selection). |
| **Elastic Net** | L1 + L2      | Combines both effects.                                                     |

---

## **2. Mathematical Formulation**

### **OLS Cost Function**

$$
J(\beta) = \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2
$$

Matrix form:

$$
J(\beta) = (y - X\beta)^T (y - X\beta)
$$

Closed-form OLS:

$$
\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y
$$

---

### **Ridge Regression Cost Function**

$$
J(\beta) = (y - X\beta)^T (y - X\beta) + \lambda \sum_{j=1}^p \beta_j^2
$$

Matrix form:

$$
J(\beta) = (y - X\beta)^T (y - X\beta) + \lambda \beta^T \beta
$$

Closed-form Ridge:

$$
\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y
$$

Where:
* $\lambda$ ≥ 0 = regularization parameter (alpha in sklearn).
* $I$ = identity matrix.

---

## **3. Mathematical Intuition**

* $\lambda \beta^T \beta$ penalizes large coefficients.
* Larger $\lambda$ → more shrinkage.
* No coefficient is exactly zero (unlike Lasso).

---

### **Effect of λ (alpha)**

* Small λ → behaves like OLS.
* Large λ → coefficients shrink heavily (possible underfitting).
* λ controls **bias-variance tradeoff**.

---

## **4. Step-by-Step Numerical Example**

Dataset:

| $x_1$ | $x_2$ | $y$ |
| ----- | ----- | --- |
| 1     | 2     | 4   |
| 2     | 3     | 5   |
| 3     | 4     | 6   |
| 4     | 5     | 7   |

Design matrix (with intercept):

$$
X =
\begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 3 \\
1 & 3 & 4 \\
1 & 4 & 5
\end{bmatrix},
\quad
y =
\begin{bmatrix}
4 \\ 5 \\ 6 \\ 7
\end{bmatrix}
$$

---

### **Step 1: Compute $X^T X$ and $X^T y$**

$$
X^T X =
\begin{bmatrix}
4 & 10 & 14 \\
10 & 30 & 40 \\
14 & 40 & 54
\end{bmatrix}
\quad
X^T y =
\begin{bmatrix}
22 \\ 60 \\ 82
\end{bmatrix}
$$

---

### **Step 2: Ridge with λ = 1**

Add λI (no intercept penalty):

$$
\begin{bmatrix}
4 & 10 & 14 \\
10 & 30 & 40 \\
14 & 40 & 54
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
4 & 10 & 14 \\
10 & 31 & 40 \\
14 & 40 & 55
\end{bmatrix}
$$

Invert & multiply:

$$
\hat{\beta}_{ridge} \approx
\begin{bmatrix}
2.7727 \\
0.4545 \\
0.4545
\end{bmatrix}
$$

---

### **Step 3: Ridge with λ = 10**

$$
\hat{\beta}_{ridge} \approx
\begin{bmatrix}
4.0 \\
0.25 \\
0.25
\end{bmatrix}
$$

---

## **5. Python Implementation**

### Manual with NumPy
```python
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([4, 5, 6, 7])

# Add intercept
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Ridge solution
lam = 1
I = np.eye(X_b.shape[1])
I[0, 0] = 0  # Don't regularize intercept

beta_ridge = np.linalg.inv(X_b.T @ X_b + lam * I) @ X_b.T @ y
print(beta_ridge)
