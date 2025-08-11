
# **Ridge Regression **

---

## **1. Introduction to Ridge Regression**

### **What is Ridge Regression?**

* Ridge Regression is a **regularized** version of **Linear Regression**.
* It adds a **penalty** (L2 regularization) to the cost function of Ordinary Least Squares (OLS).
* Goal: Prevent overfitting and improve model generalization.

**Key Idea:**

> Ordinary Linear Regression minimizes the **sum of squared errors** (SSE). Ridge Regression minimizes SSE **plus** a penalty proportional to the sum of squared coefficients.

---

### **Why Do We Need Ridge Regression?**

1. **Overfitting Problem**

   * In OLS, the model may fit the training data too closely, capturing noise.
   * Ridge controls model complexity by shrinking coefficients.

2. **Multicollinearity**

   * When features are highly correlated, OLS coefficients become unstable (large variance).
   * Ridge stabilizes them by adding a penalty term.

3. **Regularization**

   * A technique to improve generalization by penalizing large weights.

---

### **When to Use Ridge Regression**

* When you have **many correlated features**.
* When the dataset is **high-dimensional** and prone to overfitting.
* When you prefer **smaller, more stable coefficients** instead of sparse coefficients.

---

### **Comparison with Lasso & Elastic Net**

| Model           | Penalty Type | Effect                                                                     |
| --------------- | ------------ | -------------------------------------------------------------------------- |
| **Ridge**       | L2           | Shrinks coefficients but keeps all features.                               |
| **Lasso**       | L1           | Shrinks and can set some coefficients to exactly zero (feature selection). |
| **Elastic Net** | L1 + L2      | Combines both effects.                                                     |

---

## **2. Mathematical Formulation**

### **Ordinary Least Squares (OLS)**

Cost function:

$$
J(\beta) = \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2
$$

Matrix form:

$$
J(\beta) = (y - X\beta)^T (y - X\beta)
$$

Closed-form OLS solution:

$$
\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y
$$

---

### **Ridge Regression**

Cost function with L2 penalty:

$$
J(\beta) = (y - X\beta)^T (y - X\beta) + \lambda \sum_{j=1}^p \beta_j^2
$$

Matrix form:

$$
J(\beta) = (y - X\beta)^T (y - X\beta) + \lambda \beta^T \beta
$$

Closed-form Ridge solution:

$$
\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y
$$

Where:

* $\lambda$ (alpha in scikit-learn) ≥ 0 is the regularization parameter.
* $I$ is the identity matrix (size $p \times p$).

---

## **3. Mathematical Intuition**

### **How L2 Penalty Shrinks Coefficients**

* The term $\lambda \beta^T \beta$ penalizes large coefficients.
* Larger $\lambda$ → more shrinkage toward zero.
* No coefficient is exactly zero (unlike Lasso).

---

### **Geometric Interpretation**

* OLS solution: intersection of error contours and unconstrained space.
* Ridge: restricts solution to lie inside an **L2-norm ball** (circle in 2D, sphere in 3D).
* Intersection point is closer to origin → smaller coefficients.

---

### **Effect of λ (alpha)**

* **Small λ (\~0)** → behaves like OLS.
* **Large λ** → coefficients shrink heavily, underfitting may occur.
* λ controls **bias-variance tradeoff**.

---

## **4. Hyperparameter λ (alpha)**

* **Bias-Variance Tradeoff**:

  * Small λ → low bias, high variance.
  * Large λ → high bias, low variance.

* **Choosing λ**:

  * Use **k-fold cross-validation** to find λ that minimizes validation error.

---

## **5. Step-by-Step Numerical Example**

Let’s use a **tiny dataset**:

| x₁ | x₂ | y |
| -- | -- | - |
| 1  | 2  | 4 |
| 2  | 3  | 5 |
| 3  | 4  | 6 |
| 4  | 5  | 7 |

### **Matrix Form**

$$
X =
\begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 3 \\
1 & 3 & 4 \\
1 & 4 & 5
\end{bmatrix}
$$

(First column = intercept term)

$$
y =
\begin{bmatrix}
4 \\ 5 \\ 6 \\ 7
\end{bmatrix}
$$

---

### **Step 1: OLS Coefficients**

$$
\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y
$$

1. $X^T X$

$$
\begin{bmatrix}
4 & 10 & 14 \\
10 & 30 & 40 \\
14 & 40 & 54
\end{bmatrix}
$$

2. $X^T y$

$$
\begin{bmatrix}
22 \\ 65 \\ 87
\end{bmatrix}
$$

3. Invert $X^T X$ → multiply → get OLS coefficients:

$$
\hat{\beta}_{OLS} \approx [2, 1, 0]
$$

---

### **Step 2: Ridge Coefficients (λ = 1)**

$$
\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y
$$

* Add $\lambda I$ to $X^T X$:

$$
\begin{bmatrix}
4+1 & 10 & 14 \\
10 & 30+1 & 40 \\
14 & 40 & 54+1
\end{bmatrix}
$$
$=$
$$
\begin{bmatrix}
5 & 10 & 14 \\
10 & 31 & 40 \\
14 & 40 & 55
\end{bmatrix}
$$

* Invert and multiply with $X^T y$ → coefficients shrink compared to OLS.

---

## Dataset (given)

| row | $x_1$ | $x_2$ | $y$ |
| --- | ----: | ----: | --: |
| 1   |     1 |     2 |   4 |
| 2   |     2 |     3 |   5 |
| 3   |     3 |     4 |   6 |
| 4   |     4 |     5 |   7 |

We include an intercept term in $X$. So the design matrix $X$ (with column 1 = intercept) and $y$ are:

$$
X \;=\;
\begin{bmatrix}
1 & 1 & 2\\[4pt]
1 & 2 & 3\\[4pt]
1 & 3 & 4\\[4pt]
1 & 4 & 5
\end{bmatrix}
,\qquad
y=\begin{bmatrix}4\\5\\6\\7\end{bmatrix}
$$

(Columns: intercept, $x_1$, $x_2$.)

---

## 1) Compute $X^\top X$ and $X^\top y$

$$
X^\top X =
\begin{bmatrix}
4 & 10 & 14\\[4pt]
10 & 30 & 40\\[4pt]
14 & 40 & 54
\end{bmatrix}
\qquad
X^\top y =
\begin{bmatrix}22\\60\\82\end{bmatrix}
$$

(You can verify the entries by summing products row-by-row.)

---

## 2) Try OLS: $\hat\beta_{OLS} = (X^\top X)^{-1} X^\top y$

We attempt to invert $X^\top X$. But:

* $X^\top X$ is **singular** (non-invertible) for this dataset.
  Reason: **perfect multicollinearity** between the two features — here $x_2 = x_1 + 1$ (column 3 is a linear combination of column 2 and the intercept). That makes columns of $X$ linearly dependent and $X^\top X$ singular.

So the usual OLS closed-form cannot be computed (no unique $(X^\top X)^{-1}$ exists).

**This is exactly why regularization (Ridge) is useful.**

---

## 3) A pseudo-inverse (one possible OLS solution)

Although $(X^\top X)^{-1}$ does not exist, the Moore–Penrose **pseudo-inverse** produces one least-squares solution (the minimum-norm solution). Using the pseudo-inverse:

$$
\hat\beta_{\text{pinv}} = X^{+} y \approx
\begin{bmatrix}
1.66666667\\[4pt]
-0.33333333\\[4pt]
1.33333333
\end{bmatrix}
$$

Interpretation: because of perfect collinearity, there are infinitely many OLS solutions; the pseudo-inverse gives the minimum-Euclidean-norm one. But in practice we prefer a stable unique estimator — enter Ridge.

---

## 4) Ridge Regression: formula (matrix form)

$$
\hat{\beta}_{\text{Ridge}} = \bigl(X^\top X + \lambda I\bigr)^{-1} X^\top y
$$

Important practical detail used here: **do not penalize the intercept**. Implementation-wise that's handled by using a diagonal matrix $I$ with the intercept position set to 0 (i.e. $I_{00}=0$), so the intercept is not shrunk.

---

## 5) Compute Ridge for $\lambda=1$

### 5.1 Build the regularized matrix

We add $\lambda I$, but keep intercept unpenalized (so only diagonal entries for feature columns are increased):

$$
X^\top X + \lambda I =
\begin{bmatrix}
4 & 10 & 14\\
10 & 30 & 40\\
14 & 40 & 54
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
4 & 10 & 14\\
10 & 31 & 40\\
14 & 40 & 55
\end{bmatrix}
$$

### 5.2 Inverse of that matrix (numeric)

$$
\bigl(X^\top X + 1\cdot I\bigr)^{-1} \approx
\begin{bmatrix}
2.38636364 & 0.22727273 & -0.77272727\\[4pt]
0.22727273 & 0.54545455 & -0.45454545\\[4pt]
-0.77272727 & -0.45454545 & 0.54545455
\end{bmatrix}
$$

### 5.3 Multiply by $X^\top y$

Compute $\hat\beta_{\text{ridge},\lambda=1} = A^{-1} (X^\top y)$:

$$
\hat\beta_{\text{ridge},\lambda=1} \approx
\begin{bmatrix}
2.77272727\\[4pt]
0.45454545\\[4pt]
0.45454545
\end{bmatrix}
$$

So for $\lambda=1$:

* Intercept $\beta_0 \approx 2.7727$
* $\beta_1 \approx 0.4545$
* $\beta_2 \approx 0.4545$

Notice: $\beta_1$ and $\beta_2$ are equal (because of symmetry in the data since $x_2 = x_1+1$), and they are **shrunken** relative to some unconstrained solutions. Ridge gave a unique stable solution.

---

## 6) Compute Ridge for $\lambda=10$

### 6.1 Regularized matrix

$$
X^\top X + 10 \cdot I =
\begin{bmatrix}
4 & 10 & 14\\
10 & 30 & 40\\
14 & 40 & 54
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 & 0\\
0 & 10 & 0\\
0 & 0 & 10
\end{bmatrix}
=
\begin{bmatrix}
4 & 10 & 14\\
10 & 40 & 40\\
14 & 40 & 64
\end{bmatrix}
$$

### 6.2 Inverse of that matrix (numeric)

$$
\bigl(X^\top X + 10 I\bigr)^{-1} \approx
\begin{bmatrix}
1.200 & -0.100 & -0.200\\[4pt]
-0.100 & 0.075 & -0.025\\[4pt]
-0.200 & -0.025 & 0.075
\end{bmatrix}
$$

### 6.3 Multiply by $X^\top y$

$$
\hat\beta_{\text{ridge},\lambda=10} =
\begin{bmatrix}
4.0\\[4pt]
0.25\\[4pt]
0.25
\end{bmatrix}
$$

So for $\lambda=10$:

* Intercept $\beta_0 = 4.0$
* $\beta_1 = 0.25$
* $\beta_2 = 0.25$

**Interpretation:** Increasing $\lambda$ shrinks the slope coefficients more. The intercept adjusts accordingly.

---

## 7) Compact summary of computed numeric values

* $X^\top X = \begin{bmatrix}4 & 10 & 14\\10 & 30 & 40\\14 & 40 & 54\end{bmatrix}$

* $X^\top y = \begin{bmatrix}22\\60\\82\end{bmatrix}$

* **Pseudo-inverse OLS (one least-norm solution):**

  $$
  \hat\beta_{\text{pinv}} \approx \begin{bmatrix}1.66666667\\-0.33333333\\1.33333333\end{bmatrix}
  $$

  (Note: OLS closed-form inverse cannot be used because $X^\top X$ is singular.)

* **Ridge ($\lambda=1$)**

  $$
  \hat\beta_{\text{ridge},1} \approx \begin{bmatrix}2.77272727\\0.45454545\\0.45454545\end{bmatrix}
  $$

* **Ridge ($\lambda=10$)**

  $$
  \hat\beta_{\text{ridge},10} = \begin{bmatrix}4.0\\0.25\\0.25\end{bmatrix}
  $$

---

## 8) Quick checks / predicted values

Use the $\lambda=1$ Ridge model to predict the first row (intercept + $x_1=1,x_2=2$):

$$
\hat{y}_1 = 2.77272727 + 0.45454545\cdot 1 + 0.45454545\cdot 2
= 2.77272727 + 0.45454545 + 0.90909090
\approx 4.13636362
$$

Actual $y_1=4$. (This demonstrates generalization vs exact fit: because of shrinkage we don’t fit training targets exactly.)

---

## 9) What this calculation demonstrates (key takeaways)

* The dataset has **perfect multicollinearity**: $x_2 = x_1 + 1$. That makes $X^\top X$ singular → OLS closed form fails (no unique solution).
* Ridge adds $\lambda I$ (with intercept not penalized), making $X^\top X + \lambda I$ **invertible** for $\lambda>0$. That yields a **unique**, **stable** solution.
* Larger $\lambda$ → stronger shrinkage → coefficients move toward zero (but typically not exactly zero).
* The intercept is free to adjust when we choose not to penalize it.

---

## 10) If you'd like — I can also show:

* The **full row-by-row arithmetic** for one of the inverse multiplications (i.e. multiply the inverse matrix by $X^\top y$ element-wise so you can see the dot-product sums producing each coefficient).
* A small **NumPy script** that prints every step (matrix entries, inverses, intermediate vectors) so you can run it yourself and verify each numeric step.




## **6. Python Implementation**

### **Manual with NumPy**

```python
import numpy as np

# Data
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
```

---

### **Using scikit-learn**

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## **7. Advantages & Disadvantages**

### ✅ Advantages

* Handles multicollinearity well.
* Improves generalization.
* All features retained (unlike Lasso).

### ❌ Disadvantages

* No feature selection.
* Coefficients shrink but never exactly zero.
* Requires λ tuning.

---

### **Standardizing Features**

* Ridge is sensitive to feature scale.
* Standardize features before fitting:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## **8. Summary Table**

| Aspect                | Ridge Regression  | Lasso Regression |
| --------------------- | ----------------- | ---------------- |
| Penalty               | L2                | L1               |
| Coefficient Shrinkage | Yes               | Yes              |
| Feature Selection     | No                | Yes              |
| Use Case              | Multicollinearity | Sparse models    |
| λ Effect              | More shrinkage    | More zeros       |

---

📌 **Key Formula**:

$$
\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y
$$

📌 **Key Point**: Ridge helps when OLS struggles with **overfitting** or **multicollinearity**, but it won’t remove features.

---
