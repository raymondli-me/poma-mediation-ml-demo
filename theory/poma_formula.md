# PoMA Formula: Theory and Corrections

## Basic Formula

The Percentage of Mediated Accuracy (PoMA) quantifies how much of X's effect on Y is mediated through M:

```
PoMA = 1 - β_direct/β_total
```

Where:
- `β_total`: Total effect of X on Y
- `β_direct`: Direct effect of X on Y (controlling for M)

## Connection to Double Machine Learning (DML)

Under the Frisch-Waugh-Lovell theorem, the direct effect equals the coefficient from residual-on-residual regression:

```
β_direct = β_DML = Cov(e_Y, e_X) / Var(e_X)
```

Where:
- `e_Y = Y - E[Y|M]`: Residuals of Y after removing M's effect
- `e_X = X - E[X|M]`: Residuals of X after removing M's effect

## Complete Formula with Corrections

For finite samples, the formula requires correction terms:

```
PoMA = 1 - [(1 - Cov(Ŷ,X̂)/Cov(Y,X) - C1 - C2) / (1 - Var(X̂)/Var(X) - C3)]
```

### Correction Terms

**C1: Y Orthogonality Violation**
```
C1 = Cov(e_Y, X̂) / Cov(Y,X)
```
Captures correlation between Y residuals and predicted X values.

**C2: X Orthogonality Violation**
```
C2 = Cov(e_X, Ŷ) / Cov(Y,X)
```
Captures correlation between X residuals and predicted Y values.

**C3: X Self-Correlation**
```
C3 = 2·Cov(e_X, X̂) / Var(X)
```
Captures remaining correlation between X residuals and predictions.

## Why Linear Methods Fail

When relationships are symmetric/non-linear:
1. Linear correlations → 0
2. Linear regression finds no relationship
3. β_total ≈ 0, making PoMA undefined or meaningless

Example: If X → M = X² and M → Y = M²:
- Corr(X,M) ≈ 0 (symmetric U-shape)
- Corr(M,Y) ≈ 0 (symmetric U-shape)
- Corr(X,Y) ≈ 0
- Linear regression: "No relationships exist"
- ML correctly identifies: ~100% mediation through M

## Practical Implications

1. **Always test for non-linearity** before interpreting mediation results
2. **Use ML methods** when relationships might be non-linear
3. **Cross-validate** to ensure orthogonality in DML
4. **Include correction terms** for finite sample accuracy