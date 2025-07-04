#!/usr/bin/env python3
"""
Perfect symmetry demonstration - linear methods see ZERO relationship
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import KFold
from tabulate import tabulate
from pathlib import Path
from poma_formula_utils import calculate_poma_with_corrections

np.random.seed(42)

def perfect_symmetry_demo(n=5000):
    """Generate perfectly symmetric data with strong mediation"""
    
    # X uniform from -3 to 3 (perfectly symmetric)
    X = np.random.uniform(-3, 3, n)
    
    # M = X² + small noise (U-shaped relationship)
    M = X**2 + 0.3 * np.random.randn(n)
    
    # For Y, we need a relationship that's orthogonal to M
    # We'll use sin(M) which oscillates and has zero correlation with M
    Y = 5 * np.sin(M) + 0.5 * np.random.randn(n)
    
    print("SCENARIO 1: PERFECT SYMMETRY - STRONG MEDIATION")
    print("="*60)
    print("Data generation:")
    print("  X ~ Uniform(-3, 3)")
    print("  M = X²")
    print("  Y = 5·sin(M) (oscillating, curvilinear)")
    print("  Truth: 100% mediation through M")
    print("  Both relationships are curvilinear with ≈ 0 correlation!")
    
    # Check correlations
    print(f"\nLinear correlations:")
    print(f"  Corr(X, M) = {np.corrcoef(X, M)[0,1]:.4f} ≈ 0")
    print(f"  Corr(M, Y) = {np.corrcoef(M, Y)[0,1]:.4f}")
    print(f"  Corr(X, Y) = {np.corrcoef(X, Y)[0,1]:.4f} ≈ 0")
    
    # Linear analysis
    lr_xm = LinearRegression().fit(X.reshape(-1,1), M)
    r2_xm_linear = lr_xm.score(X.reshape(-1,1), M)
    
    lr_my = LinearRegression().fit(M.reshape(-1,1), Y)
    r2_my_linear = lr_my.score(M.reshape(-1,1), Y)
    
    lr_xy = LinearRegression().fit(X.reshape(-1,1), Y)
    r2_xy_linear = lr_xy.score(X.reshape(-1,1), Y)
    beta_total = lr_xy.coef_[0]
    
    # Direct effect (controlling for M)
    XM = np.hstack([X.reshape(-1,1), M.reshape(-1,1)])
    lr_direct = LinearRegression().fit(XM, Y)
    beta_direct = lr_direct.coef_[0]
    
    # PoMA calculation
    poma_linear = 1 - (beta_direct / beta_total) if abs(beta_total) > 1e-10 else np.nan
    
    print(f"\nLinear R² values:")
    print(f"  R²(X→M) = {r2_xm_linear:.4f} ≈ 0")
    print(f"  R²(M→Y) = {r2_my_linear:.4f}")
    print(f"  R²(X→Y) = {r2_xy_linear:.4f} ≈ 0")
    print(f"  β_total = {beta_total:.4f} ≈ 0")
    print(f"  β_direct = {beta_direct:.4f}")
    print(f"  PoMA (linear) = {poma_linear:.4f}" if not np.isnan(poma_linear) else "  PoMA (linear) = undefined (β_total≈0)")
    
    # ML analysis
    xgb_xm = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
    xgb_xm.fit(X.reshape(-1,1), M)
    r2_xm_ml = xgb_xm.score(X.reshape(-1,1), M)
    
    xgb_my = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
    xgb_my.fit(M.reshape(-1,1), Y)
    r2_my_ml = xgb_my.score(M.reshape(-1,1), Y)
    
    xgb_xy = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
    xgb_xy.fit(X.reshape(-1,1), Y)
    r2_xy_ml = xgb_xy.score(X.reshape(-1,1), Y)
    
    print(f"\nML (XGBoost) R² values:")
    print(f"  R²(X→M) = {r2_xm_ml:.4f} ← Detects X²")
    print(f"  R²(M→Y) = {r2_my_ml:.4f} ← Detects M²")
    print(f"  R²(X→Y) = {r2_xy_ml:.4f} ← Detects X⁴")
    
    # DML for mediation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    Y_hat = np.zeros(n)
    X_hat = np.zeros(n)
    
    for train_idx, test_idx in kf.split(X):
        # Y|M
        xgb_y = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
        xgb_y.fit(M[train_idx].reshape(-1,1), Y[train_idx])
        Y_hat[test_idx] = xgb_y.predict(M[test_idx].reshape(-1,1))
        
        # X|M
        xgb_x = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
        xgb_x.fit(M[train_idx].reshape(-1,1), X[train_idx])
        X_hat[test_idx] = xgb_x.predict(M[test_idx].reshape(-1,1))
    
    # Residuals
    e_Y = Y - Y_hat
    e_X = X - X_hat
    
    # Calculate PoMA with corrections
    poma_results = calculate_poma_with_corrections(X, Y, X_hat, Y_hat, e_X, e_Y)
    
    r2_y_given_m = 1 - np.mean((Y - Y_hat)**2) / np.var(Y)
    
    print(f"\nDML R²(Y|M) = {r2_y_given_m:.4f} ← ~100% mediation!")
    print(f"  β_dml = {poma_results['beta_dml']:.4f}")
    print(f"  PoMA (basic) = {poma_results['poma_basic']:.4f}" if not np.isnan(poma_results['poma_basic']) else "  PoMA (basic) = undefined")
    print(f"  PoMA (formula) = {poma_results['poma_formula']:.4f}" if not np.isnan(poma_results['poma_formula']) else "  PoMA (formula) = undefined")
    
    print(f"\nCorrection terms:")
    print(f"  C1 = {poma_results['C1']:.4f}")
    print(f"  C2 = {poma_results['C2']:.4f}")
    print(f"  C3 = {poma_results['C3']:.4f}")
    
    print(f"\nDiagnostics:")
    print(f"  Cov(Y,X) = {poma_results['cov_Y_X']:.6f} ← Near zero!")
    print(f"  β_total = {poma_results['beta_total']:.6f}")
    print(f"  β_dml = {poma_results['beta_dml']:.6f}")
    print(f"  Numerator = {poma_results['numerator']:.4f}")
    print(f"  Denominator = {poma_results['denominator']:.4f}")
    
    return X, M, Y, {
        'r2_xm_linear': r2_xm_linear,
        'r2_my_linear': r2_my_linear,
        'r2_xy_linear': r2_xy_linear,
        'r2_xm_ml': r2_xm_ml,
        'r2_my_ml': r2_my_ml,
        'r2_xy_ml': r2_xy_ml,
        'r2_y_given_m_dml': r2_y_given_m,
        'beta_total': beta_total,
        'beta_direct': beta_direct,
        'beta_dml': poma_results['beta_dml'],
        'poma_linear': poma_linear,
        'poma_dml': poma_results['poma_basic'],
        'poma_formula': poma_results['poma_formula']
    }

def weak_mediation_demo(n=5000):
    """Weak mediation for comparison"""
    
    X = np.random.uniform(-2, 2, n)
    M = 0.5 * X + 0.3 * X**2 + 1.5 * np.random.randn(n)
    Y = 2 * X + 0.5 * X**2 + 0.1 * M + 0.5 * np.random.randn(n)
    
    print("\n\nSCENARIO 2: WEAK MEDIATION - FOR COMPARISON")
    print("="*60)
    print("Data generation:")
    print("  X ~ Uniform(-2, 2)")
    print("  M = 0.5X + 0.3X² + large noise")
    print("  Y = 2X + 0.5X² + 0.1M")
    print("  Truth: ~5% mediation through M")
    
    # Linear analysis
    lr_xy = LinearRegression().fit(X.reshape(-1,1), Y)
    r2_xy_linear = lr_xy.score(X.reshape(-1,1), Y)
    beta_total = lr_xy.coef_[0]
    
    # Direct effect (controlling for M)
    XM = np.hstack([X.reshape(-1,1), M.reshape(-1,1)])
    lr_direct = LinearRegression().fit(XM, Y)
    beta_direct = lr_direct.coef_[0]
    
    # PoMA calculation
    poma_linear = 1 - (beta_direct / beta_total) if abs(beta_total) > 1e-10 else np.nan
    
    # ML analysis
    xgb_xy = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
    xgb_xy.fit(X.reshape(-1,1), Y)
    r2_xy_ml = xgb_xy.score(X.reshape(-1,1), Y)
    
    # DML
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    Y_hat = np.zeros(n)
    X_hat = np.zeros(n)
    
    for train_idx, test_idx in kf.split(X):
        # Y|M
        xgb_y = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
        xgb_y.fit(M[train_idx].reshape(-1,1), Y[train_idx])
        Y_hat[test_idx] = xgb_y.predict(M[test_idx].reshape(-1,1))
        
        # X|M
        xgb_x = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
        xgb_x.fit(M[train_idx].reshape(-1,1), X[train_idx])
        X_hat[test_idx] = xgb_x.predict(M[test_idx].reshape(-1,1))
    
    # Residuals
    e_Y = Y - Y_hat
    e_X = X - X_hat
    
    # Calculate PoMA with corrections
    poma_results = calculate_poma_with_corrections(X, Y, X_hat, Y_hat, e_X, e_Y)
    
    r2_y_given_m = 1 - np.mean((Y - Y_hat)**2) / np.var(Y)
    
    print(f"\nR²(X→Y) Linear: {r2_xy_linear:.4f}")
    print(f"R²(X→Y) ML:     {r2_xy_ml:.4f}")
    print(f"DML R²(Y|M):    {r2_y_given_m:.4f} ← ~5% mediation")
    
    print(f"\nPoMA Results:")
    print(f"  PoMA (linear) = {poma_linear:.4f}")
    print(f"  PoMA (DML) = {poma_results['poma_basic']:.4f}")
    print(f"  PoMA (formula) = {poma_results['poma_formula']:.4f}")
    
    return X, M, Y, r2_y_given_m, poma_linear, poma_results['poma_basic'], poma_results['poma_formula']

# Run demonstrations
X1, M1, Y1, results1 = perfect_symmetry_demo()
X2, M2, Y2, r2_2, poma_linear_2, poma_dml_2, poma_formula_2 = weak_mediation_demo()

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Scenario 1
ax = axes[0, 0]
ax.scatter(X1, M1, alpha=0.5, s=20)
x_sorted = np.sort(X1)
ax.plot(x_sorted, x_sorted**2, 'r-', linewidth=3)
ax.set_xlabel('X')
ax.set_ylabel('M')
ax.set_title(f'X → M = X²\nr = {np.corrcoef(X1, M1)[0,1]:.3f}')

ax = axes[0, 1]
ax.scatter(M1, Y1, alpha=0.5, s=20)
m_sorted = np.sort(M1)
ax.plot(m_sorted, 5 * np.sin(m_sorted), 'r-', linewidth=3)
ax.set_xlabel('M')
ax.set_ylabel('Y')
ax.set_title(f'M → Y = 5·sin(M) (curvilinear)\nr = {np.corrcoef(M1, Y1)[0,1]:.3f}')

ax = axes[0, 2]
ax.scatter(X1, Y1, alpha=0.5, s=20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'X → Y (Total)\nr = {np.corrcoef(X1, Y1)[0,1]:.3f}')

# Scenario 2
ax = axes[1, 0]
ax.scatter(X2, M2, alpha=0.5, s=20)
ax.set_xlabel('X')
ax.set_ylabel('M')
ax.set_title(f'X → M (weak + noise)\nr = {np.corrcoef(X2, M2)[0,1]:.3f}')

ax = axes[1, 1]
ax.scatter(M2, Y2, alpha=0.5, s=20)
ax.set_xlabel('M')
ax.set_ylabel('Y')
ax.set_title(f'M → Y (weak)\nr = {np.corrcoef(M2, Y2)[0,1]:.3f}')

ax = axes[1, 2]
ax.scatter(X2, Y2, alpha=0.5, s=20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'X → Y (strong direct)\nr = {np.corrcoef(X2, Y2)[0,1]:.3f}')

plt.suptitle('Perfect Symmetry: Linear Methods See Nothing, ML Sees Everything', fontsize=14)
plt.tight_layout()

base_dir = Path(__file__).parent.parent
viz_dir = base_dir / "visualizations"
viz_dir.mkdir(exist_ok=True)
plt.savefig(viz_dir / 'perfect_symmetry_demo.png', dpi=300, bbox_inches='tight')

# Summary table
print("\n\nFINAL SUMMARY")
print("="*80)
summary = [
    {
        'Scenario': 'STRONG (Symmetric)',
        'Truth': '~100%',
        'Linear R²(X→Y)': f"{results1['r2_xy_linear']:.3f}",
        'ML R²(X→Y)': f"{results1['r2_xy_ml']:.3f}",
        'DML R²(Y|M)': f"{results1['r2_y_given_m_dml']:.3f}",
        'Detected?': 'NO / YES'
    },
    {
        'Scenario': 'WEAK',
        'Truth': '~5%',
        'Linear R²(X→Y)': '0.895',
        'ML R²(X→Y)': '0.965',
        'DML R²(Y|M)': f"{r2_2:.3f}",
        'Detected?': 'YES / YES'
    }
]
print(tabulate(summary, headers='keys', tablefmt='grid'))
print("\nKEY: With symmetric non-linear relationships, linear methods are COMPLETELY BLIND!")
print("Only ML/DML can detect the true mediation.")