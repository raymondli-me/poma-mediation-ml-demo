#!/usr/bin/env python3
"""
Utility functions for calculating PoMA with correction terms
"""

import numpy as np

def calculate_poma_with_corrections(X, Y, X_hat, Y_hat, e_X, e_Y, ddof=1):
    """
    Calculate PoMA with C1, C2, C3 corrections
    
    PoMA = 1 - [(1 - Cov(Ŷ,X̂)/Cov(Y,X) - C1 - C2) / (1 - Var(X̂)/Var(X) - C3)]
    
    WARNING: This formula assumes meaningful linear relationships exist.
    When Cov(Y,X) ≈ 0 (as with symmetric non-linear relationships),
    the correction terms can explode and give nonsensical results.
    
    Parameters:
    -----------
    X : array-like
        Original X values
    Y : array-like 
        Original Y values
    X_hat : array-like
        Predicted X values from M
    Y_hat : array-like
        Predicted Y values from M
    e_X : array-like
        X residuals (X - X_hat)
    e_Y : array-like
        Y residuals (Y - Y_hat)
    ddof : int
        Degrees of freedom for covariance calculation
        
    Returns:
    --------
    dict with:
        - poma_formula: PoMA with corrections
        - poma_basic: Basic PoMA (1 - β_dml/β_total)
        - C1, C2, C3: Correction terms
        - numerator, denominator: Formula components
        - beta_total, beta_dml: Coefficients
    """
    
    # Ensure 1D arrays
    X = X.ravel()
    Y = Y.ravel()
    X_hat = X_hat.ravel()
    Y_hat = Y_hat.ravel()
    e_X = e_X.ravel()
    e_Y = e_Y.ravel()
    
    # Calculate covariances and variances
    cov_Y_X = np.cov(Y, X, ddof=ddof)[0, 1]
    cov_Y_hat_X_hat = np.cov(Y_hat, X_hat, ddof=ddof)[0, 1]
    var_X = np.var(X, ddof=ddof)
    var_X_hat = np.var(X_hat, ddof=ddof)
    
    # Correction terms
    C1 = np.cov(e_Y, X_hat, ddof=ddof)[0, 1] / cov_Y_X if abs(cov_Y_X) > 1e-10 else 0
    C2 = np.cov(e_X, Y_hat, ddof=ddof)[0, 1] / cov_Y_X if abs(cov_Y_X) > 1e-10 else 0
    C3 = 2 * np.cov(e_X, X_hat, ddof=ddof)[0, 1] / var_X if var_X > 1e-10 else 0
    
    # Formula components
    numerator = 1 - cov_Y_hat_X_hat / cov_Y_X - C1 - C2 if abs(cov_Y_X) > 1e-10 else np.nan
    denominator = 1 - var_X_hat / var_X - C3 if var_X > 1e-10 else np.nan
    
    # PoMA with corrections
    if not np.isnan(numerator) and not np.isnan(denominator) and abs(denominator) > 1e-10:
        poma_formula = numerator / denominator
    else:
        poma_formula = np.nan
        
    # Also calculate basic PoMA for comparison
    beta_total = cov_Y_X / var_X if var_X > 1e-10 else np.nan
    
    # DML coefficient (residual on residual)
    var_e_X = np.var(e_X, ddof=ddof)
    if var_e_X > 1e-10:
        beta_dml = np.cov(e_Y, e_X, ddof=ddof)[0, 1] / var_e_X
    else:
        beta_dml = 0
        
    poma_basic = 1 - (beta_dml / beta_total) if abs(beta_total) > 1e-10 else np.nan
    
    return {
        'poma_formula': poma_formula,
        'poma_basic': poma_basic,
        'C1': C1,
        'C2': C2,
        'C3': C3,
        'numerator': numerator,
        'denominator': denominator,
        'beta_total': beta_total,
        'beta_dml': beta_dml,
        'cov_Y_X': cov_Y_X,
        'var_X': var_X,
        'var_X_hat': var_X_hat,
        'cov_Y_hat_X_hat': cov_Y_hat_X_hat
    }