# PoMA (Percentage of Mediated Accuracy) - ML vs Linear Methods Demonstration

This repository demonstrates critical findings about mediation analysis using Machine Learning (ML) vs traditional linear methods. The key insight: **Linear methods can be completely blind to strong non-linear mediation relationships**.

## Key Findings

1. **Equivalence with Linear Data**: Traditional mediation analysis, Frisch-Waugh-Lovell (FWL), and Double Machine Learning (DML) are mathematically equivalent when relationships are linear.

2. **Linear Blindness**: When relationships are curvilinear/symmetric (e.g., X→M=X², M→Y=(M-c)²), linear methods detect ~0% mediation while ML correctly identifies ~95% mediation.

3. **ML Calibration**: ML methods correctly scale their estimates based on true mediation strength, detecting high mediation when present and low mediation when absent.

## Repository Structure

```
├── demonstrations/           # Core demonstration scripts
│   ├── test_three_way_equivalence.py    # Proves equivalence with linear data
│   ├── test_perfect_symmetry.py         # Shows complete linear blindness
│   └── test_linear_blindness.py         # Detailed linear blindness analysis
├── visualizations/          # Generated plots
├── theory/                  # Theoretical background
│   └── poma_formula.md     # PoMA formula with corrections
└── README.md
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/poma-mediation-ml-demo.git
cd poma-mediation-ml-demo

# Install requirements
pip install -r requirements.txt

# Run demonstrations
python demonstrations/test_three_way_equivalence.py
python demonstrations/test_perfect_symmetry.py
```

## Core Concepts

### PoMA Formula
The Percentage of Mediated Accuracy (PoMA) measures how much of X's effect on Y flows through mediator M:

```
PoMA = 1 - β_direct/β_total
```

With correction terms for finite sample bias:
```
PoMA = 1 - [(1 - Cov(Ŷ,X̂)/Cov(Y,X) - C1 - C2) / (1 - Var(X̂)/Var(X) - C3)]
```

### Why This Matters

In many real-world scenarios, relationships are non-linear:
- Dose-response curves (U-shaped)
- Threshold effects
- Saturation effects
- Interaction effects

Traditional linear mediation analysis **completely misses** these relationships, while ML/DML methods detect them accurately.

## Key Results

### 1. Perfect Symmetry with Curvilinear Relationships
- **X→M**: M = X² (U-shaped)
- **M→Y**: Y = (M-4.5)² (U-shaped, curvilinear)
- **Truth**: ~100% mediation
- **Linear Detection**: 0.1% (R²≈0, correlations ≈ 0)
- **ML Detection**: 99.5% (R²≈1)

### 2. Linear Data Equivalence
- Traditional Mediation ≡ FWL ≡ DML
- Maximum difference: 1.44e-15 (machine precision)

### 3. ML Calibration
- Strong non-linear: ML detects ~99% (Truth: ~99%)
- Weak non-linear: ML detects ~10% (Truth: ~10%)
- Linear fails in both cases

## Citation

If you use this code or findings, please cite:
```
@software{poma_ml_demo,
  title = {PoMA Mediation Analysis: ML vs Linear Methods},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/poma-mediation-ml-demo}
}
```

## License

MIT License - See LICENSE file for details.