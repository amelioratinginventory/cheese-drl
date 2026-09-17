# Deep Reinforcement Learning for Aging Cheese Inventory Management Under Correlated Prices

This repository contains the code, problem instances, and detailed results accompanying the paper:

> Pahr, A., Krumm, P., Kolemesina, A., and Grunow, M. *Deep Reinforcement Learning for Aging Cheese Inventory Management Under Correlated Prices.* Working Paper submitted to Elsevier.

It provides everything needed to reproduce the computational results reported in the paper, in line with the EJOR policy on computational experiments: the data sources and problem instances, the individual results for each instance, and the code used to generate them.

## Repository structure

| Path | Description |
| --- | --- |
| `mainCheese_singleAgent.py` | Entry point for training the single-agent actor-critic architectures. |
| `mainCheese_multiAgent.py` | Entry point for training the multi-agent actor architectures. |
| `CheeseEnvironment.py` | MDP simulation environment for the aging cheese inventory problem (single-agent). |
| `MultiAgentCheeseEnvironment.py` | MDP simulation environment for the multi-agent setting. |
| `numerical_analysis_proposition1.py` | Self-contained numerical verification of Proposition 1 (value of issuance flexibility vs. mean reversion) across a wide range of parameter settings. |
| `AR(1)_cheese_analysis.ipynb` | Empirical validation of the AR(1) purchase-price model on real milk-price data (coefficient significance, R^2, augmented Dickey-Fuller test). |
| `create_regression_data.ipynb` | Generation of the common random number simulations of the trained policies for evaluation. |
| `evaluate_methodology.ipynb` | Evaluation of the trained architectures and reproduction of the reported performance comparisons. |
| `analysis_cheese.ipynb` | Post-processing and generation of the figures and tables reported in the paper. |
| `instances/` | Problem instances used in the numerical study, with the individual results per instance. |
| `figures/` | Figures generated from the results. |

## Requirements

The code is written in Python 3.10. The core dependencies are:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `statsmodels` (for the AR(1) analysis)
- `jupyter` (to run the notebooks)


## Reproducing the results

### Verifying Proposition 1

The analytical result is verified numerically over a large parameter grid:

```bash
python numerical_analysis_proposition1.py --full
```

This evaluates the value of issuance flexibility across [1,455] parameter settings and nine mean-reversion levels and reports whether the monotonicity of Proposition 1 holds in every setting. [State the expected runtime and any resolution flags.]

### Empirical validation of the price model

Run `AR(1)_cheese_analysis.ipynb` to reproduce the validation of the AR(1) purchase-price model on the milk-price data, including the coefficient significance, the reported R^2 values, and the stationarity tests.

### Training the DRL architectures

The single-agent and multi-agent architectures are trained via:

```bash
python mainCheese_singleAgent.py 
python mainCheese_multiAgent.py 
```

### Evaluation and figures

After training, `create_regression_data.ipynb`, `evaluate_methodology.ipynb` and `analysis_cheese.ipynb` reproduce the performance comparisons, figures, and tables reported in the paper.


## Instances and detailed results

The `instances/` directory contains the problem instances together with their config.json file used in the numerical study, together with the individual results for each instance.

## Citation

If you use this code, please cite:

```bibtex
@article{Pahr2026cheese,
  title   = {Deep Reinforcement Learning for Aging Cheese Inventory Management Under Correlated Prices},
  author  = {Pahr, Alexander and Krumm, Paula and Kolemesina, Anna and Grunow, Martin},
  journal = {SSRN Electronic Journal},
  year    = {2026},
  doi    = {10.2139/ssrn.5538320}
}
```

