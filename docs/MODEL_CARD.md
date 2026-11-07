# Model card — FD001 LSTM champion

Run `fd001-c9c5dec1-seed42` selected a one-layer, 48-hidden-unit LSTM using the lowest RMSE across 165 causal pseudo-test endpoints from 15 held-out engines. Seventy engines trained models, 15 tuned candidates, 15 calibrated probabilities/intervals, and the 100 official test engines were used once for reporting.

| Metric | Capped RUL | Uncapped RUL |
| --- | ---: | ---: |
| RMSE | 15.139 | 16.021 |
| MAE | 10.755 | 11.825 |
| R² | 0.857 | 0.851 |
| NASA asymmetric score | 578.708 | 602.126 |

The 30-cycle calibrated risk model achieved Brier 0.0537, AUROC 0.9763, and ECE 0.0326. A nominal 90% split-conformal interval achieved 85% empirical official-test coverage with mean width 34.68 cycles; this undercoverage must be shown to users and monitored. Explanations are held-out permutation importance for the strong boosted-tree challenger, not LSTM-specific causal explanations.

Intended use: research, portfolio demonstration, and non-safety-critical maintenance workflow prototyping. Do not use for flight operations or real equipment without representative data, domain validation, cost-sensitive thresholds, monitoring, and human approval.
