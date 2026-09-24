# Baseline results

Test split: 72332 fragments, 1696 plasmid (2.345%), 70636 chromosome.

## 1. Fragment-level test performance (ensemble, threshold 0.5)

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| always-chromosome | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0234 |
| PlasFlow (vote of 3) - regime A: balanced | 0.8105 | 0.7910 | 0.0894 | 0.7706 | 0.8115 | 0.1602 | 0.2193 | 0.8710 | 0.1565 |
| PlasFlow (vote of 3) - regime B: full/imbalanced | 0.9766 | 0.5026 | 0.7500 | 0.0053 | 1.0000 | 0.0105 | 0.0618 | 0.8881 | 0.2110 |
| Zou et al. (mean of 3) - regime A: balanced | 0.8164 | 0.8090 | 0.0950 | 0.8013 | 0.8167 | 0.1699 | 0.2348 | 0.8830 | 0.1889 |
| Zou et al. (mean of 3) - regime B: full/imbalanced | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9092 | 0.2733 |
| PlasClass (logistic regression) - regime A: balanced | 0.7075 | 0.6995 | 0.0537 | 0.6910 | 0.7079 | 0.0997 | 0.1315 | 0.7594 | 0.0672 |

## 2. Balanced test subsample (3392 fragments, 1:1)

Published accuracies (PlasFlow ~96%, Zou et al. 89-92%) were measured on balanced data; this table is the comparable view of our runs.

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| PlasFlow (vote of 3) - regime A: balanced | 0.7883 | 0.7883 | 0.7989 | 0.7706 | 0.8060 | 0.7845 | 0.5770 | 0.8741 | 0.8609 |
| PlasFlow (vote of 3) - regime B: full/imbalanced | 0.5027 | 0.5027 | 1.0000 | 0.0053 | 1.0000 | 0.0106 | 0.0516 | 0.8908 | 0.8849 |
| Zou et al. (mean of 3) - regime A: balanced | 0.8072 | 0.8072 | 0.8109 | 0.8013 | 0.8131 | 0.8060 | 0.6144 | 0.8874 | 0.8812 |
| Zou et al. (mean of 3) - regime B: full/imbalanced | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9136 | 0.9130 |
| PlasClass (logistic regression) - regime A: balanced | 0.6940 | 0.6940 | 0.6951 | 0.6910 | 0.6969 | 0.6931 | 0.3880 | 0.7543 | 0.7282 |

## 3. Confusion matrices on the test split

| model | TN | FP | FN | TP | plasmid recall | plasmid precision |
|---|---|---|---|---|---|---|
| PlasFlow (vote of 3) - A: balanced | 57318 | 13318 | 389 | 1307 | 0.7706 | 0.0894 |
| PlasFlow (vote of 3) - B: full/imbalanced | 70633 | 3 | 1687 | 9 | 0.0053 | 0.7500 |
| Zou et al. (mean of 3) - A: balanced | 57691 | 12945 | 337 | 1359 | 0.8013 | 0.0950 |
| Zou et al. (mean of 3) - B: full/imbalanced | 70636 | 0 | 1696 | 0 | 0.0000 | 0.0000 |
| PlasClass (logistic regression) - A: balanced | 50002 | 20634 | 524 | 1172 | 0.6910 | 0.0537 |

## 4. Individual members of each ensemble (test split)

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| plasflow A: balanced - k5 | 0.8180 | 0.7782 | 0.0894 | 0.7364 | 0.8199 | 0.1595 | 0.2133 | 0.8565 | 0.1236 |
| plasflow A: balanced - k6 | 0.7939 | 0.7771 | 0.0816 | 0.7594 | 0.7947 | 0.1473 | 0.2030 | 0.8502 | 0.1334 |
| plasflow A: balanced - k7 | 0.7327 | 0.7319 | 0.0616 | 0.7311 | 0.7327 | 0.1137 | 0.1566 | 0.8079 | 0.1058 |
| plasflow B: full/imbalanced - k5 | 0.9765 | 0.5046 | 0.4211 | 0.0094 | 0.9997 | 0.0185 | 0.0602 | 0.8808 | 0.1911 |
| plasflow B: full/imbalanced - k6 | 0.9765 | 0.5023 | 0.4211 | 0.0047 | 0.9998 | 0.0093 | 0.0426 | 0.8779 | 0.2098 |
| plasflow B: full/imbalanced - k7 | 0.9766 | 0.5084 | 0.5370 | 0.0171 | 0.9996 | 0.0331 | 0.0928 | 0.8661 | 0.1737 |
| zou A: balanced - model0 | 0.8013 | 0.8001 | 0.0880 | 0.7989 | 0.8013 | 0.1586 | 0.2219 | 0.8763 | 0.1720 |
| zou A: balanced - model1 | 0.8118 | 0.7998 | 0.0915 | 0.7871 | 0.8124 | 0.1640 | 0.2261 | 0.8809 | 0.1703 |
| zou A: balanced - model2 | 0.8239 | 0.7956 | 0.0952 | 0.7659 | 0.8253 | 0.1694 | 0.2287 | 0.8742 | 0.1730 |
| zou B: full/imbalanced - model0 | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8961 | 0.2194 |
| zou B: full/imbalanced - model1 | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9024 | 0.2565 |
| zou B: full/imbalanced - model2 | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9121 | 0.2898 |
| plasclass A: balanced - logreg | 0.7075 | 0.6995 | 0.0537 | 0.6910 | 0.7079 | 0.0997 | 0.1315 | 0.7594 | 0.0672 |

## 5. Decision-threshold variants (test split)

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| plasflow A: balanced - threshold 0.5 | 0.8105 | 0.7910 | 0.0894 | 0.7706 | 0.8115 | 0.1602 | 0.2193 | 0.8710 | 0.1565 |
| plasflow A: balanced - threshold 0.966 (best F1 on validation) | 0.9433 | 0.6798 | 0.1812 | 0.4033 | 0.9563 | 0.2501 | 0.2447 | 0.8710 | 0.1565 |
| plasflow A: balanced - majority vote of members | 0.8084 | 0.7897 | 0.0884 | 0.7700 | 0.8093 | 0.1586 | 0.2175 | 0.7897 | 0.0735 |
| plasflow B: full/imbalanced - threshold 0.5 | 0.9766 | 0.5026 | 0.7500 | 0.0053 | 1.0000 | 0.0105 | 0.0618 | 0.8881 | 0.2110 |
| plasflow B: full/imbalanced - threshold 0.092 (best F1 on validation) | 0.9456 | 0.7028 | 0.2020 | 0.4481 | 0.9575 | 0.2785 | 0.2764 | 0.8881 | 0.2110 |
| plasflow B: full/imbalanced - majority vote of members | 0.9766 | 0.5032 | 0.5789 | 0.0065 | 0.9999 | 0.0128 | 0.0595 | 0.5032 | 0.0271 |
| zou A: balanced - threshold 0.5 | 0.8164 | 0.8090 | 0.0950 | 0.8013 | 0.8167 | 0.1699 | 0.2348 | 0.8830 | 0.1889 |
| zou A: balanced - threshold 0.991 (best F1 on validation) | 0.9336 | 0.7237 | 0.1773 | 0.5035 | 0.9439 | 0.2622 | 0.2716 | 0.8830 | 0.1889 |
| zou A: balanced - three-vote rule | 0.8161 | 0.8063 | 0.0944 | 0.7960 | 0.8166 | 0.1687 | 0.2327 | 0.8063 | 0.0799 |
| zou B: full/imbalanced - threshold 0.5 | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9092 | 0.2733 |
| zou B: full/imbalanced - threshold 0.161 (best F1 on validation) | 0.9672 | 0.6791 | 0.3272 | 0.3768 | 0.9814 | 0.3502 | 0.3344 | 0.9092 | 0.2733 |
| zou B: full/imbalanced - three-vote rule | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0234 |
| plasclass A: balanced - threshold 0.5 | 0.7075 | 0.6995 | 0.0537 | 0.6910 | 0.7079 | 0.0997 | 0.1315 | 0.7594 | 0.0672 |
| plasclass A: balanced - threshold 1.000 (best F1 on validation) | 0.8881 | 0.6239 | 0.0776 | 0.3467 | 0.9011 | 0.1269 | 0.1225 | 0.7594 | 0.0672 |

## 6. PlasFlow probability-threshold rule (0.7)

| model | classified | unclassified % | unclassified % (plasmid) | accuracy on classified | F1 on classified | recall on classified |
|---|---|---|---|---|---|---|
| A: balanced | 56518 | 21.86 | 26.65 | 0.8663 | 0.2177 | 0.8449 |
| B: full/imbalanced | 72068 | 0.36 | 6.49 | 0.9780 | 0.0000 | 0.0000 |

## 7. Replicon-level performance (mean fragment probability per replicon)

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| PlasFlow (vote of 3) - A: balanced | 0.8651 | 0.8562 | 0.8333 | 0.8163 | 0.8961 | 0.8247 | 0.7152 | 0.9247 | 0.9043 |
| PlasFlow (vote of 3) - B: full/imbalanced | 0.6111 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9009 | 0.8699 |
| Zou et al. (mean of 3) - A: balanced | 0.8889 | 0.8794 | 0.8723 | 0.8367 | 0.9221 | 0.8542 | 0.7649 | 0.9231 | 0.9106 |
| Zou et al. (mean of 3) - B: full/imbalanced | 0.6111 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8728 | 0.8520 |
| PlasClass (logistic regression) - A: balanced | 0.8016 | 0.7931 | 0.7400 | 0.7551 | 0.8312 | 0.7475 | 0.5842 | 0.8712 | 0.8391 |

## 8. Training cost

| run | train fragments | members | total training seconds |
|---|---|---|---|
| PlasFlow (vote of 3) - A: balanced | 15830 | 3 | 556 |
| PlasFlow (vote of 3) - B: full/imbalanced | 337512 | 3 | 1094 |
| Zou et al. (mean of 3) - A: balanced | 15830 | 3 | 705 |
| Zou et al. (mean of 3) - B: full/imbalanced | 337512 | 3 | 908 |
| PlasClass (logistic regression) - A: balanced | 15830 | 1 | 58 |
