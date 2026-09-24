# ESM-2 attention vs. the k-mer baselines

Evaluation subsample of the test split: **9696 fragments** (1696 plasmid + 8000 chromosome). Every metric is weighted back to the full split (72,332 fragments, 1,696 plasmid = 2.345%), so the numbers estimate performance on the whole test split.

All models below are scored on exactly these fragments with the same threshold protocol: the decision threshold is chosen on the weighted validation subsample and then applied to the test subsample.

## 1. Head-to-head at the default threshold 0.5

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| always-chromosome | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0234 |
| PlasFlow (vote of 3) - balanced | 0.8081 | 0.7898 | 0.0883 | 0.7706 | 0.8090 | 0.1585 | 0.2174 | 0.8707 | 0.1590 |
| PlasFlow (vote of 3) - full | 0.9766 | 0.5026 | 0.5048 | 0.0053 | 0.9999 | 0.0105 | 0.0499 | 0.8895 | 0.2140 |
| Zou et al. (mean of 3) - balanced | 0.8157 | 0.8086 | 0.0947 | 0.8013 | 0.8160 | 0.1693 | 0.2342 | 0.8829 | 0.1979 |
| Zou et al. (mean of 3) - full | 0.9766 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9092 | 0.2811 |
| PlasClass (logistic regression) - balanced | 0.7056 | 0.6985 | 0.0534 | 0.6910 | 0.7060 | 0.0992 | 0.1307 | 0.7569 | 0.0652 |
| **ESM-2 + attention (mean of 3, final epoch)** | 0.8074 | 0.7949 | 0.0891 | 0.7818 | 0.8080 | 0.1599 | 0.2208 | 0.8673 | 0.1464 |
| ESM-2 + attention (mean of 3, best epoch) | 0.7781 | 0.7958 | 0.0807 | 0.8143 | 0.7772 | 0.1468 | 0.2106 | 0.8648 | 0.1504 |

## 2. Head-to-head at the validation-tuned threshold

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| PlasFlow (vote of 3) - balanced @ 0.958 | 0.9409 | 0.6878 | 0.1786 | 0.4222 | 0.9534 | 0.2510 | 0.2484 | 0.8707 | 0.1590 |
| PlasFlow (vote of 3) - full @ 0.094 | 0.9481 | 0.7006 | 0.2104 | 0.4410 | 0.9603 | 0.2849 | 0.2809 | 0.8895 | 0.2140 |
| Zou et al. (mean of 3) - balanced @ 0.970 | 0.9139 | 0.7510 | 0.1513 | 0.5802 | 0.9219 | 0.2400 | 0.2656 | 0.8829 | 0.1979 |
| Zou et al. (mean of 3) - full @ 0.172 | 0.9692 | 0.6669 | 0.3459 | 0.3496 | 0.9841 | 0.3478 | 0.3320 | 0.9092 | 0.2811 |
| PlasClass (logistic regression) - balanced @ 1.000 | 0.8953 | 0.6144 | 0.0779 | 0.3196 | 0.9091 | 0.1252 | 0.1173 | 0.7569 | 0.0652 |
| **ESM-2 + attention (final) @ 0.834** | 0.9495 | 0.6470 | 0.1818 | 0.3296 | 0.9644 | 0.2343 | 0.2205 | 0.8673 | 0.1464 |
| ESM-2 + attention (best epoch) @ 0.893 | 0.9563 | 0.6220 | 0.1929 | 0.2712 | 0.9728 | 0.2254 | 0.2068 | 0.8648 | 0.1504 |

## 3. ESM-2 run-to-run spread (individual seeds, weighted, threshold 0.5)

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| seed 42 (best epoch 9) | 0.8075 | 0.7956 | 0.0892 | 0.7830 | 0.8081 | 0.1602 | 0.2213 | 0.8686 | 0.1426 |
| seed 142 (best epoch 15) | 0.8038 | 0.7799 | 0.0850 | 0.7547 | 0.8050 | 0.1528 | 0.2086 | 0.8498 | 0.1273 |
| seed 242 (best epoch 12) | 0.7425 | 0.7651 | 0.0682 | 0.7889 | 0.7414 | 0.1256 | 0.1805 | 0.8398 | 0.1268 |
| **ensemble (mean probability)** | 0.8074 | 0.7949 | 0.0891 | 0.7818 | 0.8080 | 0.1599 | 0.2208 | 0.8673 | 0.1464 |

## 4. Replicon-level (mean probability over a replicon's sampled fragments)

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| PlasFlow (vote of 3) - balanced | 0.8571 | 0.8497 | 0.8163 | 0.8163 | 0.8831 | 0.8163 | 0.6994 | 0.9189 | 0.8822 |
| PlasFlow (vote of 3) - full | 0.6111 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9035 | 0.8708 |
| Zou et al. (mean of 3) - balanced | 0.8730 | 0.8664 | 0.8367 | 0.8367 | 0.8961 | 0.8367 | 0.7328 | 0.9194 | 0.9022 |
| Zou et al. (mean of 3) - full | 0.6111 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8749 | 0.8565 |
| PlasClass (logistic regression) - balanced | 0.7778 | 0.7737 | 0.6981 | 0.7551 | 0.7922 | 0.7255 | 0.5405 | 0.8622 | 0.8227 |
| **ESM-2 + attention (final)** | 0.8651 | 0.8562 | 0.8333 | 0.8163 | 0.8961 | 0.8247 | 0.7152 | 0.9218 | 0.9260 |

## 5. Unweighted view of the sampled fragments (not prevalence-corrected)

Useful only for checking the weighting; the plasmid share of the subsample is 17.5%, not the real 2.345%.

| model | accuracy | balanced_accuracy | precision | recall | specificity | f1 | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| ESM-2 + attention (final) | 0.8034 | 0.7949 | 0.4633 | 0.7818 | 0.8080 | 0.5818 | 0.4913 | 0.8673 | 0.5700 |

## 6. Evaluation subsample and ORF statistics

| split | sampled | plasmid | effective size | ORFs per fragment |
|---|---|---|---|---|
| train | 8000 | 4000 | 337,512 | 9.25 |
| validation | 5000 | 1000 | 72,317 | 9.21 |
| test | 9696 | 1696 | 72,332 | 9.41 |

## 7. Cost

| item | value |
|---|---|
| ESM-2 embedding (all three splits) | see results/extract_features.log |
| attention-head training, median seconds per seed | 241 s |
| median seconds per training epoch | 15 s |
| seeds trained | 3 |
