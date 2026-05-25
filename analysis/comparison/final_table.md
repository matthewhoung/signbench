## Cross-method comparison

GTSRB test-split metrics for all three methods, plus the Taiwan domain-transfer probe (universal-class signs correct / total; mean softmax confidence on out-of-distribution Taiwan-unique signs).

| Method | GTSRB test accuracy | Precision | Recall | F1 | Taiwan universal correct | Taiwan OOD mean confidence |
| --- | --- | --- | --- | --- | --- | --- |
| rf_hog | 0.9032 | 0.9086 | 0.9032 | 0.8995 | 2/4 | 0.133 |
| plain_cnn | 0.9678 | 0.9691 | 0.9678 | 0.9675 | 3/4 | 0.932 |
| stn_cnn | 0.9899 | 0.9904 | 0.9899 | 0.9900 | 3/4 | 0.594 |
