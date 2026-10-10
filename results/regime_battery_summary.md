# Regime battery summary

| dataset | cell | solver | time (s) | dist | accuracy | oracle | non-converged |
|--|--|--|--:|--:|--:|--:|--:|
| a9a | FEAS | TA-I | 0.7 ± 0.0 | 0.00099 | - | - | 0/5 |
| a9a | KHM | K-ETA | 600.1 ± 0.0 | 0.00002 | 0.6854 | 6,003 | 5/5 |
| a9a | KHM | LIBSVM | 18.5 ± 2.0 | 0.00327 | 0.7518 | - | 0/5 |
| a9a | KL2 | K-ETA | 13.2 ± 1.0 | 0.03019 | 0.7949 | 7,622 | 0/5 |
| a9a | KL2 | K-SMO | 8.9 ± 1.5 | 0.03019 | 0.7949 | 7,615 | 0/5 |
| a9a | KL2 | SVC-L1-ref | 4.5 ± 0.4 | - | 0.7889 | - | 0/5 |
| covtype | FEAS | TA-I | 0.6 ± 0.3 | 0.00098 | - | - | 0/5 |
| covtype | KHM | K-ETA | 600.1 ± 0.0 | 0.00044 | 0.7110 | 7,570 | 5/5 |
| covtype | KHM | LIBSVM | 122.2 ± 33.2 | 0.00025 | 0.7119 | - | 5/5 |
| covtype | KL2 | K-ETA | 16.8 ± 4.3 | 0.02609 | 0.7919 | 9,086 | 0/5 |
| covtype | KL2 | K-SMO | 5.8 ± 1.2 | 0.02609 | 0.7919 | 9,080 | 0/5 |
| covtype | KL2 | SVC-L1-ref | 3.1 ± 0.3 | - | 0.7831 | - | 0/5 |
| gisette | FEAS | TA-I | 10.3 ± 1.1 | 5.07253 | - | - | 0/5 |
| gisette | KHM | K-ETA | 64.6 ± 6.2 | 0.05215 | 0.9790 | 2,604 | 0/5 |
| gisette | KHM | LIBSVM | 75.6 ± 3.1 | 0.05215 | 0.9790 | - | 0/5 |
| gisette | KL2 | K-ETA | 89.8 ± 7.8 | 0.06882 | 0.9710 | 3,653 | 0/5 |
| gisette | KL2 | K-SMO | 90.6 ± 8.5 | 0.06882 | 0.9710 | 3,649 | 0/5 |
| gisette | KL2 | SVC-L1-ref | 75.1 ± 4.8 | - | 0.9770 | - | 0/5 |
| gisette | LIN | ETA | 38.2 ± 3.2 | 3.66381 | 0.9810 | 1,468 | 0/5 |
| gisette | LIN | LIBSVM | 38.5 ± 3.1 | 3.66396 | 0.9810 | - | 0/5 |
| gisette | LIN | SMO | 34.0 ± 3.2 | 3.66402 | 0.9810 | 1,353 | 0/5 |
| ijcnn1 | FEAS | TA-I | 0.1 ± 0.0 | 0.00095 | - | - | 0/5 |
| ijcnn1 | KHM | K-ETA | 242.3 ± 69.1 | 0.00430 | 0.9459 | 2,588 | 0/5 |
| ijcnn1 | KHM | LIBSVM | 1.1 ± 0.2 | 0.00430 | 0.9458 | - | 0/5 |
| ijcnn1 | KL2 | K-ETA | 5.3 ± 0.3 | 0.04790 | 0.9588 | 4,603 | 0/5 |
| ijcnn1 | KL2 | K-SMO | 2.6 ± 0.3 | 0.04790 | 0.9588 | 4,525 | 0/5 |
| ijcnn1 | KL2 | SVC-L1-ref | 0.9 ± 0.1 | - | 0.9502 | - | 0/5 |
| w8a | FEAS | TA-I | 28.9 ± 1.1 | 0.00100 | - | - | 0/5 |
| w8a | KHM | K-ETA | 7.2 ± 20.1 | 0.00000 | - | 521 | 0/5 |
| w8a | KL2 | K-ETA | 6.3 ± 1.0 | 0.05682 | 0.9745 | 3,578 | 0/5 |
| w8a | KL2 | K-SMO | 5.2 ± 1.5 | 0.05682 | 0.9745 | 3,573 | 0/5 |
| w8a | KL2 | SVC-L1-ref | 2.2 ± 0.7 | - | 0.9746 | - | 0/5 |

## Linear hard-margin feasibility (TA I)

- a9a: intersect=5
- covtype: intersect=5
- gisette: separated=5
- ijcnn1: intersect=5
- w8a: intersect=5

## RBF feature-space hard-margin feasibility (K-ETA; intersect = conflicting duplicates)

- a9a: timeout=5
- covtype: timeout=5
- gisette: converged=5
- ijcnn1: converged=5
- w8a: intersect=5
