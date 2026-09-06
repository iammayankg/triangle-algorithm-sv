# Regime battery summary

| dataset | cell | solver | time (s) | dist | accuracy | oracle | non-converged |
|--|--|--|--:|--:|--:|--:|--:|
| a9a | FEAS | TA-I | 0.9 ± 0.0 | 0.00099 | - | - | 0/5 |
| a9a | KHM | K-ETA | 600.1 ± 0.0 | 0.00006 | 0.7066 | 5,947 | 5/5 |
| a9a | KHM | LIBSVM | 19.8 ± 1.3 | 0.00327 | 0.7518 | - | 0/5 |
| a9a | KL2 | K-ETA | 15.3 ± 2.8 | 0.03019 | 0.7949 | 7,622 | 0/5 |
| a9a | KL2 | K-SMO | 27.0 ± 4.7 | 0.03019 | 0.7949 | 43,623 | 0/5 |
| a9a | KL2 | SVC-L1-ref | 4.4 ± 0.6 | - | 0.7889 | - | 0/5 |
| covtype | FEAS | TA-I | 0.8 ± 0.3 | 0.00098 | - | - | 0/5 |
| covtype | KHM | K-ETA | 600.1 ± 0.0 | 0.00055 | 0.6936 | 7,476 | 5/5 |
| covtype | KHM | LIBSVM | 134.0 ± 3.1 | 0.00025 | 0.7119 | - | 5/5 |
| covtype | KL2 | K-ETA | 21.4 ± 1.8 | 0.02609 | 0.7919 | 9,086 | 0/5 |
| covtype | KL2 | K-SMO | 15.4 ± 0.6 | 0.02609 | 0.7919 | 56,170 | 0/5 |
| covtype | KL2 | SVC-L1-ref | 3.1 ± 0.1 | - | 0.7831 | - | 0/5 |
| gisette | FEAS | TA-I | 8.2 ± 0.3 | 5.07253 | - | - | 0/5 |
| gisette | KHM | K-ETA | 57.8 ± 8.7 | 0.05215 | 0.9790 | 2,604 | 0/5 |
| gisette | KHM | LIBSVM | 68.1 ± 13.4 | 0.05215 | 0.9790 | - | 0/5 |
| gisette | KL2 | K-ETA | 84.8 ± 15.4 | 0.06882 | 0.9710 | 3,653 | 0/5 |
| gisette | KL2 | K-SMO | 242.8 ± 15.0 | 0.06882 | 0.9710 | 15,200 | 0/5 |
| gisette | KL2 | SVC-L1-ref | 63.3 ± 13.8 | - | 0.9770 | - | 0/5 |
| gisette | LIN | ETA | 31.6 ± 2.8 | 3.66381 | 0.9810 | 1,468 | 0/5 |
| gisette | LIN | LIBSVM | 36.5 ± 5.9 | 3.66396 | 0.9810 | - | 0/5 |
| gisette | LIN | SMO | 258.8 ± 30.5 | 3.66402 | 0.9810 | 11,079 | 0/5 |
| ijcnn1 | FEAS | TA-I | 0.1 ± 0.0 | 0.00095 | - | - | 0/5 |
| ijcnn1 | KHM | K-ETA | 350.7 ± 109.4 | 0.00430 | 0.9459 | 2,605 | 0/5 |
| ijcnn1 | KHM | LIBSVM | 1.1 ± 0.2 | 0.00430 | 0.9458 | - | 0/5 |
| ijcnn1 | KL2 | K-ETA | 7.7 ± 1.9 | 0.04790 | 0.9588 | 4,603 | 0/5 |
| ijcnn1 | KL2 | K-SMO | 5.3 ± 0.4 | 0.04790 | 0.9588 | 26,143 | 0/5 |
| ijcnn1 | KL2 | SVC-L1-ref | 1.0 ± 0.1 | - | 0.9502 | - | 0/5 |
| w8a | FEAS | TA-I | 32.4 ± 4.8 | 0.00100 | - | - | 0/5 |
| w8a | KHM | K-ETA | 11.7 ± 32.4 | 0.00000 | - | 522 | 0/5 |
| w8a | KL2 | K-ETA | 8.7 ± 1.6 | 0.05682 | 0.9745 | 3,578 | 0/5 |
| w8a | KL2 | K-SMO | 28.5 ± 1.3 | 0.05682 | 0.9745 | 20,245 | 0/5 |
| w8a | KL2 | SVC-L1-ref | 2.7 ± 0.7 | - | 0.9746 | - | 0/5 |

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
