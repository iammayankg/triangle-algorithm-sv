# Regime battery summary

| dataset | cell | solver | time (s) | dist | accuracy | oracle | non-converged |
|--|--|--|--:|--:|--:|--:|--:|
| a9a | FEAS | TA-I | 1.1 ± 0.2 | 0.00099 | - | - | 0/5 |
| a9a | KHM | K-ETA | 600.1 ± 0.0 | 0.00009 | 0.6900 | 5,881 | 5/5 |
| a9a | KHM | LIBSVM | 26.4 ± 2.9 | 0.00327 | 0.7518 | - | 0/5 |
| a9a | KL2 | K-ETA | 52.8 ± 71.0 | 0.03019 | 0.7949 | 7,621 | 0/5 |
| a9a | KL2 | K-SMO | 57.7 ± 23.2 | 0.03019 | 0.7949 | 43,623 | 0/5 |
| a9a | KL2 | SVC-L1-ref | 8.6 ± 2.9 | - | 0.7889 | - | 0/5 |
| covtype | FEAS | TA-I | 1.1 ± 0.6 | 0.00098 | - | - | 0/5 |
| covtype | KHM | K-ETA | 600.2 ± 0.1 | 0.00053 | 0.7044 | 7,207 | 5/5 |
| covtype | KHM | LIBSVM | 144.4 ± 6.7 | 0.00025 | 0.7119 | - | 5/5 |
| covtype | KL2 | K-ETA | 26.0 ± 4.2 | 0.02609 | 0.7919 | 9,084 | 0/5 |
| covtype | KL2 | K-SMO | 17.6 ± 0.5 | 0.02609 | 0.7919 | 56,170 | 0/5 |
| covtype | KL2 | SVC-L1-ref | 3.4 ± 0.3 | - | 0.7831 | - | 0/5 |
| gisette | FEAS | TA-I | 15.4 ± 0.5 | 5.07253 | - | - | 0/5 |
| gisette | KHM | K-ETA | 86.2 ± 14.8 | 0.05215 | 0.9790 | 2,595 | 0/5 |
| gisette | KHM | LIBSVM | 104.6 ± 20.2 | 0.05215 | 0.9790 | - | 0/5 |
| gisette | KL2 | K-ETA | 131.3 ± 33.6 | 0.06882 | 0.9710 | 3,648 | 0/5 |
| gisette | KL2 | K-SMO | 419.6 ± 48.5 | 0.06882 | 0.9710 | 15,200 | 0/5 |
| gisette | KL2 | SVC-L1-ref | 83.5 ± 13.1 | - | 0.9770 | - | 0/5 |
| gisette | LIN | ETA | 56.6 ± 5.9 | 3.66381 | 0.9810 | 1,498 | 0/5 |
| gisette | LIN | LIBSVM | 51.4 ± 13.1 | 3.66396 | 0.9810 | - | 0/5 |
| gisette | LIN | SMO | 405.1 ± 61.9 | 3.66402 | 0.9810 | 11,079 | 0/5 |
| ijcnn1 | FEAS | TA-I | 0.2 ± 0.0 | 0.00095 | - | - | 0/5 |
| ijcnn1 | KHM | K-ETA | 420.8 ± 148.8 | 0.00430 | 0.9458 | 2,476 | 1/5 |
| ijcnn1 | KHM | LIBSVM | 1.3 ± 0.3 | 0.00430 | 0.9458 | - | 0/5 |
| ijcnn1 | KL2 | K-ETA | 9.2 ± 2.5 | 0.04790 | 0.9589 | 4,594 | 0/5 |
| ijcnn1 | KL2 | K-SMO | 6.1 ± 0.7 | 0.04790 | 0.9588 | 26,143 | 0/5 |
| ijcnn1 | KL2 | SVC-L1-ref | 1.1 ± 0.0 | - | 0.9502 | - | 0/5 |
| w8a | FEAS | TA-I | 42.1 ± 4.7 | 0.00100 | - | - | 0/5 |
| w8a | KHM | K-ETA | 101.2 ± 159.0 | 0.00000 | - | 1,931 | 0/5 |
| w8a | KL2 | K-ETA | 11.5 ± 2.6 | 0.05682 | 0.9745 | 3,590 | 0/5 |
| w8a | KL2 | K-SMO | 41.7 ± 10.0 | 0.05682 | 0.9745 | 20,245 | 0/5 |
| w8a | KL2 | SVC-L1-ref | 4.9 ± 1.2 | - | 0.9746 | - | 0/5 |

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
- ijcnn1: converged=4, timeout=1
- w8a: intersect=5
