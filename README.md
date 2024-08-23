# LaHiCaSl

**Generalized Independent Noise Condition for Estimating Causal Structure with Latent Variables, JMLR**

This is the Implementation of LaHiCaSl algorithm, Version 0.1.

## Overview

This project estimates the causal structure among the latent variables in the Linear Non-Gaussian Latent Hierarchical Models, using the GIN condition.

### Main function: LaHiCaSl.py

```
Latent Hierarchical Causal Structure Learning (LaHiCaSL)

Input:
    Parameters:
    data : set of observed variables
    alpha: Threshold
Output:
    Causal_Matrix : Causal structure matrix over both observed and latent variables
```

### Test

Use TestDemo.py to apply the LaHiCaSl algorithm.

### Note

```
Our method relies heavily on independence tests. 
Here, HSIC-based Independence Test is used.

Reference: Q. Zhang, S. Filippi, A. Gretton, and D. Sejdinovic, Large-Scale Kernel Methods for Independence Testing, Statistics and Computing, 2018.

Moreover, the CCA-Rank test  also be used.

Reference: Huang B, Low C J H, Xie F, et al. Latent hierarchical causal structure discovery with rank constraints[J]. Advances in neural information processing systems, 2022, 35: 5549-5561.
```
### Citation
If you use this code, please cite the following paper:

Xie F*, Huang B*, Chen, Z., Cai, R., Glymour, C., Geng, Z., and Zhang, K. Generalized independent noise condition for estimating causal structure with latent variables[J]. Journal of Machine Learning Research, 2024, 25: 1-61.

