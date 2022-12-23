# Generalized Integral Transform Neural Network (GIT-Net) for Operator Learning

This work is to propose a novel neural network for operator learning, which is generalized integral transform (GIT-Net). It is demonstrated by solving several PDEs and compared with three exsiting neural network operators, `PCA-Net`, `FNO`, and `POD-DeepOnet` from the view of test error, error profile, and evaluation cost.

## Data (PDE paired input-output functions)
The PDE problems used for validation are:
1. Navier-Stokes equation (2D, structured grids)
2. Helmholtz equation (2D, structured grids)
3. Structural mechanics (2D, unstructured discretization)
4. Advection equation (1D, structured grids)
5. Darcy flow (2D, unstructured discretization)

All data for training and testing can be found in https://drive.google.com/drive/folders/1vmmPTwiIIbdOVTC209OKArcyjuYZcMHU?usp=sharing

The data used on problem 1, 2, and 4 are from paper Maarten V. de Hoop, Daniel Zhengyu Huang, Elizabeth Qian, Andrew M. Stuart. "[The Cost-Accuracy Trade-Off In Operator Learning With Neural Networks](https://arxiv.org/abs/2203.13181)." And the original address is https://data.caltech.edu/records/20091.
The data used on problem 5 is generated as suggested in paper Lu Lu, Xuhui Meng, Shengze Cai, Zhiping Mao, Somdatta Goswami, Zhongqiang Zhang, and George Em
Karniadakis. "A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data"
