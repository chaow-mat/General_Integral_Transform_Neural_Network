# Generalized Integral Transform Neural Network (GIT-Net) for Operator Learning

This work is to propose a novel neural network, GIT-Net, for operator learning. It is demonstrated by solving several PDEs and compared with three exsiting neural network operators, `PCA-Net`, `FNO`, and `POD-DeepOnet` from the view of test error, error profile, and evaluation cost.

## Data (paired input-output functions)
The PDE problems used for validation are:
1. Navier-Stokes equation (2D, structured meshes)
2. Helmholtz equation (2D, structured meshes)
3. Structural mechanics (2D, unstructured meshes)
4. Advection equation (1D, structured meshes)
5. Darcy flow (2D, unstructured meshes)

All data for training and testing can be found in https://drive.google.com/drive/folders/1vmmPTwiIIbdOVTC209OKArcyjuYZcMHU?usp=sharing. For the problems defined in unstructured meshes, the intepolated data on a structured meshes are provided.

### Reference
1. The data used on problem 1, 2, and 4 are from paper _Maarten V. de Hoop, Daniel Zhengyu Huang, Elizabeth Qian, Andrew M. Stuart. "[The Cost-Accuracy Trade-Off In Operator Learning With Neural Networks](https://arxiv.org/abs/2203.13181)."_ And the original address is https://data.caltech.edu/records/20091.
2. The data used on problem 5 is generated as suggested in paper _Lu Lu, Xuhui Meng, Shengze Cai, Zhiping Mao, Somdatta Goswami, Zhongqiang Zhang, and George Em Karniadakis. "A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data"_.

## Code 
### Training
Example of GIT-Net for Navier-Stokes equation
```
cd Navier-stokes/
python3 GIT_ns.py --c_width 32 --d_width 512 --M 2500 --state 'train' --device 0
```

### Testing
Example of GIT-Net for Navier-Stokes equation
```
cd Navier-stokes/
python3 GIT_ns.py --c_width 32 --d_width 512 --M 2500 --state 'eval' --path_model 'models/GIT/GIT_2500_dw_512_cw32.model' --device 0
```

## Results (comparison among GIT-Net, PCA-Net, POD-DeepOnet, and FNO)
### Test error

### Error profile

### Evalustion cost



