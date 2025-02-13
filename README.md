# Physics-informed Machine Learning for Solving Forward and Inverse 3D Multi Layered Terzaghi Consolidation Problems

This repository contains files used for my UROP placement supervised by Dr. Truong Le, focussing on using physics-informed machine learning to create forward and inversion solvers for Terzaghi problems.

In our research, we focus on developing a PINN model that solves the Terzaghi consolidation problem. Typically, this requires FEMs (finite element methods) which are computationally expensive.

PINNs work by incorporating physics into a neural network model and can be used to solve a partial differential equation (PDE) given boundary conditions and constrains. This is particularly useful for modelling fluid flows and heat diffusion. 

All PINNs have been developed using [SciANN](https://github.com/sciann/) [1], a high-level artificial neural networks API, written in Python using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) backends. As recommended by the developers, all code is based on `Python == 3.9` and `Tensorflow/Keras == 2.10`. The package allows for users to learn and change the physics of the problem simply.

We acknowledge computational resources and support provided by the Imperial College Research Computing Service (http://doi.org/10.14469/hpc/2232).

## 3D Terzaghi Problem

The Terzaghi consolidation problem is essential for predicting the behaviour of soil layers under a variety of loading conditions. The pore pressure flow 𝑢 through a 3D soil layer can be described by the following PDE, where $c_{v}$ is the coefficient of consolidation in $m^{3}$/year which depends on the properties of the soil, such as its compressibility

$$c_x \frac{\partial^2 u}{\partial x^2} + c_y \frac{\partial^2 u}{\partial y^2} + c_z \frac{\partial^2 u}{\partial z^2} = \frac{\partial u}{\partial t}$$

## Method

1. **Generate training data**  
   10,000 random samples are generated over a 1x1x1 grid from t = 0 to 1, with a higher concentration of samples at the boundary conditions.

2. **Construct the PINN**  
   The PINN is constructed using the `tanh` activation function with 4 hidden layers of 40 neurons each. The x, y, z, t values of the training data are used as inputs.

3. **Define the Physics**  
   `u(x,y,z,0) = p_0`  
   `u(0,y,z,t) = u(l,y,z,t) = 0`  
   `u(x,0,z,t) = u(x,b,z,t) = 0`  
   `∂u(x,y,0,t)/∂z = ∂u(x,y,h,t)/∂z = 0`

The above physics is defined for the 3D case with undrained top and bottom boundaries. An initial pore pressure is applied with magnitude `p_0 = 1.0 Pa`. The physics-informed targets are rearranged to give target values of 0.

## Results

### Training

The loss function evaluates the error of the PINN from the target values. The model uses the physics-informed targets and is trained to minimise our total loss with a decaying learning rate. For a trained model of 100,000 epochs, we obtain a minimal loss which has a negligible difference when compared to an exact solution obtained via FEM/FDM methods.

![Epochs_3D_50k](https://github.com/user-attachments/assets/09091535-44ea-455c-ab98-5acacf116e1e)

### Forward

Once trained, the model can quickly evaluate a pore pressure for given x, y, z, t variables. We consider both a one layer and three layers case by varying 𝑐_𝑣 with depth for soil, peat and sand layers. For undrained top and bottom boundaries, we achieve the following contour plots animations for a 100x100x100 grid:

![3D-Animation_80](https://github.com/user-attachments/assets/93f26311-7af9-44e1-bd5a-056667739eb6)

![3D-Animation_Layered_60](https://github.com/user-attachments/assets/d602549d-0811-4aef-91e4-3dd707ba1cdc)

The result is mesh-free, allowing us to effectively ‘zoom’ into the contour plot and view areas in more detail. This is not possible using FEM software as the data is constrained to a mesh. The model uses interpolation to predict the pore pressure values of untrained points.

![image](https://github.com/user-attachments/assets/10b9d76f-37af-4fc4-ba35-e84062bc2ee3)


### Inversion

We can also use the PINN to accurately predict the coefficient of consolidation for a given exact solution. By configuring the loss function to be between the PINN’s PDE solution and the exact solution, we train the model to vary an initialised and unknown $𝑐_{v}$ value to minimise the total loss. The MSE (mean squared error) loss function is given below, where $û_{i}$ is the exact pore pressure value and $u_{i}$  is the PINN’s approximation for a given data point 𝑖 and dimension 𝑛. $N_{𝑢}$  = 5000 points are randomly selected from the exact solution.

$$MSE=\frac1{N_u}\sum_{i=1}^{N_u}|\hat{u_i}-u_i|^2+\frac1{N_u}\sum_{i=1}^{N_u}\left|\frac{\partial\hat{u_i}}{\partial t}-C_v\left(\sum\frac{\partial^2\hat{u_i}}{\partial n^2}\right)\right|^2$$

For a consolidation parameter of $c_{v}$ = 0.75, we train over 1000 epochs and achieve fast convergence to the true value with minimal error. The model was trained for 122 seconds. To train the model, we use the Adam optimizer, which uses the MSE loss function to update the parameter after each epoch. After 1000 epochs, the L-BFGS optimizer can be used for further accuracy.

![download](https://github.com/user-attachments/assets/2eabd0ac-5d78-4b16-9c68-d5bb2725a5b1)

## Conclusion

Our results show that the PINN model can accurately output pore pressure data and predict unknown parameters from exact solution data. FEM software is typically costly and challenging to learn, while the PINN model can be easily tweaked by Python users for different drainage conditions. However,

-  The PINN model takes a few hours to train and can only provide an accurate output for a single parameter value. However, there has been research to solve this problem via transfer learning. [3]
-  The PINN model requires a high-core CPU or GPU to train and would otherwise take longer than typical FEM software to generate pore pressure data.

This research can be extended to more realistic and unsymmetrical examples to aid civil engineers in making informed decisions in the real world.

## References

1.  Yuan, B., Heitor, A., Wang, H. and Chen, X. (2024a). Physics-informed Deep Learning to Solve Three-dimensional Terzaghi Consolidation Equation: Forward and Inverse Problems. arXiv (Cornell University). doi:https://doi.org/10.48550/arxiv.2401.05439.
2.  Haghighat, E. and Juanes, R. (2021). SciANN: A Keras/TensorFlow wrapper for scientific computations and physics-informed deep learning using artificial neural networks. Computer Methods in Applied Mechanics and Engineering, [online] 373, p.113552. doi:https://doi.org/10.1016/j.cma.2020.113552.
3.  Haghighat, E., Raissi, M., Moure, A., Gomez, H. and Juanes, R. (2021). A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. Computer Methods in Applied Mechanics and Engineering, 379, p.113741. doi:https://doi.org/10.1016/j.cma.2021.113741.
