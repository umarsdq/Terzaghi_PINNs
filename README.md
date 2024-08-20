# Terzaghi_PINNs
This repository contains files used for my UROP placement supervised by Dr. Truong Le, focussing on using physics-informed machine learning to create forward and inversion solvers for Terzaghi problems.

In our research, we focus on developing a PINN model that solves the Terzaghi consolidation problem. Typically, this requires FEMs (finite element methods) which are computationally expensive.

PINNs work by incorporating physics into a neural network model and can be used to solve a partial differential equation (PDE) given boundary conditions and constrains. This is particularly useful for modelling fluid flows and heat diffusion. 

The Terzaghi consolidation problem, which describes the pore pressure flow through a soil layer, can be described by the following PDE in 3-dimensions, where cx, cy, cz are the consolidation parameters for each dimension given in m^2/year, and u is the pore pressure.

$$
c_x \frac{\partial^2 u}{\partial x^2} + c_y \frac{\partial^2 u}{\partial y^2} + c_z \frac{\partial^2 u}{\partial z^2} = \frac{\partial u}{\partial t}
$$

For a trained model of 50,000 epochs, we obtain a minimal loss which has a negligible difference when compared to an exact solution obtained via FEM/FDM methods.

![Epochs_3D_50k](https://github.com/user-attachments/assets/09091535-44ea-455c-ab98-5acacf116e1e)

We can also use the PINN to accurately predict the input parameters for any given exact solution. By changing the loss function to be between the PINN’s PDE solution and the exact solution, we minimise the loss by varying an initialised and unknown cv value. For a 2D case, the parameters converge to the true value fairly quickly: 

![InvertedCxCz_2D_LR](https://github.com/user-attachments/assets/084a6127-b7b9-413b-b442-82c40ec50180)

Finally, we consider a layered case by varying the cv parameter with depth. For the two cases of one layer and a typical sand, peat and soil layer, we achieve the following contour plots:

![3D-Animation_80](https://github.com/user-attachments/assets/93f26311-7af9-44e1-bd5a-056667739eb6)

![3D-Animation_Layered_60](https://github.com/user-attachments/assets/d602549d-0811-4aef-91e4-3dd707ba1cdc)

We acknowledge computational resources and support provided by the Imperial College Research Computing Service (http://doi.org/10.14469/hpc/2232).


