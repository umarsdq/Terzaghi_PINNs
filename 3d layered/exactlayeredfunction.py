import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

save_directory = '/rds/general/user/us322/ephemeral/RMSE Testing'
os.makedirs(save_directory, exist_ok=True)

def exactlayered(N, Nt, cv1, cv2, cv3):
    Delta_x, Delta_y, Delta_z = 1/N, 1/N, 1/N
    Delta_t = 1/Nt  # Time step 4.63e-5 used in paper

    x_range = np.linspace(0, 1, N)
    y_range = np.linspace(0, 1, N) # Doubling for half plot
    z_range = np.linspace(0, 1, N)
    t_range = np.linspace(0, 1, Nt) # Change for t1+t2+t3 if needed

    layer_boundaries = [0, 1/3, 2/3, 1.0]  # Boundaries of the layers (normalized to [0,1])
    u_0 = 1.0  # Initial excess pore water pressure

    # Function to determine Cv based on z position
    def get_cv(z):
        for i in range(len(layer_boundaries) - 1):
            if layer_boundaries[i] <= z < layer_boundaries[i + 1]:
                return cv_layers[i]
        return cv_layers[-1]  # Default to the last layer's Cv

    cv_layers = [cv1,cv2,cv3]
    
    u = np.zeros((len(x_range), len(y_range), len(z_range), len(t_range)))
    u[:, :, :, 0] = u_0

    # Start time for progress tracking
    start_time = time.time()

    # Time-stepping loop
    for n in range(0, len(t_range) - 1):
        for i in range(1, len(x_range) - 1):
            for j in range(1, len(y_range) - 1):
                for k in range(1, len(z_range) - 1):
                    cv = get_cv(z_range[k])
                    u[i, j, k, n + 1] = u[i, j, k, n] + Delta_t * (
                        cv * ((u[i + 1, j, k, n] - 2 * u[i, j, k, n] + u[i - 1, j, k, n]) / Delta_x**2 +
                              (u[i, j + 1, k, n] - 2 * u[i, j, k, n] + u[i, j - 1, k, n]) / Delta_y**2 +
                              (u[i, j, k + 1, n] - 2 * u[i, j, k, n] + u[i, j, k - 1, n]) / Delta_z**2)
                    )

        # Apply boundary conditions for the 3D Terzaghi problem (Case 2)
        u[0, :, :, n + 1] = 0  # u(0,y,z,t) = 0
        u[-1, :, :, n + 1] = 0  # u(l,y,z,t) = 0
        u[:, 0, :, n + 1] = 0  # u(x,0,z,t) = 0
        u[:, -1, :, n + 1] = 0  # u(x,b,z,t) = 0

        # Neumann boundary conditions for z direction (zero flux)
        u[:, :, 0, n + 1] = u[:, :, 1, n + 1]  # ∂u/∂z at z=0
        u[:, :, -1, n + 1] = u[:, :, -2, n + 1]  # ∂u/∂z at z=h

        # Continuity of pore pressure and flux at layer interfaces
        for k in range(1, len(z_range) - 1):
            if z_range[k] in layer_boundaries:
                cv_below = get_cv(z_range[k - 1])
                cv_above = get_cv(z_range[k + 1])
                u[:, :, k, n + 1] = (cv_below * u[:, :, k - 1, n + 1] + cv_above * u[:, :, k + 1, n + 1]) / (cv_below + cv_above)

        # Print progress every 500 time steps and write to file
        if (n + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * (len(t_range) - n - 1) / (n + 1)
            hours, rem = divmod(remaining_time, 3600)
            minutes, seconds = divmod(rem, 60)
            progress_message = f"Time step {n + 1}/{len(t_range) - 1} completed. Estimated time remaining: {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            print(progress_message)
            
    # Save pore pressure data to a file after completing the simulation
    # np.save(os.path.join(save_directory, f'3D_Layered_{cv1}_{cv2}_{cv3}_{N}.npy'), u)

    print(f"Simulation completed for {cv1,cv2,cv3}")
    
    return u