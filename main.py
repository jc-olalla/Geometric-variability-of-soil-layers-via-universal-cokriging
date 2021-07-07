import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cokriging_toolbox as cokrig
import data_generation  # generate synthetic data

# Input
data_folder = 'data/'  # folder where the interface and orientation data are located
contact_points_name = 'contact_points.csv'  # contact points between soil layers from Local point data
orientations_name = 'orientations.csv'  # orientations
drift_type = 'second_order'  # drift_type='zero_order', drift_type='first_order', drift_type='second_order'

dest_folder = 'potential_field/'  # Destination folder

# Domain boundaries
x_ini = 0.0; x_end = 50.0
y_ini = 0.0; y_end = 1.0
z_ini = -10.0; z_end = 0.0
nx = 100; ny = 1; nz = 20  # number of points along each coordinate

model_bounds = [x_ini, x_end, y_ini, y_end, z_ini, z_end]
model_resolution = [nx, ny, nz]

# Creat destination folder
try:
    os.mkdir(dest_folder)
except:
    pass

# Data reading
df_int = pd.read_csv(data_folder + contact_points_name)
df_or = pd.read_csv(data_folder + orientations_name)

# Model parameters (Chiles 2004)
a_range = ((x_end - x_ini) ** 2 + (y_end - y_ini) ** 2 + (z_end - z_ini) ** 2) ** 0.5  # range of the covariance
C0 = (a_range ** 2) / (14 * 3)  # variance

# Potential field method - Universal cokriging
geomodel = cokrig.potential_field_interpolation(df_int, df_or, C0, a_range)
geomodel.covariance_Z1new_Z1new()  # Interface covariance matrix
geomodel.covariance_Z2_Z2()  # Orientation covariance matrix
geomodel.covariance_Z1new_Z2()  # cross-covariance matrix
if drift_type == 'zero_order':
    geomodel.assemble_covariance(drift_type='zero_order')  # Full covariance matrix
if drift_type == 'first_order':
    geomodel.assemble_covariance(drift_type='first_order')  # Full covariance matrix
if drift_type == 'second_order':
    geomodel.assemble_covariance(drift_type='second_order')  # Full covariance matrix

# Solve system
geomodel.make_grid(0.0, 55.0, -10.0, 0.0, dl=0.5)
geomodel.calc_potential_field()

# Plot scalar field
fig = plt.figure(figsize=(5, 2.0))
ax = fig.add_axes([0.2, 0.3, 0.5, 0.5])
CS = ax.contour(geomodel.X_grid, geomodel.Z_grid, geomodel.Z1_est, cmap='jet', levels=40)
cbar = fig.colorbar(CS)
ax.set_xlabel('Distance X (m)')
ax.set_ylabel('Depth\nZ (m)')

for i in range(len(df_int)):
    if df_int['formation'][i] == 'layer1':
        plt.plot(df_int['X'][i], df_int['Z'][i], 'ro')
    else:
        plt.plot(df_int['X'][i], df_int['Z'][i], 'bo')

u_beta = np.zeros((len(df_or), ))
v_beta = np.zeros((len(df_or), ))
x_beta = np.zeros((len(df_or), ))
z_beta = np.zeros((len(df_or), ))
for i, (azi, dipi) in enumerate(zip(df_or['azimuth'].__array__(), df_or['dip'].__array__())):
    u_beta[i] = np.sin(dipi * np.pi / 180.0) * np.sin(azi * np.pi / 180.0)
    v_beta[i] = np.cos(dipi * np.pi / 180.0)
    x_beta[i] = df_or['X'][i]
    z_beta[i] = df_or['Z'][i]

ax.quiver(x_beta, z_beta, u_beta, v_beta, color='black', label='orientations')
ax.set_ylim(-10, 0)
ax.set_xlim(0, 55.0)
fig.savefig(dest_folder + 'potential_field.png', format='png', dpi=1200)

# # Plot gradient of the potential field
# fig.gca().set_aspect('equal')
# gradz, gradx = np.gradient(geomodel.Z1_est, 0.5, 0.5)
# mod_grad = (gradx ** 2 + gradz ** 2) ** 0.5
# gradx = gradx / mod_grad
# gradz = gradz / mod_grad
# ax.quiver(geomodel.X_grid, geomodel.Z_grid, gradx, gradz, color='black')



plt.show()

