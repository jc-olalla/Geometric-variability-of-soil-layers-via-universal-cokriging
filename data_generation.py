import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Input
data_folder = 'data/'
contact_points_name = 'contact_points.csv'
orientations_name = 'orientations.csv'

formation_name_1 = 'layer1'
x_1_alpha = np.array([10.0, 25.0, 40.0])  # x-coordinates of the interfaces of the first layer
z_1_alpha = np.array([-2.0, -1.0, -2.5])  # z-coordinates of the interfaces of the first layer

formation_name_2 = 'layer2'
x_2_alpha = np.array([10.0, 25.0, 45.0])  # x-coordinates of the second layer
z_2_alpha = np.array([-7.0, -8.5, -5.0])  # y-coordinates of the second layer

x_beta = np.array([5.0, 15.0, 15.0, 30.0, 35.0])  # x-coordinates of the orientations
z_beta = np.array([-5.0, -8.5, -4.0, -5.0, -8.0])  # z-coordinates of the orientations
azimut_beta = np.array([270.0, 90.0, 270.0, 90.0, 270.0])  # azimuth of the orientations
dip_beta = np.array([5.0, 25.0, 15.0, 25.0, 20.0])  # dip of the orientations

try:
    os.mkdir(data_folder)
except:
    pass

# Data frame generation: contact points
formation_1_alpha = [formation_name_1] * len(x_1_alpha)
formation_2_alpha = [formation_name_2] * len(x_2_alpha)

d_contact = {'X': x_1_alpha.tolist() + x_2_alpha.tolist(),
             'Y': np.zeros_like(x_1_alpha).tolist()+np.zeros_like(x_2_alpha).tolist(),
             'Z': z_1_alpha.tolist() + z_2_alpha.tolist(), 'formation': formation_1_alpha + formation_2_alpha}
df_contact = pd.DataFrame(data=d_contact)
df_contact.to_csv(data_folder + contact_points_name, index=False)  # save data frame

# Data frame generation: orientations
d_or = {'X': x_beta.tolist(),
        'Y': 0.0 * (len(x_beta) + len(x_beta)),
        'Z': z_beta.tolist(),
        'azimuth': azimut_beta.tolist(),
        'dip': dip_beta.tolist(),
        'polarity': 1,
        'formation': 'common'}  # formation is not a necessary field
df_or = pd.DataFrame(data=d_or)
df_or.to_csv(data_folder + orientations_name, index=False)  # save data frame

# Plot data
fig = plt.figure(figsize=(5, 2.0))
ax = fig.add_axes([0.2, 0.3, 0.4, 0.5])
ax.plot(x_1_alpha, z_1_alpha, 'ro', label='contact point\nlayer 1')
ax.plot(x_2_alpha, z_2_alpha, 'bo', label='contact point\nlayer 2')

u_beta = np.zeros((len(x_beta), ))
v_beta = np.zeros((len(x_beta), ))
for i, (azi, dipi) in enumerate(zip(azimut_beta, dip_beta)):
    u_beta[i] = np.sin(dipi * np.pi / 180.0) * np.sin(azi * np.pi / 180.0)
    v_beta[i] = np.cos(dipi * np.pi / 180.0)

ax.quiver(x_beta, z_beta, u_beta, v_beta, color='black', label='orientations')

ax.legend(bbox_to_anchor=(1.0, 1.05))
ax.set_xlabel('Distance X (m)')
ax.set_ylabel('Depth\nZ (m)')
ax.set_ylim(-10, 0)
ax.set_xlim(0, 55.0)


# Add labels to plot
for i, (x_1_alpha_i, z_1_alpha_i) in enumerate(zip(x_1_alpha, z_1_alpha)):  # contact points layer1
    label = '$x_{1\\alpha' + str(i) + '}$'
    ax.text(x_1_alpha_i + 1.5, z_1_alpha_i, label)

for i, (x_2_alpha_i, z_2_alpha_i) in enumerate(zip(x_2_alpha, z_2_alpha)):  # contact points layer2
    label = '$x_{2\\alpha' + str(i) + '}$'
    ax.text(x_2_alpha_i + 1.5, z_2_alpha_i, label)

for i, (x_beta_i, z_beta_i) in enumerate(zip(x_beta, z_beta)):
    label = '$x_{\\beta' + str(i + 1) + '}$'
    ax.text(x_beta_i + 1.5, z_beta_i, label)

fig.savefig(data_folder + 'data.png', format='png', dpi=1200)


# plt.show()
