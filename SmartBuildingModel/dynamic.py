# Simon Weideskog, Louise Calmered, Marie Halling
# 2022 June

import numpy as np
import dm4bem
import matplotlib.pyplot as plt
import pandas as pd
import tuto


# Room dimensions
ROOF_HEIGHT = 2.5
WINDOW_AREA = 2*1.5
APT_LENGTH = 5
APT_WIDTH = 4
BATHROOM_WIDTH = 2
BATHROOM_LENGTH = 2
RADIATOR_AREA = 2*0.5
DOOR_AREA = 2*0.6

# Wall thicknesses
INSULATION_THICKNESS = 0.1
OUTER_BRICK_WALL_THICKNESS = 0.15
HALL_BRICK_WALL_THICKNESS = 0.15
BATHROOM_BRICK_WALL_THICKNESS = 0.1

# Wall properties
ε_wLW = 0.85    # long wave wall emmisivity (brick)
α_wSW = 0.3     # absortivity white expanded polysterene surface
α_wSB = 0.4     # absortivity white expanded polysterene surface
ε_gLW = 0.9     # long wave glass emmisivity (glass pyrex)
τ_gSW = 0.83    # short wave glass transmitance (glass)
α_gSW = 0.1     # short wave glass absortivity

σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

# Regulator settings
Kp = 1e-3
#Kp = 1

# --------  A-MATRIX -----------
# ------------------------------
A = np.zeros([16, 10]) #create an matrix with zeros
# adds values for the ones /= 0
A[0, 0] = 1
A[1, 0], A[1, 1] = -1, 1
A[2, 1], A[2, 2] = -1, 1
A[3, 2], A[3, 3] = -1, 1
A[4, 3], A[4, 4] = -1, 1
A[5, 4], A[5, 5] = -1, 1
A[6, 5] = 1
A[7, 5] = 1
A[8, 5], A[8, 6] = 1,-1
A[9, 6], A[9, 7] = -1, 1
A[10, 5], A[10, 7]  = 1, -1
A[11, 8] = 1
A[12, 6], A[12, 8]  = 1, -1
A[13, 9] = 1
A[14, 5], A[14, 9]  = 1, -1
A[15, 5] = 1

# --------  G-MATRIX -----------
# ------------------------------
h_brick_air = 53.44 # Taken from previous work on wall construction
h_ins_air = 1.46 # Taken from previous work on wall construction
k_ins = 0.04 # Expanded polystyrene
k_brick = 0.6

conv_outer_wall = h_ins_air*(APT_WIDTH*ROOF_HEIGHT-WINDOW_AREA)
cond_ins = k_ins*(APT_WIDTH*ROOF_HEIGHT-WINDOW_AREA)/(INSULATION_THICKNESS/2)
cond_brick_out = k_brick*(APT_WIDTH*ROOF_HEIGHT-WINDOW_AREA)/(OUTER_BRICK_WALL_THICKNESS/2)
conv_inner_wall = h_brick_air*(APT_WIDTH*ROOF_HEIGHT-WINDOW_AREA)
conductivity_window = 1.2*WINDOW_AREA # https://aspirebifolds.co.uk/2018/03/what-are-typical-u-values-on-windows-and-doors/
conv_radiator = 10*RADIATOR_AREA
conductivity_bath_door = 1.06*DOOR_AREA # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://millpanel.com/_file/159/u-values_uk.pdf
conductivity_bath_wall = k_brick*((BATHROOM_WIDTH+BATHROOM_LENGTH)*ROOF_HEIGHT-DOOR_AREA)/(BATHROOM_BRICK_WALL_THICKNESS/2)
conductivity_hall_bathroom_wall = k_brick*(BATHROOM_WIDTH*ROOF_HEIGHT)/(HALL_BRICK_WALL_THICKNESS/2)
conductivity_hall_wall = k_brick*(BATHROOM_WIDTH*ROOF_HEIGHT-DOOR_AREA)/(HALL_BRICK_WALL_THICKNESS/2)
conductivity_hall_door = 1.06*DOOR_AREA

g=[None]*16
g[0]=conv_outer_wall
g[1]=cond_ins
g[2]=cond_ins
g[3]=cond_brick_out
g[4]=cond_brick_out
g[5]=conv_inner_wall
g[6]=conductivity_window
#g[7]=conv_radiator
g[7]=conv_radiator*Kp
g[8]=conductivity_bath_door
g[9]=conductivity_bath_wall
g[10]=conductivity_bath_wall
g[11]=conductivity_hall_bathroom_wall
g[12]=conductivity_hall_bathroom_wall
g[13]=conductivity_hall_wall
g[14]=conductivity_hall_wall
g[15]=conductivity_hall_door

G = np.diag(g)

# --------  b-VECTOR -----------
# ------------------------------

T0 = 20
T_isp = 20
T_hall = 20

b = np.array([T0, 0, 0, 0, 0, 0, T0, T_isp, 0, 0, 0, T_hall, 0, T_hall, 0, T_hall])

# --------  f-VECTOR -----------
# ------------------------------

F0 = 50 * (APT_WIDTH*ROOF_HEIGHT-WINDOW_AREA) # Average during 24h
F1 = 50 * τ_gSW * WINDOW_AREA # https://www.engineeringtoolbox.com/radiant-heat-windows-d_1005.html
Q_a = 50
Q_b = 100

#f = [0]*10
f = np.array([F0, 0, 0, 0, F1, Q_a, Q_b, 0, 0, 0])

A_sys = np.transpose(A) @ G @ A # The A-matrix in the equation B = A*theta
B_sys = np.transpose(A) @ G @ b + f
theta = np.linalg.solve(A_sys,B_sys)
#print(theta)

print('Main room temp: '+str(theta[5]))

# --------  C-VECTOR -----------
# ------------------------------

c_ins = 850*22*INSULATION_THICKNESS*(ROOF_HEIGHT*APT_WIDTH-WINDOW_AREA)
c_wall = 900*2000*OUTER_BRICK_WALL_THICKNESS*(ROOF_HEIGHT*APT_WIDTH-WINDOW_AREA)
c_bathroom_to_room = 900*2000*BATHROOM_BRICK_WALL_THICKNESS*((BATHROOM_WIDTH+BATHROOM_LENGTH)*ROOF_HEIGHT-DOOR_AREA)
c_hall_bathroom = 900*2000*HALL_BRICK_WALL_THICKNESS*BATHROOM_WIDTH*ROOF_HEIGHT
c_hall_room = 900*2000*((APT_WIDTH-BATHROOM_WIDTH)*ROOF_HEIGHT-DOOR_AREA)*HALL_BRICK_WALL_THICKNESS
c_air_main_room = 1006*1.208*(APT_WIDTH*APT_LENGTH-BATHROOM_WIDTH*BATHROOM_LENGTH)*ROOF_HEIGHT
c_air_bathroom = 1006*1.208*(BATHROOM_WIDTH*BATHROOM_LENGTH*ROOF_HEIGHT)


C = np.diag([0, c_ins, 0, c_wall, 0, c_air_main_room, c_air_bathroom, c_bathroom_to_room, c_hall_bathroom, c_hall_room])

# -------  Step responce --------
# -------------------------------

# b = np.array([T0, 0, 0, 0, 0, 0, T0, T_isp, 0, 0, 0, T_hall, 0, T_hall, 0, T_hall])
b_step = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1])

# f = np.array([F0, 0, 0, 0, F1, Q_a, Q_b, 0, 0, 0])
f_step = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0])

y = np.zeros(10)
y[[5]] = 1

[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, C, f, y)

dt_max = min(-2. / np.linalg.eig(As)[0])
print('Max time step: '+str(dt_max)) # = 164,7

dt = 150

duration = 3600 * 24 * 10

n = int(np.floor(duration/dt))

t = np.arange(0, n*dt, dt)

n_tC = As.shape[0]


# u_step = [T0, T0, T_isp, T_hall, T_hall, T_hall, F0, F1, Q_a, Q_b]
u_step = np.zeros([10, n])
#u_step[0:2, :] = np.ones([2, n])
u_step[0:3, :] = np.ones([3, n])
u_step[3:6, :] = np.ones([3, n])

temp_exp = np.zeros([n_tC, t.shape[0]])

I = np.eye(n_tC)
for i in range(n-1):
    temp_exp[:, i+1] = (I + dt*As) @ temp_exp[:, i] + dt*Bs @ u_step[:, i]

y_exp = Cs @ temp_exp + Ds @ u_step

fig, ax = plt.subplots()
ax.plot(t / 3600, y_exp.T)
ax.set(xlabel='Time [h]',
       ylabel='$T_i$ [°C]',
       title='Step input: To = 1°C')

# -------  With weather data --------
# -----------------------------------

'''
GIVEN CODE from file t03CubeFB.ipynb:
https://github.com/cghiaus/dm4bem/blob/main/t03/t03CubeFB.ipynb
'''
filename = 'FRA_Lyon.074810_IWEC.epw'
start_date = '2000-01-03 12:00:00'
end_date = '2000-01-08 12:00:00'

[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[['temp_air','dir_n_rad','dif_h_rad']]
del data
weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather[(weather.index >= start_date) & (weather.index < end_date)]

surface_orientation = {'slope': 90, 'azimuth': 0, 'latitude': 45}
albedo = 0.2
rad_surf1 = dm4bem.sol_rad_tilt_surf(weather, surface_orientation, albedo)
rad_surf1['Φt1'] = rad_surf1.sum(axis=1)

data = pd.concat([weather['temp_air'], rad_surf1['Φt1']], axis=1)
data = data.resample(str(dt) + 'S').interpolate(method='linear')
data = data.rename(columns={'temp_air': 'To'})

data['Ti'] = T_isp * np.ones(data.shape[0])
data['Th'] = T_hall * np.ones(data.shape[0])
data['Qa'] = Q_a * np.ones(data.shape[0])
data['Qb'] = Q_b * np.ones(data.shape[0])


t = dt * np.arange(data.shape[0])

# u = [T0, T0, T_isp, T_hall, T_hall, T_hall, F0, F1, Q_a, Q_b]
u = pd.concat([data['To'], data['To'], data['Ti'], data['Th'], data['Th'], data['Th'],
               α_wSW * (APT_WIDTH*ROOF_HEIGHT-WINDOW_AREA) * data['Φt1'],
               τ_gSW * α_wSB * WINDOW_AREA * data['Φt1'],
               data['Qa'], data['Qb']], axis=1)

temp_exp = 20 * np.ones([As.shape[0], u.shape[0]])

for k in range(u.shape[0] - 1):
    temp_exp[:, k + 1] = (I + dt * As) @ temp_exp[:, k]\
        + dt * Bs @ u.iloc[k, :]

# Plotting
y_exp = Cs @ temp_exp + Ds @ u.to_numpy().T
#q_HVAC = Kp * (data['Ti'] - y_exp[0, :])
q_radiator = (data['Ti'] - y_exp[0, :])*conv_radiator

fig, axs = plt.subplots(2, 1)
# plot indoor and outdoor temperature
axs[0].plot(t / 3600, y_exp[0, :], label='$T_{indoor}$')
axs[0].plot(t / 3600, data['To'], label='$T_{outdoor}$')
axs[0].set(xlabel='Time [h]',
           ylabel='Temperatures [°C]',
           title='Simulation for weather')
axs[0].legend(loc='upper right')

# plot total solar radiation and HVAC heat flow
axs[1].plot(t / 3600,  q_radiator, label='$q_{radiator}$')
axs[1].plot(t / 3600, data['Φt1'], label='$Φ_{total}$')
axs[1].set(xlabel='Time [h]',
           ylabel='Heat flows [W]')
axs[1].legend(loc='upper right')

fig.tight_layout()

plt.show()
