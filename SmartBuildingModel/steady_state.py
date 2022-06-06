# Simon Weideskog 2022

import numpy as np

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

τ_gSW = 0.83    # short wave glass transmitance (glass)

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
g[7]=conv_radiator
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

T0 = 10
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

print('Main room temp: '+str(theta[5]))
print('All temperatures: '+str(theta))

# --------  C-VECTOR -----------
# ------------------------------

c_ins = 850*22*INSULATION_THICKNESS*(ROOF_HEIGHT*APT_WIDTH-WINDOW_AREA)
c_wall = 900*2000*OUTER_BRICK_WALL_THICKNESS*(ROOF_HEIGHT*APT_WIDTH-WINDOW_AREA)
c_bathroom_to_room = 900*2000*BATHROOM_BRICK_WALL_THICKNESS*((BATHROOM_WIDTH+BATHROOM_LENGTH)*ROOF_HEIGHT-DOOR_AREA)
c_hall_bathroom = 900*2000*HALL_BRICK_WALL_THICKNESS*BATHROOM_WIDTH*ROOF_HEIGHT
c_hall_room = 900*2000*((APT_WIDTH-BATHROOM_WIDTH)*ROOF_HEIGHT-DOOR_AREA)*HALL_BRICK_WALL_THICKNESS
c_air_main_room = 1006*1.208*(APT_WIDTH*APT_LENGTH-BATHROOM_WIDTH*BATHROOM_LENGTH)*ROOF_HEIGHT
c_air_bathroom = 1006*1.208*(BATHROOM_WIDTH*BATHROOM_LENGTH*ROOF_HEIGHT)

C = np.array([0, c_ins, 0, c_wall, 0, c_air_main_room, c_air_bathroom, c_bathroom_to_room, c_hall_bathroom, c_hall_room])
