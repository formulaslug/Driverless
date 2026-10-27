
#PID COEFF
K_P_LONGITUDINAL = 0.5 #increase to make more aggressive on error
K_I_LONGITUDINAL = 0.00 #increase to correct for steady state error
K_D_LONGITUDINAL = 0.01 #increase to correct for overshoot

#PURE PURSUIT

K_DD = 0.3 #lookahead distance scaler, increase to look further ahead
WHEELBASE = 1.529 # meters
MIN_LOOKAHEAD = 2.0 # meters




STEERING_RATIO = 6.49

#vd
M = 45.36 #total vehichle + driver mass (kg)
R_WHEEL = 5.0 #rear wheel radius ( meters)
T_GEARBOX = 3.25 #Gearbox reduction ratio
C_DRAG = 1.747 #drag coefficient
P_AIR = 1.162 #air density
A_FRONT = 0.91 #frontal area
C_R1 = 0.01 #rolling resistance coefficient

#placeholder but unsued below
J_WHEEL = 1  # wheel rotational moment inertia
J_M = 1.0 #electric motor rotational moment inertia
J_E = J_M + (T_GEARBOX ** 2) * J_WHEEL + M * (R_WHEEL * T_GEARBOX) ** 2 #computed total effective rotational inertia of the vehicle


