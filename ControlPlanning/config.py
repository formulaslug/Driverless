
#PID COEFF
K_P_LONGITUDINAL = 0.5 #increase to make more aggressive on error
K_I_LONGITUDINAL = 0.00 #increase to correct for steady state error
K_D_LONGITUDINAL = 0.01 #increase to correct for overshoot

#PURE PURSUIT

K_DD = 0.3 #lookahead distance scaler, increase to look further ahead
WHEELBASE = 1.5
MIN_LOOKAHEAD = 2.0


#vd
STEERING_RATIO = 5.0
M = 1.0 #total vehichle + driver mass
R_WHEEL = 1.0 #rear wheel radius
T_GEARBOX = 1.0 #Gearbox reduction ratio
C_DRAG = 0.3 #drag coefficient
P_AIR = 1.225 #air density
A_FRONT = 2.2 #frontal area
C_R1 = 0.01 #rolling resistance coefficient
J_M = 1.0 #electric motor rotational moment inertia
J_WHEEL = 1  # wheel rotational moment inertia

J_E = J_M + (T_GEARBOX ** 2) * J_WHEEL + M * (R_WHEEL * T_GEARBOX) ** 2 #computed total effective rotational inertia of the vehicle


