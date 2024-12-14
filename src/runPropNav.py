"""
Main driver file for running the propNav simulation
set the desired input variables here, and loop through all combinations
"""

import numpy as np
import matplotlib.pyplot as plt
from eom_propnav import eom_propnav
from RK4 import RK4

# ------------- Define parameters here -----------------
theta_HE = np.array([2, 7, 15])*(np.pi/180)  # heading error (rad)
beta_0 = 10*(np.pi/180)  # initial target heading (rad)
r0_P = [0, 1000]  # initial pursuer position (m)
r0_T = [3000, 1000]  # initial target position (m)
mach1 = 340  # speed of sound (m/s)
v0_P = 0.9*mach1  # initial pursuer velocity magnitude (m/s)
v0_T = 0.25*mach1  # initial target velocity magnitude (m/s)
N = [2, 3, 4]  # prop nav gain (dimensionless)
accel_limit_P = 50  # pursuer perpendicular acceleration limit (m/s^2)
dt = 0.1  # time step (s)
tf = [3, 10, 15]  # final sim flight time (s)
# ------------------------------------------------------

# initialize the quantities to be plotted
miss_distance = np.zeros((3,3,3))
theta_intercept = np.zeros((3,3,3))

# loop over the 3 varying inputs, running the simulation each time
for i in range(len(theta_HE)):
    for j in range(len(N)):
        for k in range(len(tf)):

            # construct the time vector
            time = np.arange(0, tf[k], dt)

            # initial state vector based on given params
            init_state = np.transpose(np.array(
                [beta_0,
                 r0_T[0], r0_T[1],
                 r0_P[0], r0_P[1],
                 v0_T*np.cos(beta_0), v0_T*np.sin(beta_0),
                 v0_P*np.cos(theta_HE[i]), v0_P*np.sin(theta_HE[i])]
            ))

            # run the sim and collect data
            state, miss_distance[i,j,k], theta_leadHE = RK4(lambda y: eom_propnav(y,0,N[j],theta_HE[i],accel_limit_P), init_state, time)

            # calculate the intercept angle
            theta_intercept[i,j,k] = (theta_leadHE + theta_HE[i])*(180/np.pi)  # deg


# plot miss distance vs tf
leg = []
plt.figure()
for i in range(len(theta_HE)):
    for j in range(len(N)):
        plt.plot(tf,miss_distance[i,j,:])
        leg.append('theta_HE = {:.0f}, N = {:.0f}'.format(theta_HE[i]*(180/np.pi), N[j]))
plt.xlabel('time of flight (s)')
plt.ylabel('miss distance (m)')
plt.title('Miss distance')
plt.legend(leg)
plt.grid()
# plt.show()
plt.savefig('outputs/miss_distance.png')

# plot intercept angle
leg = []
plt.figure()
for i in range(len(theta_HE)):
    for j in range(len(N)):
        plt.plot(tf,theta_intercept[i,j,:])
        leg.append('theta_HE = {:.0f}, N = {:.0f}'.format(theta_HE[i]*(180/np.pi), N[j]))
plt.xlabel('time of flight (s)')
plt.ylabel('angle (deg)')
plt.title('Intercept angle')
plt.legend(leg)
plt.grid()
# plt.show()
plt.savefig('outputs/intercept_angle.png')

# # trajectory plot
# plt.figure()
# plt.plot(state[1,:], state[2,:], state[3,:], state[4,:])
# plt.xlabel('range (m)')
# plt.ylabel('altitude (m)')
# plt.grid()
# plt.show()
