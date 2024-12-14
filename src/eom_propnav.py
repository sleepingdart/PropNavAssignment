"""
Equations of motion for the proportional navigation scenario

This function defines the state space representation of the differential equations governing this 2D simulation. Given
a state input, it computes the state derivative. Additional inputs for the heading error, propnav gain, and pursuer
acceleration limits are also needed.

all state variables referenced to the fixed inertial frame:
state[0] = target heading angle
state[1] = target position, x axis
state[2] = target position, y axis
state[3] = pursuer position, x axis
state[4] = pursuer position, y axis
state[5] = target velocity, x axis
state[6] = target velocity, y axis
state[7] = pursuer velocity, x axis
state[8] = pursuer velocity, y axis

Some assumptions made for simplicity:
mass of the pursuer and target does not change over time (time-invariant differential equations)
target and pursuer are only subject to gravity as an external force
no sensing or actuation dynamics involved; pursuer instantaneously measures its own state and angles to the target, and
instantaneously achieves any control commands

"""

def eom_propnav(state,accel_T,N,theta_HE,accel_lim_P):
    import numpy as np
    # first calculate various angles relating the pursuer and target:

    # Line of sight (LOS) angle, based on target and pursuer coordinates
    dposx_TP = state[1]-state[3] # delta x position between target/pursuer
    dposy_TP = state[2]-state[4] # delta y position between target/pursuer
    theta_LOS = np.arctan(dposy_TP/dposx_TP)
    # LOS rate of change
    dvelx_TP = state[5]-state[7] # delta x velocity between target/pursuer
    dvely_TP = state[6]-state[8] # delta y velocity between target/pursuer
    theta_dot_LOS = (dposx_TP*dvely_TP - dposy_TP*dvelx_TP) / np.square((np.hypot(dposx_TP,dposy_TP)))
    # pursuer to target velocity magnitude ratio:
    gamma = np.hypot(state[7],state[8]) / np.hypot(state[5],state[6])
    # lead angle:
    theta_lead = np.arcsin(np.sin(state[0]+theta_LOS)/gamma) - theta_HE

    # pure propNav guidance law defines the pursuer acceleration to be applied (perpendicular to the pursuer's velocity)
    accel_P = N*np.hypot(state[7],state[8])*theta_dot_LOS
    # however, we're given a bound on how much acceleration the pursuer can achieve:
    if np.abs(accel_P) > accel_lim_P:
        # saturate the pursuer's acceleration based on given input
        accel_P = np.sign(accel_P)*accel_lim_P

    # now calculate the state derivatives:
    state_dot = np.zeros(9)

    # rate of change, target heading angle:
    state_dot[0] = accel_T / np.hypot(state[5],state[6])

    # rate of change, target position; this is just the state variable target velocity
    state_dot[1] = state[5]
    state_dot[2] = state[6]

    # rate of change, pursuer position; this is just the state variable pursuer velocity
    state_dot[3] = state[7]
    state_dot[4] = state[8]

    # rate of change, target velocity: influenced by any acceleration of the target
    state_dot[5] = accel_T*np.sin(state[0])
    state_dot[6] = accel_T*np.cos(state[0]) - 9.81  # gravitational acceleration on target

    # rate of change, pursuer velocity: apply the propNav acceleration to determine how the pursuer velocity is updated
    state_dot[7] = -accel_P*np.sin(theta_LOS + theta_lead + theta_HE)
    state_dot[8] = accel_P*np.cos(theta_LOS + theta_lead + theta_HE) - 9.81  # gravitational acceleration on pursuer

    return state_dot

