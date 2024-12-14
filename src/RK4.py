"""
Runge-Kutta 4th order integration

Given some differential equation function definition, initial conditions, and a time vector:
Numerically solve the system state over time using RK4

"""

def RK4(ydot,y0,time):
    import numpy as np

    y = np.zeros((len(y0), len(time)))  # initialize the output
    y[:,0] = y0  # set the given initial conditions

    distance2target = np.zeros(len(time)) # initialize the miss distance array
    distance2target[0] = np.hypot(y0[3]-y0[1],y0[4]-y0[2])

    # loop over the given time vector
    for i in range(len(time)-1):
        # define the step size (although I'm just using fixed step integration here)
        h = time[i + 1] - time[i]
        # calculate the four stage updates
        # (I assumed that the system is time-invariant so no time value has to be passed into the function here)
        k1 = ydot(y[:,i])
        k2 = ydot(y[:,i] + h * k1 / 2)
        k3 = ydot(y[:,i] + h * k2 / 2)
        k4 = ydot(y[:,i] + h * k3)
        # final update
        y[:,i + 1] = y[:,i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # calculate the distance from target to pursuer
        distance2target[i+1] = np.hypot(y[3,i]-y[1,i],y[4,i]-y[2,i])

    # find miss distance as the minimum distance between target and pursuer
    # some input value combinations result in the pursuer not having enough time to reach the target;
    # others the pursuer gets there really fast and then "overshoots" for the rest of the sim time
    miss_distance = np.min(distance2target)

    # time index of the minimum miss distance (i'm calling this the intercept point)
    m = np.argmin(distance2target)

    # calculate the lead angle at the intercept point
    # Line of sight (LOS) angle:
    dposx_TP = y[1,m]-y[3,m] # delta x position between target/pursuer
    dposy_TP = y[2,m]-y[4,m] # delta y position between target/pursuer
    theta_LOS = np.arctan(dposy_TP/dposx_TP)
    # pursuer to target velocity magnitude ratio:
    gamma = np.hypot(y[7,m],y[8,m]) / np.hypot(y[5,m],y[6,m])
    # lead angle + heading error
    # this is just to avoid passing in theta_HE unnecessarily here; return the summed value and let the main script
    # subtract out theta_HE as needed
    theta_leadHE = np.arcsin(np.sin(y[0,m]+theta_LOS)/gamma)

    return y, miss_distance, theta_leadHE

