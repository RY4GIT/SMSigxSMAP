import numpy as np
def q_drydown(t, k, q, delta_theta, theta_star=1.0, theta_w=0.0):
    s0 = (delta_theta - theta_w) ** (1 - q)

    a = (1 - q) / ((theta_star - theta_w) ** q)

    return (-k * a * t + s0) ** (1 / (1 - q)) + theta_w

def exponential_drydown(t, delta_theta, theta_w, tau):
    return delta_theta * np.exp(-t / tau) + theta_w

def loss_model(theta, q, k, theta_wp=0., theta_star=1.):
    d_theta = -k * ( ( theta - theta_wp ) / ( theta_star - theta_wp ) ) ** (q)
    return d_theta
