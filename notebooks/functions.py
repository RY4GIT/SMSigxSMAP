import numpy as np


def q_drydown(t, k, q, delta_theta, theta_star=1.0, theta_w=0.0):
    """
    Calculate the drydown curve for soil moisture over time using non-linear plant stress model.

    Parameters:
    t (int): Timestep, in day.
    k (float): Product of soil thickness (z) and maximum rate of change in normalized soil moisture (k), equivalent to maximum ET rate (ETmax), in m3/m3/day.
    q (float): Degree of non-linearity in the soil moisture response.
    delta_theta (float): Shift/increment in soil moisture after precipitation, in m3/m3.
    theta_star (float, optional): Critical soil moisture content, equal to s_star * porosity, in m3/m3. Default is 1.0.
    theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3. Default is 0.0.

    Returns:
    float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.
    """

    b = delta_theta ** (1 - q)

    a = (1 - q) / ((theta_star - theta_w) ** q)

    return (-k * a * t + b) ** (1 / (1 - q)) + theta_w


def exponential_drydown(t, delta_theta, theta_w, tau):
    """
    Calculate the drydown curve for soil moisture over time using non-linear plant stress model.

    Parameters:
        t (int): Timestep, in day.
        delta_theta (float): Shift/increment in soil moisture after precipitation, in m3/m3.
        theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3. Default is 0.0.
        tau (float): decay rate, in 1/day.

    Returns:
        float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.

    Reference:
        McColl, K.A., W. Wang, B. Peng, R. Akbar, D.J. Short Gianotti, et al. 2017.
        Global characterization of surface soil moisture drydowns.
        Geophys. Res. Lett. 44(8): 3682–3690. doi: 10.1002/2017GL072819.
    """
    return delta_theta * np.exp(-t / tau) + theta_w


def exponential_drydown2(t, delta_theta, theta_w, theta_star, k):
    """
    Same as exponential_drydown but tau parameter is calculated using the same parameter set as non-linear model
    """
    tau = (theta_star - theta_w) / k
    return delta_theta * np.exp(-t / tau) + theta_w


def loss_model(theta, q, k, theta_wp=0.0, theta_star=1.0):
    """
    Calculate the loss function (dtheta/dt vs theta relationship) using non-linear plant stress model

    Parameters:
    theta (float): Volumetric soil moisture content, in m3/m3.
    q (float): Degree of non-linearity in the soil moisture response.
    k (float): Product of soil thickness (z) and maximum rate of change in normalized soil moisture (k), equivalent to maximum ET rate (ETmax), in m3/m3/day.
    theta_wp (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3. Default is 0.0.
    theta_star (float, optional): Critical soil moisture content, equal to s_star * porosity, in m3/m3. Default is 1.0.

    Returns:
    float: Rate of change in soil moisture (dtheta/dt) for the given soil mositure content, in m3/m3/day.
    """

    d_theta = -k * ((theta - theta_wp) / (theta_star - theta_wp)) ** (q)
    return d_theta


def loss_sigmoid(t, theta, theta50, k, a):
    """
    Calculate the loss function (dtheta/dt vs theta relationship) using sigmoid model

    Parameters:
    t (int): Timestep, in day.
    theta (float): Volumetric soil moisture content, in m3/m3.
    theta50 (float, optional): 50 percentile soil moisture content, equal to s50 * porosity, in m3/m3
    k (float): Degree of non-linearity in the soil moisture response. k = k0 (original coefficient of sigmoid) / n (porosity), in m3/m3
    a (float): The spremum of dtheta/dt, a [-/day] = ETmax [mm/day] / z [mm]
    Returns:
    float: Rate of change in soil moisture (dtheta/dt) for the given soil mositure content, in m3/m3/day.
    """
    d_theta = -1 * a / (1 + np.exp(-k * (theta - theta50)))
    return d_theta


def loss_sigmoid2(theta, theta50, k, Emax, theta_wp=0.0):
    """
    Same fucntion as loss_sigmoid() but without t as an argument
    """
    d_theta = -1 * (Emax + theta_wp) / (1 + np.exp(-k * (theta - theta50)))
    return d_theta
