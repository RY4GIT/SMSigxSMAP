import numpy as np


def tau_exp_model(t, delta_theta, theta_w, tau):
    """
    Calculate the drydown curve for soil moisture over time using linear loss function model.
    Analytical solution of the linear loss function is exponential function, with the time decaying factor tau

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


def exp_model(t, ETmax, theta_0, theta_star, theta_w, z=50.0, t_star=0.0):
    """Calculate the drydown curve for soil moisture over time using linear loss function model.
    The above tau_exp_model can be better constrained using the loss function variables, rather than tau models.

    Parameters:
        t (int): Timestep, in day.
        delta_theta (float): Shift/increment in soil moisture after precipitation, in m3/m3.
        theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3. Default is 0.0.
        tau (float): decay rate, in 1/day.

    Returns:
        float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.

    """

    tau = z * (theta_star - theta_w) / ETmax

    if theta_0 > theta_star:
        theta_0_ii = theta_star
    else:
        theta_0_ii = theta_0

    return (theta_0_ii - theta_w) * np.exp(-(t - t_star) / tau) + theta_w


def q_model(t, q, ETmax, theta_0, theta_star, theta_w, z=50.0, t_star=0.0):
    """
    Calculate the drydown curve for soil moisture over time using non-linear plant stress model.

    Parameters:
        t (int): Timestep, in day.
        z (float): Soil thicness in mm. Default is 50 mm
        ETmax (float): Maximum evapotranpisration rate in mm/day.
        q (float): Degree of non-linearity in the soil moisture response.
        theta_0 (float): The initial soil moisture after precipitation, in m3/m3
        theta_star (float, optional): Critical soil moisture content, equal to s_star * porosity, in m3/m3
        theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3

    Returns:
        float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.
    """
    if theta_0 > theta_star:
        theta_0_ii = theta_star
    else:
        theta_0_ii = theta_0

    k = (
        ETmax / z
    )  # Constant term. Convert ETmax to maximum dtheta/dt rate from a unit volume of soil

    b = (theta_0_ii - theta_w) ** (1 - q)

    a = (1 - q) / ((theta_star - theta_w) ** q)

    return (-k * a * (t - t_star) + b) ** (1 / (1 - q)) + theta_w


def drydown_piecewise(t, model, ETmax, theta_0, theta_star, z=50.0):
    """ "
    Calculate the drydown assuming that both Stage I and II are happening. Estimate theta_star
    """

    k = (
        ETmax / z
    )  # Constant term. Convert ETmax to maximum dtheta/dt rate from a unit volume of soil

    t_star = (
        theta_0 - theta_star
    ) / k  # Time it takes from theta_0 to theta_star (Stage II ET)

    return np.where(t_star > t, -k * t + theta_0, model)


def q_model_piecewise(t, q, ETmax, theta_0, theta_star, theta_w, z=50.0):

    k = (
        ETmax / z
    )  # Constant term. Convert ETmax to maximum dtheta/dt rate from a unit volume of soil

    t_star = (theta_0 - theta_star) / k  # Time it takes from theta_0 to theta_star

    return np.where(
        t_star > t,
        -k * t + theta_0,
        q_model(
            t, q, ETmax, theta_0, theta_star, theta_w, t_star=np.maximum(t_star, 0)
        ),
    )


def exp_model_piecewise(t, ETmax, theta_0, theta_star, theta_w, z=50.0):
    k = ETmax / z
    t_star = (theta_0 - theta_star) / k
    return np.where(
        t_star > t,
        -k * t + theta_0,
        exp_model(t, ETmax, theta_0, theta_star, theta_w, t_star=np.maximum(t_star, 0)),
    )


def loss_model(theta, q, ETmax, theta_w=0.0, theta_star=1.0, z=50.0):
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
    k = ETmax / z
    d_theta_ii = -k * ((theta - theta_w) / (theta_star - theta_w)) ** (q)
    d_theta_i = -k
    return np.where(theta > theta_star, d_theta_i, d_theta_ii)


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
    exp_arg = np.clip(
        -k * (theta - theta50), -np.inf, 10000
    )  # Clip exponent item to avoid failure
    d_theta = -1 * a / (1 + np.exp(exp_arg))
    return d_theta


# Function to solve the DE with given parameters and return y at the time points
def solve_de(t_obs, y_init, parameters):
    """
    The sigmoid loss function is a differential equation of dy/dt = f(y, a, b), which cannot be analytically solved,
    so the fitting of this model to drydown is numerically impelmented.
    solve_ivp finds y(t) approximately satisfying the differential equations, given an initial value y(t0)=y0.

    Parameters:
    t_obs (int): Timestep, in day.
    y_init (float): Observed volumetric soil moisture content, in m3/m3.
    parameters: a list of the follwing parameters
        theta50 (float, optional): 50 percentile soil moisture content, equal to s50 * porosity, in m3/m3
        k (float): Degree of non-linearity in the soil moisture response. k = k0 (original coefficient of sigmoid) / n (porosity), in m3/m3
        a (float): The spremum of dtheta/dt, a [-/day] = ETmax [mm/day] / z [mm]
    """
    theta50, k, a = parameters
    sol = solve_ivp(
        lambda t, theta: loss_sigmoid(t, theta, theta50, k, a),
        [t_obs[0], t_obs[-1]],
        [y_init],
        t_eval=t_obs,
        vectorized=True,
    )
    return sol.y.ravel()
