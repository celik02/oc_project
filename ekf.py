import filterpy
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from carla_setup import CarlaManager
state_dim = 5  # State dimension
measurement_dim = 2  # Measurement dimension
control_dim = 2  # Control dimension


def vehicle_dynamics(x, dt, u):
    """
    # TODO we should standardize the vehicle dynamics model across all files
    Bicycle kinematic model with longitudinal dynamics

    States:
        x[0]: x position
        x[1]: y position
        x[2]: heading angle (psi)
        x[3]: velocity
        x[4]: acceleration

    Controls:
        u[0]: acceleration command
        u[1]: steering angle
    """
    # Vehicle parameters
    L = 3  # Wheelbase
    tau_a = 0.5  # Acceleration time constant
    a = u[0]  # Acceleration command
    delta = u[1]  # Steering angle
    # State update
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + x[3] * np.cos(x[2]) * dt
    x_next[1] = x[1] + x[3] * np.sin(x[2]) * dt
    x_next[2] = x[2] + x[3] * np.tan(delta) / L * dt
    x_next[3] = x[3] + x[4] * dt
    x_next[4] = x[4] + (a - x[4]) / tau_a * dt

    return x_next


def F_jacobian(x, dt, u):
    """
    Jacobian of the vehicle dynamics model
    """
    L = 3  # Wheelbase
    tau_a = 0.5  # Acceleration time constant
    a = u[0]  # Acceleration command
    delta = u[1]  # Steering angle
    p_x, p_y, psi, v, a_curr = x
    # State Jacobian
    F = np.zeros((5, 5))

    F[0, 0] = 1
    F[0, 2] = -v * np.sin(psi) * dt
    F[0, 3] = np.cos(psi) * dt

    F[1, 1] = 1
    F[1, 2] = v * np.cos(psi) * dt
    F[1, 3] = np.sin(psi) * dt

    F[2, 2] = 1
    F[2, 3] = (np.tan(delta) / L) * dt

    F[3, 3] = 1
    F[3, 4] = dt

    F[4, 4] = 1 - dt / tau_a

    return F


def hx(x):
    """
    Measurement function that maps the state to the measurement space.
    Here, we assume the GPS provides position measurements [x, y].
    """
    return np.array([x[0], x[1]])


def H_jacobian(x):
    """
    Jacobian of the measurement function with respect to the state.
    Since our measurement is simply the position, the Jacobian is:

    H = [1  0  0  0
        0  1  0  0]
    """
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H


def set_extended_kalman_filter(x0, P0, Q, R):
    """
    Initialize the Extended Kalman Filter with the given parameters.
    """
    ekf = ExtendedKalmanFilter(dim_x=state_dim, dim_z=measurement_dim, dim_u=control_dim)
    ekf.x = x0  # Initial state
    ekf.P = P0  # Initial covariance
    ekf.Q = Q  # Process noise covariance
    ekf.R = R  # Measurement noise covariance
    ekf.F = F_jacobian  # State transition function
    ekf.H = H_jacobian  # Measurement function
    ekf.hx = hx  # Measurement function
    return ekf


x0 = np.array([0, 0, 0, 0, 0])  # Initial state
P0 = np.array([10., 10., 10., 10, 10.])  # Initial covariance
Q = np.eye(state_dim) * 0.1  # Process noise covariance
R = np.eye(measurement_dim) * 1  # Measurement noise covariance


if __name__ == "__main__":
    ekf = set_extended_kalman_filter(x0, P0, Q, R)
    print("Initial state:", ekf.x)
    print("Initial covariance:", ekf.P)
    print("Process noise covariance:", ekf.Q)
    print("Measurement noise covariance:", ekf.R)
    carla_manager = CarlaManager()
    vehicle = carla_manager.spawn_vehicle("vehicle.tesla.model3", [0, 0, 0])
    vehicle.set_autopilot(False)
    carla_manager.set_vehicle_control(vehicle, throttle=0.5, steer=0.1)

    # Simulate vehicle dynamics
    dt = 0.1  # Time step
    u = np.array([0.5, 0.1])  # Control input (acceleration, steering angle)
    
    for _ in range(10):
        # Simulate vehicle dynamics
        ekf.predict(u=u, dt=dt)
        print("Predicted state:", ekf.x)
        print("Predicted covariance:", ekf.P)

        # Simulate GPS measurement
        z = np.array([ekf.x[0] + np.random.normal(0, 1), ekf.x[1] + np.random.normal(0, 1)])
        ekf.update(z)
        print("Updated state:", ekf.x)
        print("Updated covariance:", ekf.P)
