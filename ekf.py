import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from carla_setup import CarlaManager
from vehicle import Vehicle
import copy
import time
import threading
from numpy import dot
import logging
import sys
# global logger settings
FORMAT = "[%(filename)26s:%(lineno)3s - %(funcName)27s() ] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True, format=FORMAT)

# local logger settings
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

state_dim = 5  # State dimension
measurement_dim = 2  # Measurement dimension
control_dim = 2  # Control dimension


class CustomEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dim_u):
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.dim_u = dim_u  # Control dimension
        self.u = None  # Control input

    def predict_x(self, u=0,  dt=0.1):
        if u is not None:
            self.u = u
        # F updated with the current state estimate and control input
        logger.debug('inside custom predict_x with u:%s', u)
        self.F = self.F_jacobian(self.x, dt, self.u)
        self.x = self.vehicle_dynamics(self.x, dt, self.u)  # state prediction
        logger.debug('Predicting state:%s', self.x)

    def predict_x_imu(self, imu_data=None, dt=0.01):
        """
        Predict the state using IMU data.
        :param imu_data: IMU data to use for prediction
        :param dt: Time step for prediction
        """
        if imu_data is None:
            return
        # IMU reference frame aligned with the vehicle's coordinate system
        # I need ax for forward acceleration,
        # gz for angular rate around z-axis (yaw rate)
        ax, ay, az, gx, gy, gz = imu_data
        u = [ax, gz]
        # Use the IMU data to predict the state
        self.F = self.F_jacobian(self.x, dt, u=u, imu_prediction=True)
        self.x = self.vehicle_dynamics(self.x, dt, u=u, imu_prediction=True)  # state prediction

        # self.P = dot(self.F, self.P).dot(self.F.T) + self.Q  # this is prior
        logger.debug('Predicting with IMU data:%s', self.x)

    def predict(self, u=None):
        self.predict_x(u)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q  # this is prior

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, residual=np.subtract):
        logger.debug('inside update:%s', z)
        super().update(z, HJacobian=self.HJacobian_GPS, Hx=self.Hx, R=R,
                       residual=residual)
        self.u = None  # Reset control input after update

    def vehicle_dynamics(self, x, dt, u, imu_prediction=False):
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
        if imu_prediction:
            # If using IMU prediction, we can skip the heading update from control input
            # This is a placeholder for IMU prediction
            x_next[2] = x[2] + u[1] * dt
        else:
            x_next[2] = x[2] + x[3] * np.tan(delta) / L * dt
        x_next[3] = x[3] + x[4] * dt
        x_next[4] = x[4] + (a - x[4]) / tau_a * dt
        logger.debug('xnext: %s', x_next)
        return x_next

    @staticmethod
    def F_jacobian(x, dt, u, imu_prediction=False):
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

        if imu_prediction:
            # If using IMU prediction, we can skip the heading update from control input
            F[2, 3] = 0
        else:
            F[2, 3] = (np.tan(delta) / L) * dt

        F[3, 3] = 1
        F[3, 4] = dt

        F[4, 4] = 1 - dt / tau_a

        return F

    @staticmethod
    def Hx(x):
        """
        Measurement function that maps the state to the measurement space.
        Here, we assume the GPS provides position measurements [x, y].
        """
        return np.array([x[0], x[1]])

    @staticmethod
    def HJacobian_GPS(x):
        """ for GPS measurement
        Jacobian of the measurement function with respect to the state.
        Since our measurement is simply the position, the Jacobian is:

        H = [1  0  0  0 0
            0  1  0  0 0]
        """
        H = np.zeros((2, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    # # measurement jacobian for IMU sensor
    # @staticmethod
    # def HJacobian_IMU(x):
    #     """ for IMU measurement
    #     Jacobian of the measurement function with respect to the state.
    #     Since our measurement is simply the position, the Jacobian is:

    #     H = [1  0  0  0 0
    #         0  1  0  0 0]
    #     """
    #     # TODO inspect the IMU measurement
    #     H = np.zeros((2, 5))
    #     H[0, 3] = x[3] * np.cos(x[2])
    #     H[1, 4] = x[4] * np.sin(x[2])
    #     return H


def set_extended_kalman_filter(x0, P0, Q, R):
    """
    Initialize the Extended Kalman Filter with the given parameters.
    """
    ekf = CustomEKF(dim_x=state_dim, dim_z=measurement_dim, dim_u=control_dim)
    ekf.x = x0  # Initial state
    ekf.P = P0  # Initial covariance
    ekf.Q = Q  # Process noise covariance
    ekf.R = R  # Measurement noise covariance

    # ekf.F_jacobian = lambda x, dt, u: F_jacobian(x, dt, u)

    # ekf.H_jacobian = HJacobian  # Measurement function
    # ekf.hx = hx  # Measurement function
    # ekf.B = None
    return ekf


def ekf_update_loop(vehicle):
    """
    A loop to continuously update the EKF for a vehicle.
    :param vehicle: The vehicle object containing the EKF instance.
    """
    while True:
        update_interval = 0.1  # 20 Hz
        while True:
            vehicle.update_ekf()
            time.sleep(update_interval)


x0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Initial state
P0 = np.array([100., 100., 1., 1, 1.])  # Initial covariance
Q = np.eye(state_dim) * 0.5  # Process noise covariance
R = np.eye(measurement_dim) * 3  # Measurement noise covariance

SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera


if __name__ == "__main__":
    ekf: CustomEKF = set_extended_kalman_filter(x0, P0, Q, R)
    print("Initial state:", ekf.x)
    print("Initial covariance:", ekf.P)
    print("Process noise covariance:", ekf.Q)
    print("Measurement noise covariance:", ekf.R)

    carla_manager = CarlaManager()
    ego, imu, gps = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION, GPS=True, IMU=True)
    time.sleep(2)  # Allow some time for the vehicle to initialize
    ego.set_autopilot(True)
    ego_vehicle = Vehicle(ego, copy.deepcopy(ekf))
    ego_vehicle.attach_sensors(GPS=gps, IMU=imu)

    # Simulate vehicle dynamics
    dt = 0.1  # Time step
    # u = np.array([0, 0.0])  # Control input (acceleration, steering angle)

    # plot the results
    import matplotlib.pyplot as plt
    plt.ion()  # Turn on interactive mode
    plt.figure()
    plt.title("EKF State Estimation")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid()
    # print(ekf)
    ekf_thread = threading.Thread(target=ekf_update_loop, args=(ego_vehicle,), daemon=True)
    ekf_thread.start()
    while True:
        # Simulate vehicle dynamics
        # ekf.predict(u=u)

        # Simulate GPS measurement
        # z = np.array([ekf.x[0] + np.random.normal(0, 1) + u[0], ekf.x[1] + np.random.normal(0, 1) + u[1]])
        # ekf.update(z)
        control = ego_vehicle.actor.get_control()
        print(control)  # Ensure the actor is in control mode
        u = np.array([1.5, 0.5])
        # Plot the current state
        ekf_state_x = ego_vehicle.get_latest_state()
        current_location = ego_vehicle.actor.get_location()
        current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)
        next_waypoint = current_wp.next(2)[0]
        print('frenet states:', ego_vehicle.get_frenet_states(next_waypoint))

        plt.scatter(ekf_state_x[0], ekf_state_x[1], color='r', marker='x', label='EKF Estimate')  # Plot EKF estimated position
        plt.scatter(ego_vehicle.actor.get_transform().location.x,
                    ego_vehicle.actor.get_transform().location.y,
                    color='b', label='Actual Position')

        # plot gps location
        plt.scatter(ego_vehicle.carla_coords[0], ego_vehicle.carla_coords[1], color='g', label='GPS Location')

        ego_vehicle.ekf.predict(u=u)
        # ego_vehicle.update_ekf()

        # update spectator camera to follow the vehicle
        spectator_location = ego_vehicle.actor.get_transform()
        spectator_location.location.z += 5
        spectator_location.location.x += 5
        carla_manager.spectator.set_transform(spectator_location)  # Keep the spectator camera in the same location
        # update the world
        # add some noise to vehicle control
        ego_control = ego_vehicle.actor.get_control()
        ego_control.throttle += np.random.normal(0, 0.5)
        ego_control.steer += np.random.normal(0, 0.1)
        ego_vehicle.actor.apply_control(ego_control)
        carla_manager.world.tick()
        plt.pause(0.1)
    ekf_thread.join()  # Ensure the EKF update loop thread is joined before exiting
    carla_manager.__del__()  # Clean up the Carla manager and close the simulation
