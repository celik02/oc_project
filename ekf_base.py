from filterpy.kalman import ExtendedKalmanFilter
from numpy import dot
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CustomEKF_NEW(ExtendedKalmanFilter):
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
        logger.debug('inside predict:%s', u)
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
        Bicycle kinematic model with longitudinal dynamics

        States:
            x[0]: x position
            x[1]: y position
            x[2]: heading angle (psi)
            x[3]: velocity

        Controls:
            u[0]: acceleration command
            u[1]: steering angle
        """
        # Vehicle parameters
        L = 3  # Wheelbase # FIXME
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
        x_next[3] = x[3] + a * dt
        return x_next

    @staticmethod
    def F_jacobian(x, dt, u, imu_prediction=False):
        """
        Jacobian of the vehicle dynamics model
        """
        L = 2.89  # Wheelbase
        # a = u[0]  # Acceleration command
        delta = u[1]  # Steering angle
        p_x, p_y, psi, v = x

        # State Jacobian
        F = np.zeros((4, 4))

        F[0, 0] = 1
        F[0, 2] = -v * np.sin(psi) * dt
        F[0, 3] = np.cos(psi) * dt

        F[1, 1] = 1
        F[1, 2] = v * np.cos(psi) * dt
        F[1, 3] = np.sin(psi) * dt

        F[2, 2] = 1

        # if imu_prediction:
        #     # If using IMU prediction, we can skip the heading update from control input
        #     F[2, 3] = 0
        # else:
        F[2, 3] = (np.tan(delta) / L) * dt

        F[3, 3] = 1

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

        H = [1  0  0  0
            0  1  0  0 ]
        """
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    # # measurement jacobian for IMU sensor
    # @staticmethod
    # def HJacobian_IMU(x):
    #     """ for IMU measurement
    #     Jacobian of the measurement function with respect to the state.
    #     Since our measurement is simply the position, the Jacobian is:

    #     H = [1  0  0  0
    #         0  1  0  0]
    #     """
    #     # TODO inspect the IMU measurement
    #     H = np.zeros((2, 5))
    #     H[0, 3] = x[3] * np.cos(x[2])
    #     H[1, 4] = x[4] * np.sin(x[2])
    #     return H


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
        logger.debug('inside predict:%s', u)
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
        # logger.debug('xnext: %s', x_next)
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
