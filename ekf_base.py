from filterpy.kalman import ExtendedKalmanFilter
from numpy import dot
import numpy as np
import logging
from copy import deepcopy
from numpy import eye
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CustomEKF_NEW(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dim_u):
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.dim_u = dim_u  # Control dimension
        self.u = None  # Control input
        self.prev_yaw = None  # Previous yaw angle
        self.cumulative_yaw = 0.0  # For continuous tracking

    def predict_x(self, u=0,  dt=0.01):
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
        u = [ax, -gz]
        # Use the IMU data to predict the state
        self.F = self.F_jacobian(self.x, dt, u=u, imu_prediction=True)
        self.x = self.vehicle_dynamics(self.x, dt, u=u, imu_prediction=True)  # state prediction

        logger.debug('Predicting with IMU data:%s', self.x)

    def predict(self, u=None, imu_prediction=False, dt=0.01):
        logger.debug('inside predict:%s', u)
        if imu_prediction:
            self.predict_x_imu(u, dt)
        else:
            self.predict_x(u, dt)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q  # this is prior
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, residual=np.subtract):
        logger.debug('inside update:%s', z)

        # super().update(z, HJacobian=self.HJacobian_GPS, Hx=self.Hx, R=R,
        #                residual=residual)

        """ Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, posterior is not computed

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable. for robot localization you might need to pass in
            information about the map and time of day, so you might have
            `args=(map_data, time)`, where the signature of HCacobian will
            be `def HJacobian(x, map, t)`

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        # if not isinstance(args, tuple):
        #     args = (args,)

        # if not isinstance(hx_args, tuple):
        #     hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = self.HJacobian_GPS(self.x)

        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.K = PHT.dot(np.linalg.inv(self.S))

        hx = self.Hx(self.x)
        self.y = residual(z, hx)
        self.x = self.x + dot(self.K, self.y)

        # # correct the yaw angle
        # if self.prev_yaw_update is not None:
        #     # Calculate the difference between the current and previous yaw angles
        #     yaw_diff = self.x[2] - self.prev_yaw_update
        #     # Normalize the yaw difference to be within [-pi, pi]
        #     yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        #     # Update the previous yaw angle
        #     self.prev_yaw_update = self.x[2]
        #     self.x[2] = self.prev_yaw_update + yaw_diff
        # else:
        #     # If this is the first prediction, just set the previous yaw to the current one
        #     self.prev_yaw_update = self.x[2]
        # # Normalize the yaw angle to be within [-pi, pi]
        # self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
        # # # now wrap the yaw state (index 2) into [-π, π)
        # self.x[2] = wrap_to_pi(self.x[2])

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

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
        L = 2.89  # Wheelbase # FIXME
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
            # normalize heading
            x_next[2] = wrap_to_pi(x_next[2])

        else:
            x_next[2] = x[2] + x[3] * np.tan(delta) / L * dt
        x_next[3] = x[3] + a * dt

        return x_next

    @staticmethod
    def angle_diff(a, b):
        """Calculate the smallest angle difference"""
        return ((a - b + np.pi) % (2 * np.pi)) - np.pi

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

        if imu_prediction:
            # If using IMU prediction, we can skip the heading update from control input
            F[2, 3] = 0
        else:
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


def wrap_to_pi(angle):
    """Wrap angle to [−π, +π)."""
    return (angle + np.pi) % (2*np.pi) - np.pi


'''class CustomEKF(ExtendedKalmanFilter):
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
    #     return H'''
