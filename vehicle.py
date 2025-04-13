# from ekf import CustomEKF
import numpy as np
import threading
import queue
import copy
import traceback
from scipy.optimize import least_squares
import carla
import math
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Vehicle:
    """
    Vehicle class to represent a vehicle in the simulation.
    """

    def __init__(self, actor, ekf=None):
        """
        Initialize the vehicle with a Carla actor.
        :param actor: The Carla actor representing the vehicle.
        """
        self.actor = actor
        self.ekf = ekf
        self.imu_data_queue = queue.Queue(maxsize=10)
        self.gps_data_queue = queue.Queue(maxsize=5)
        self.state_lock = threading.Lock()
        self.latest_state = np.zeros(5)  # Initialize latest state to zero, will be updated by EKF]
        self.wheel_base = 2.5  # Example wheel base, adjust as necessary  # FIXME
        # TODO set initial state of EKF
        # self.ekf.x = np.array([0, 0, 0, 0, 0])  # Example initial state
        self.carla_coords = np.array([0.0, 0.0, 0.0])  # Placeholder for GPS coordinates

        # initialize states
        self.s = 0
        self.prev_loc = self.actor.get_transform().location

    def attach_sensors(self, GPS=None, IMU=None):
        """
        Attach sensors to the vehicle.
        :param GPS: Optional GPS sensor to attach.
        :param IMU: Optional IMU sensor to attach.
        """
        self.gps = GPS
        self.imu = IMU

        self.gps.listen(self.gps_callback) if GPS else None
        self.imu.listen(self.imu_callback) if IMU else None

    def gps_callback(self, data):
        """
        Callback function for GPS data.
        :param data: The GPS data received from the sensor.
        """
        print("Received GPS data callback")  # Debug statement to confirm callback is called
        self.gps_data = data
        self.gps_arr = copy.copy(np.array([self.gps_data.latitude, self.gps_data.longitude, self.gps_data.altitude]))
        (carla_x, carla_y, carla_z) = self.convert_gps_to_carla(self.gps_arr)
        self.carla_coords = (carla_x, carla_y, carla_z)
        try:
            self.gps_data_queue.put_nowait(copy.deepcopy(self.carla_coords))
        except queue.Full:
            # print("GPS queue full, dropping data.")
            pass

        # print('Vehicle actual position:', self.actor.get_transform().location)

    def convert_gps_to_carla(self, gps):
        """
        Converts GPS signal into the CARLA coordinate frame
        :param gps: gps from gnss sensor
        :return: gps as numpy array in CARLA coordinates
        """
        mean = np.array([0, 0, 0])
        scale = np.array([111319.49082349832, 111319.49079327358, 1.0])
        gps = (gps - mean) * scale
        # GPS uses a different coordinate system than CARLA.
        # This converts from GPS -> CARLA (90° rotation)
        gps = np.array([gps[1], -gps[0], gps[2]])
        return gps

    def imu_callback(self, data):
        """
        Callback function for IMU data.
        :param data: The IMU data received from the sensor.
        """
        # TODO if necessary data could be put into thread safe queue
        # Extract IMU data
        self.imu_data = data
        self.imu_data_array = np.array([-self.imu_data.accelerometer.x, self.imu_data.accelerometer.y, self.imu_data.accelerometer.z,
                                        self.imu_data.gyroscope.x, self.imu_data.gyroscope.y, self.imu_data.gyroscope.z])
        # print(f"IMU data received: {self.imu_data_array}")
        try:
            self.imu_data_queue.put_nowait(copy.deepcopy(self.imu_data_array))
        except queue.Full:
            # print("IMU queue full, dropping data.")
            pass

    def apply_control(self, control):
        """
        Apply control to the vehicle.
        :param control: The control to apply (e.g., throttle, brake, steer).
        """
        self.actor.apply_control(control)

    def update_ekf(self):
        """
        Update the Extended Kalman Filter (EKF) if it is set.
        This function can be called to update the EKF with the latest sensor data.
        """
        # try:
        #     imu_data = self.imu_data_queue.get_nowait()
        #     print(f"Retrieved IMU data for EKF update: {imu_data}")
        #     self.ekf.predict_x_imu(imu_data)  # FIXME # Perform the prediction step before update
        # except queue.Empty:
        #     print("No IMU data available for EKF update.")
        #     imu_data = None
        # except Exception as e:
        #     print('ERROR', e)
        #     traceback.print_exc()

        # try:
        #     gps_data = self.gps_data_queue.get_nowait()
        #     z = np.array([gps_data[0], gps_data[1]])
        #     self.ekf.update(z)
        #     print(f"Updating EKF with GPS data: {gps_data}")
        # except queue.Empty:
        #     print("No GPS data available for EKF update.")
        #     gps_data = None
        # except Exception as e:
        #     print('ERROR', e)
        #     traceback.print_exc()

        with self.state_lock:
            self.latest_state = self.ekf.x.copy()

    def get_latest_state(self):
        with self.state_lock:
            return self.latest_state

    def get_frenet_states(self, next_waypoint):
        """
        Get the Frenet states of the vehicle.
        :return: The Frenet coordinates as a numpy array [s, d, mu, speed, steering_angle, curvature]
        """
        velocity = self.actor.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # speed in m/s

        # Get vehicle control (steering)
        control = self.actor.get_control()
        steering_angle = np.clip(control.steer, -1, 1)

        # calculate heading error
        self.waypoint = next_waypoint

        # # Approximate curvature (kappa) based on steering angle and vehicle wheel base
        # wheel_base = 2.5  # You may need to adjust this based on the vehicle
        # curvature = steering_angle / wheel_base

        # if we need curvature, we need to calculate it --> we need next 15 or so waypoints
        # _, _, r_fit = fit_circle(self.path_to_follow_waypoints, next_waypoint)

        try:
            # self.curvature_speed = 1 / r_fit
            curvature = 0
        except FloatingPointError as e:
            curvature = 0
            print(f"Caught a floating point error: {e}")
        except Exception as e:
            print(f"Caught an exception: {e}")
            traceback.print_exc()

        self.target_heading = np.radians(self.waypoint.transform.rotation.yaw) % (
            2 * np.pi
        )

        mu = (
            np.radians(
                self.waypoint.transform.rotation.yaw
                - (
                    self.actor.get_transform().rotation.yaw
                    + np.arctan(steering_angle / self.wheel_base)
                )
            )
            + np.pi
        ) % (2 * np.pi) - np.pi

        self.s += self.calculate_progress()

        d = self.calculate_deviation(self.waypoint)
        state_data = np.array([self.s, d, mu, speed, steering_angle, curvature])

        return state_data, self.waypoint

    def calculate_deviation(self, waypoint):
        """
        Calculate the signed deviation of the vehicle from the center of the lane.
        """
        # Get the current location and heading of the vehicle
        vehicle_location = self.actor.get_location()
        logger.debug(f"Vehicle location: {vehicle_location}")
        logger.debug(f"next wp location: {waypoint.transform.location}")
        # Get the map and the waypoint at the vehicle's location

        # Get the lane center and lane direction
        lane_transform = waypoint.transform
        lane_center_location = lane_transform.location
        lane_heading = lane_transform.rotation.yaw
        lane_heading_rad = math.radians(lane_heading)

        # Calculate the direction vector of the lane
        lane_direction = carla.Vector3D(
            math.cos(lane_heading_rad), math.sin(lane_heading_rad), 0
        )

        # Calculate the relative position vector from the lane center to the vehicle
        vehicle_vector = carla.Vector3D(
            vehicle_location.x - lane_center_location.x,
            vehicle_location.y - lane_center_location.y,
            0,
        )

        # Calculate the cross product (z-component) to determine the side
        cross_product_z = (
            lane_direction.x * vehicle_vector.y - lane_direction.y * vehicle_vector.x
        )

        # # Calculate the Euclidean distance for the deviation
        # deviation = math.sqrt(vehicle_vector.x**2 + vehicle_vector.y**2)

        # # Apply the sign based on the cross product
        # if cross_product_z < 0:
        #     deviation = -deviation

        return cross_product_z

    def calculate_progress(self):
        """
        Calculate the progress 's' along the lane for a CARLA vehicle.
        Returns:
        - s (float): The progress along the lane.
        """
        # Get the vehicle's transform
        vehicle_transform = self.actor.get_transform()
        vehicle_location = vehicle_transform.location
        s = vehicle_location.distance(self.prev_loc)
        self.prev_loc = vehicle_location
        return s


# if we need curvature we will need this function
def fit_circle(path_to_follow_waypoints, next_waypoint):
    # Initial guess for parameters: center at mean of points, radius as half of maximum distance from center
    x_centerline = []
    y_centerline = []
    index = path_to_follow_waypoints.index(next_waypoint)
    waypoint = next_waypoint[0]
    # for _ in range(0, 20):
    #     x_centerline.append(waypoint.transform.location.x)
    #     y_centerline.append(waypoint.transform.location.y)
    #     waypoint = waypoint.next(.5)[0]
    for i in range(0, 15):
        if i + index >= len(path_to_follow_waypoints):
            break
        x_centerline.append(path_to_follow_waypoints[index + i][0].transform.location.x)
        y_centerline.append(path_to_follow_waypoints[index + i][0].transform.location.y)

    points = np.column_stack((x_centerline, y_centerline))
    xc_initial = np.mean(points[:, 0])
    yc_initial = np.mean(points[:, 1])
    r_initial = (
        np.max(
            np.sqrt((points[:, 0] - xc_initial) ** 2 + (points[:, 1] - yc_initial) ** 2)
        )
        / 2
    )

    params_initial = [xc_initial, yc_initial, r_initial]

    result = least_squares(circle_residuals, params_initial, args=(points,))
    xc, yc, r = result.x

    first_point = points[0]
    tangent_vector = np.array(
        [
            waypoint.transform.rotation.get_forward_vector().x,
            waypoint.transform.rotation.get_forward_vector().y,
        ]
    )

    # Calculate vector from circle center to the first point
    center_to_point_vector = first_point - np.array([xc, yc])

    # Calculate the cross product
    cross_product = np.cross(
        np.append(tangent_vector, 0), np.append(center_to_point_vector, 0)
    )

    # If the z-component of the cross product is positive, it's a right turn; if negative, it's a left turn
    turn_direction = (
        -1 if cross_product[2] > 0 else 1
    )  # Left turn: negative curvature, Right turn: positive curvature
    return xc, yc, r * turn_direction


def circle_residuals(params, points):
    xc, yc, r = params
    return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2) - r
