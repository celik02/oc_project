from ekf_base import CustomEKF_NEW
import numpy as np
import threading
import queue
import copy
import traceback
from scipy.optimize import least_squares
import carla
import math
import logging
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

state_dim = 4  # State dimension
measurement_dim = 2  # Measurement dimension
control_dim = 2  # Control dimension


class Vehicle:
    """
    Vehicle class to represent a vehicle in the simulation.
    """

    def __init__(self, actor):
        """
        Initialize the vehicle with a Carla actor.
        :param actor: The Carla actor representing the vehicle.
        """
        self.actor = actor
        self.ekf = None
        self.imu_data_queue = queue.Queue(maxsize=10)
        self.gps_data_queue = queue.Queue(maxsize=5)
        self.state_lock = threading.Lock()
        self.latest_state = np.zeros(5)  # Initialize latest state to zero, will be updated by EKF]
        self.wheel_base = 2.5  # Example wheel base, adjust as necessary  # FIXME

        # initialize states
        self.carla_coords = None  # Placeholder for GPS coordinates
        self.imu_data_array = None  # Placeholder for IMU data
        self.s = 0
        self.prev_loc = self.actor.get_transform().location  # FIXME this may not be correct given the s values

    def attach_sensors(self, GPS=None, IMU=None):
        """
        Attach sensors to the vehicle.
        :param GPS: Optional GPS sensor to attach.
        :param IMU: Optional IMU sensor to attach.
        """
        self.gps = GPS
        self.imu = IMU

        self.gps.listen(self._gps_callback) if GPS else None
        self.imu.listen(self._imu_callback) if IMU else None
        logger.debug("Sensors attached to vehicle GPS:%s IMU:%s", GPS, IMU)

    def initialize_ekf(self, P0, Q, R):
        """
        Initialize the Extended Kalman Filter (EKF) with an initial state.
        :param initial_state: The initial state to set for the EKF.
        """
        self.ekf = CustomEKF_NEW(dim_x=state_dim, dim_z=measurement_dim, dim_u=control_dim)
        self.ekf.x = np.zeros(state_dim)  # Example initial state
        self.ekf.P = P0  # Initial covariance
        self.ekf.Q = Q  # Process noise covariance
        self.ekf.R = R  # Measurement noise covariance
        gps_data = None
        while True:
            try:
                gps_data = self.gps_data_queue.get_nowait()
            except queue.Empty:
                gps_data = None
                print("GPS queue empty")
            if gps_data is not None:
                print(f"GPS data received: {gps_data}")
                self.ekf.x[0] = self.carla_coords[0]
                self.ekf.x[1] = self.carla_coords[1]
                logger.debug("EKF initialized with GPS data %s", self.ekf.x)

                break
            else:
                time.sleep(0.05)
                logger.debug("Waiting for GPS data to initialize EKF")

        yaw = self.actor.get_transform().rotation.yaw
        self.ekf.x[2] = np.radians(yaw) % (2 * np.pi)

    def _gps_callback(self, data):
        """
        Callback function for GPS data.
        :param data: The GPS data received from the sensor.
        """
        logger.debug("Received GPS data callback")  # Debug statement to confirm callback is called
        self.gps_data = data
        print('data timestamp', self.gps_data.timestamp)
        self.gps_arr = copy.copy(np.array([self.gps_data.latitude, self.gps_data.longitude, self.gps_data.altitude]))
        (carla_x, carla_y, carla_z) = self._convert_gps_to_carla(self.gps_arr)
        self.carla_coords = (carla_x, carla_y, carla_z)
        try:
            self.gps_data_queue.put_nowait(copy.deepcopy(self.carla_coords))
        except queue.Full:
            # print("GPS queue full, dropping data.")
            pass

        # print('Vehicle actual position:', self.actor.get_transform().location)

    def _convert_gps_to_carla(self, gps):
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

    def _imu_callback(self, data):
        """
        Callback function for IMU data.
        :param data: The IMU data received from the sensor.
        """
        # TODO if necessary data could be put into thread safe queue
        # Extract IMU data
        self.imu_data = data
        self.imu_data_array = np.array([-self.imu_data.accelerometer.x, self.imu_data.accelerometer.y, self.imu_data.accelerometer.z,
                                        self.imu_data.gyroscope.x, self.imu_data.gyroscope.y, self.imu_data.gyroscope.z])
        logger.debug("Received IMU data callback %s ", self.imu_data_array)
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

    def apply_control_accel(self, accel, steer):
        """
        Apply control to the vehicle.
        :param throttle: The throttle value to apply.
        :param steer: The steering value to apply.
        """
        throttle, brake = throttle_brake_mapping1(accel)
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
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

        try:
            gps_data = self.gps_data_queue.get_nowait()
            z = np.array([gps_data[0], gps_data[1]])
            self.ekf.update(z)
            logger.debug(f"Updating EKF with GPS data: {gps_data}")
        except queue.Empty:
            logger.debug("No GPS data available for EKF update.")
            gps_data = None
        except Exception as e:
            logger.error('%s', e)
            traceback.print_exc()

        with self.state_lock:
            self.latest_state = self.ekf.x.copy()

    def get_latest_state(self):
        ''' returns the latest state of the vehicle from the EKF'''
        with self.state_lock:
            return self.latest_state

    def get_frenet_states(self, next_waypoint):
        """
        Get the Frenet states of the vehicle.
        :return: The Frenet coordinates as a numpy array [s, d, mu, speed, steering_angle, curvature]
        s: Progress along the lane.
        d: Deviation from the lane center.
        mu: Heading error.
        speed: Speed of the vehicle.
        steering_angle: Steering angle of the vehicle.
        curvature: Curvature of the path. [not IMPLEMENTED since we are going straight]
        """
        velocity = self.actor.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # speed in m/s

        # Get vehicle control (steering)
        control = self.actor.get_control()
        steering_angle = np.clip(control.steer, -1, 1)
        max_steering_angle = np.radians(69.99999237060547)
        steering_angle = steering_angle * max_steering_angle

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

        self.s += self._calculate_progress()

        d = self._calculate_deviation(self.waypoint)
        state_data = np.array([self.s, d, mu, speed, steering_angle, curvature])

        return state_data, self.waypoint

    def _calculate_deviation(self, waypoint):
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

    def _calculate_progress(self):
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

    # Add these methods to your Vehicle class
    def create_ego_coordinate_system(self, use_current_transform=False):
        """
        Create a coordinate system with origin at the ego vehicle's spawn/current location,
        with x-axis pointing forward, y-axis to the right, and z-axis up.
        
        Args:
            use_current_transform: If True, use the vehicle's current transform instead of spawn transform
            
        Returns:
            world_to_ego_matrix: 4x4 homogeneous transformation matrix to convert world to ego coordinates
            ego_to_world_matrix: 4x4 homogeneous transformation matrix to convert ego to world coordinates
        """
        if not hasattr(self, 'transform_to_spawn'):
            self.transform_to_spawn = self.actor.get_transform()
        
        vehicle_transform = self.actor.get_transform() if use_current_transform else self.transform_to_spawn
        
        # Extract position from transform
        tx = vehicle_transform.location.x
        ty = vehicle_transform.location.y
        tz = vehicle_transform.location.z
        
        # Extract rotation from transform (in radians)
        yaw = math.radians(vehicle_transform.rotation.yaw)
        pitch = math.radians(vehicle_transform.rotation.pitch)
        roll = math.radians(vehicle_transform.rotation.roll)
        
        # Create rotation matrices for each axis
        # Rotation around Z-axis (yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        R_z = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Rotation around Y-axis (pitch)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        R_y = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        # Rotation around X-axis (roll)
        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        R_x = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])
        
        # Combine rotations (order: yaw, pitch, roll as per CARLA convention)
        R = R_z @ R_y @ R_x
        
        # For world to ego transformation, we need the transpose of R
        R_world_to_ego = R.T
        
        # Create homogeneous transformation matrix (world to ego)
        world_to_ego = np.eye(4)
        world_to_ego[:3, :3] = R_world_to_ego
        
        # Translation component (after rotation)
        world_to_ego[:3, 3] = -R_world_to_ego @ np.array([tx, ty, tz])
        
        # Create inverse transformation matrix (ego to world)
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = R
        ego_to_world[:3, 3] = np.array([tx, ty, tz])
        
        return world_to_ego, ego_to_world

    def world_to_ego_coordinates(self, point, use_current_transform=False):
        """
        Transform a point from world coordinates to ego vehicle coordinates.
        
        Args:
            point: A world point (can be a carla.Location or a numpy array)
            use_current_transform: If True, use the vehicle's current transform instead of spawn transform
            
        Returns:
            The point in ego vehicle coordinates (numpy array)
        """
        world_to_ego, _ = self.create_ego_coordinate_system(use_current_transform)
        
        # Convert to homogeneous coordinates
        if isinstance(point, carla.Location):
            point_homogeneous = np.array([[point.x], [point.y], [point.z], [1]])
        else:
            point_homogeneous = np.array([[point[0]], [point[1]], [point[2]], [1]])
        
        # Transform the point
        transformed_point = world_to_ego @ point_homogeneous
        
        return transformed_point[:3, 0]

    def ego_to_world_coordinates(self, point, use_current_transform=False):
        """
        Transform a point from ego vehicle coordinates to world coordinates.
        
        Args:
            point: A point in ego vehicle coordinates (numpy array)
            use_current_transform: If True, use the vehicle's current transform instead of spawn transform
            
        Returns:
            The point in world coordinates (numpy array)
        """
        _, ego_to_world = self.create_ego_coordinate_system(use_current_transform)
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.array([[point[0]], [point[1]], [point[2]], [1]])
        
        # Transform the point
        transformed_point = ego_to_world @ point_homogeneous
        
        return transformed_point[:3, 0]

    def get_transform_matrices(self, use_current_transform=False):
        """
        Get both transformation matrices (world to ego and ego to world).
        
        Args:
            use_current_transform: If True, use the vehicle's current transform instead of spawn transform
            
        Returns:
            world_to_ego_matrix: 4x4 homogeneous transformation matrix
            ego_to_world_matrix: 4x4 homogeneous transformation matrix
        """
        return self.create_ego_coordinate_system(use_current_transform)
    
    def get_vehicle_state(self, use_current_transform=False):
        """
        Get the full bicycle model state vector [x, y, psi, v, a] for the vehicle.
        
        Returns:
            State vector: numpy array [x, y, psi, v, a]
        """
        # Get the vehicle transform in world coordinates
        vehicle_transform = self.actor.get_transform()
        
        # Get position
        x = vehicle_transform.location.x
        y = vehicle_transform.location.y
        # traform to ego coordinates if necessary
        x, y, _ = self.world_to_ego_coordinates((x, y, vehicle_transform.location.z), use_current_transform)

        # Get orientation (psi)
        if use_current_transform:
            # When using current transform, psi is zero in ego coordinates
            psi = 0.0
        else:
            # When using spawn transform as reference, we need to calculate relative yaw
            current_yaw = math.radians(vehicle_transform.rotation.yaw)
            spawn_yaw = math.radians(self.transform_to_spawn.rotation.yaw)
            psi = (current_yaw - spawn_yaw) % (2 * math.pi)
            
            # Normalize to [-pi, pi]
            if psi > math.pi:
                psi -= 2 * math.pi
        
        # Get velocity
        velocity = self.actor.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Estimate acceleration from control inputs
        control = self.actor.get_control()
        if control.throttle > 0:
            accel = control.throttle * 3.0  # Approximate based on throttle
        else:
            accel = -control.brake * 3.0    # Approximate based on brake
        
        return np.array([x, y, psi, speed, accel])
    
    def get_vehicle_state_kalman(self, use_current_transform=False):
        """
        Get the vehicle state from the Kalman filter.
        
        Returns:
            State vector: numpy array [x, y, psi, v, a]
        """
        

def throttle_brake_mapping1(a):
    if a >= 0:
        throttle = a / 4
        brake = 0
    else:
        throttle = 0
        brake = a / 6

    return throttle, brake


# if we need curvature, we will need this function
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
