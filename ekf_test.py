import numpy as np
from carla_setup import CarlaManager
from vehicle import Vehicle
import copy
import time
import threading
import logging
import sys
import matplotlib.pyplot as plt

# global logger settings
# inlcude timestamp in log messages
FORMAT = "[%(asctime)s.%(msecs)03d %(filename)15s:%(lineno)3s - %(funcName)17s() ] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True, format=FORMAT, datefmt='%H:%M:%S')

# local logger settings
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

state_dim = 4  # State dimension
measurement_dim = 2  # Measurement dimension
control_dim = 2  # Control dimension


# def set_extended_kalman_filter(x0, P0, Q, R):
#     """
#     Initialize the Extended Kalman Filter with the given parameters.
#     """
#     ekf = CustomEKF(dim_x=state_dim, dim_z=measurement_dim, dim_u=control_dim)
#     ekf.x = x0  # Initial state
#     ekf.P = P0  # Initial covariance
#     ekf.Q = Q  # Process noise covariance
#     ekf.R = R  # Measurement noise covariance

#     # ekf.F_jacobian = lambda x, dt, u: F_jacobian(x, dt, u)

#     # ekf.H_jacobian = HJacobian  # Measurement function
#     # ekf.hx = hx  # Measurement function
#     # ekf.B = None
#     return ekf


# def ekf_update_loop(vehicle: Vehicle):
#     """
#     A loop to continuously update the EKF for a vehicle.
#     :param vehicle: The vehicle object containing the EKF instance.
#     """
#     while True:
#         update_interval = 0.05  # 20 Hz
#         while True:
#             vehicle.update_ekf()
#             time.sleep(update_interval)


# P0 = np.array([1., 1., 1., 1.])  # Initial covariance
P0 = np.array([0., 0., 0., 0.])
# Q = np.eye(state_dim) * 0.1  # Process noise covariance
Q = np.diag([0.1, 0.1, 0.001, 0.5])  # Process noise covariance

R = np.eye(measurement_dim) * 3  # Measurement noise covariance

SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera


if __name__ == "__main__":
    # # ekf: CustomEKF = set_extended_kalman_filter(x0, P0, Q, R)
    # print("Initial state:", ekf.x)
    # print("Initial covariance:", ekf.P)
    # print("Process noise covariance:", ekf.Q)
    # print("Measurement noise covariance:", ekf.R)

    carla_manager = CarlaManager()
    ego, imu, gps = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION, GPS=True, IMU=True)
    time.sleep(2)  # Allow some time for the vehicle to initialize
    ego.set_autopilot(True)
    ego_vehicle = Vehicle(ego)
    ego_vehicle.attach_sensors(GPS=gps, IMU=imu)
    time.sleep(1)  # Allow some time for the sensors to initialize
    carla_manager.world.tick()  # NOTE To get the initial sensor data
    carla_manager.world.tick()  # NOTE To get the initial sensor data
    carla_manager.world.tick()  # NOTE To get the initial sensor data
    ego_vehicle.initialize_ekf(P0, Q, R)

    # Simulate vehicle dynamics
    dt = 0.1  # Time step
    # u = np.array([0, 0.0])  # Control input (acceleration, steering angle)

    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("EKF State Estimation")

    # plot the results
    ax1.set_title("Position")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.grid()

    # Yaw plot (bottom)
    ax2.set_title("Yaw Angle")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Yaw (rad)")
    ax2.grid()
    # print(ekf)
    # ekf_thread = threading.Thread(target=ekf_update_loop, args=(ego_vehicle,), daemon=True)
    # ekf_thread.start()
    i = 0
    while True:
        # Simulate vehicle dynamics
        # ekf.predict(u=u)

        # Simulate GPS measurement
        # z = np.array([ekf.x[0] + np.random.normal(0, 1) + u[0], ekf.x[1] + np.random.normal(0, 1) + u[1]])
        # ekf.update(z)
        control = ego_vehicle.actor.get_control()
        normalized_steer = control.steer  # Normalize the steering angle to [-1, 1]
        max_steer_angle = 0.5  # Maximum steering angle
        delta = normalized_steer * max_steer_angle  # Steering angle in radians
        u = np.array([control.throttle*4, 0])
        # Plot the current state
        ekf_state_x = ego_vehicle.get_latest_state()

        # # This is to verify the frenet states
        # current_location = ego_vehicle.actor.get_location()
        # current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)
        # next_waypoint = current_wp.next(2)[0]
        # print('frenet states:', ego_vehicle.get_frenet_states(next_waypoint))

        ax1.scatter(ekf_state_x[0], ekf_state_x[1], color='r', marker='x', label='EKF Estimate')  # Plot EKF estimated position
        ax1.scatter(ego_vehicle.ekf.x_prior[0], ego_vehicle.ekf.x_prior[1], color='y', marker='o', label='EKF Prior')  # Plot EKF prior position
        ax1.scatter(ego_vehicle.actor.get_transform().location.x,
                    ego_vehicle.actor.get_transform().location.y,
                    color='b', label='Actual Position')

        ax2.scatter(i, ego_vehicle.ekf.x[2], color='r', marker='x', label='EKF Estimate')  # Plot EKF estimated yaw
        ax2.scatter(i, ego_vehicle.ekf.x_prior[2], color='y', marker='x', label='EKF prediction')  # Plot EKF estimated yaw

        i += 1
        # plot gps location
        ax1.scatter(ego_vehicle.carla_coords[0], ego_vehicle.carla_coords[1], color='g', label='GPS Location')
        ego_vehicle.ekf.predict(u=u)
        ego_vehicle.update_ekf()

        # update spectator camera to follow the vehicle
        spectator_location = ego_vehicle.actor.get_transform()
        spectator_location.location.z += 5
        spectator_location.location.x += 5
        carla_manager.spectator.set_transform(spectator_location)  # Keep the spectator camera in the same location

        # update the world
        # add some noise to vehicle control
        carla_manager.world.tick()
        print(ego_vehicle.ekf)
        plt.pause(.1)
        print(ego_vehicle.ekf)
        print('Ground Truth:', ego_vehicle.actor.get_transform().location)
        print('GT yaw:', ego_vehicle.actor.get_transform().rotation.yaw, np.deg2rad(ego_vehicle.actor.get_transform().rotation.yaw))
        print('vehicle velocity:', ego_vehicle.actor.get_velocity())

        # TODO exit kalman filter when 100 meter reached
        # TODO plot yaw angle and speed too.
        # input('enter to continue...')
    carla_manager.__del__()  # Clean up the Carla manager and close the simulation
