import numpy as np
from pid import CarlaManager
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

# P0 = np.array([1., 1., 1., 1.])  # Initial covariance
P0 = np.array([1.5, 1.5, 1.5, 1.])

# Q = np.eye(state_dim) * 0.1  # Process noise covariance
Q = np.diag([0.1, 0.1, 0.005, 0.5])  # Process noise covariance

R = np.eye(measurement_dim) * 2  # Measurement noise covariance
# R = np.diag([1, 1])  # Measurement noise covariance

SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera


# Add this function outside the main loop
def plot_covariance_ellipse(ax, x, P, sigma=2, color='r', alpha=0.3):
    """
    Plot a covariance ellipse for a 2D state with covariance P.

    Parameters:
    - ax: Matplotlib axis object
    - x: State vector [x, y]
    - P: Covariance matrix
    - sigma: Confidence interval (1=68%, 2=95.4%, 3=99.7%)
    - color: Ellipse color
    - alpha: Ellipse transparency
    """
    print('p shape:', P.shape)
    assert P.shape == (4, 4) or P.shape == (2, 2), "Covariance matrix has incorrect dimensions"

    # Extract position covariance (2x2 submatrix)
    if P.shape == (4, 4):
        cov = np.array([
            [P[0, 0], P[0, 1]],
            [P[1, 0], P[1, 1]]
        ])
    else:
        cov = P

    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(cov)

    # Get largest eigenvalue and eigenvector
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Calculate ellipse parameters
    # 95% confidence interval for 2 dimensional data
    chisquare_val = sigma**2
    theta = np.linspace(0, 2*np.pi, 100)

    # Ellipse parameterization
    a = np.sqrt(chisquare_val * eigenvals[0])  # Major axis
    b = np.sqrt(chisquare_val * eigenvals[1])  # Minor axis

    # Generate ellipse points
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)

    # Rotate ellipse points
    R = eigenvecs
    ellipse_pts = np.array([ellipse_x, ellipse_y]).T @ R

    # Shift to center at x
    ellipse_pts[:, 0] += x[0]
    ellipse_pts[:, 1] += x[1]

    # Plot ellipse
    return ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], color=color, alpha=alpha)[0]


if __name__ == "__main__":
    # set up the carla manager and spawn the vehicle
    carla_manager = CarlaManager()
    ego, imu, gps = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION, GPS=True, IMU=True)
    time.sleep(2)  # Allow some time for the vehicle to initialize
    ego.set_autopilot(True)

    # attach sensors to the vehicle
    ego_vehicle = Vehicle(ego)
    ego_vehicle.attach_sensors(GPS=gps, IMU=imu)
    time.sleep(1)  # Allow some time for the sensors to initialize
    carla_manager.world.tick()  # NOTE To get the initial sensor data
    carla_manager.world.tick()  # NOTE To get the initial sensor data
    carla_manager.world.tick()  # NOTE To get the initial sensor data

    # Initialize the EKF
    ego_vehicle.initialize_ekf(P0, Q, R)

    # Turn on interactive mode
    plt.ion()
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
    plt.legend()

    i = 0
    initial_location = ego_vehicle.actor.get_transform().location  # to stop the simulation after 100m
    while True:
        if not plt.fignum_exists(fig.number):
            # Recreate the figure and axes if the window was closed
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            fig.suptitle("EKF State Estimation")

            # Reset the plot settings
            ax1.set_title("Position")
            ax1.set_xlabel("X position")
            ax1.set_ylabel("Y position")
            ax1.grid()

            ax2.set_title("Yaw Angle")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Yaw (rad)")
            ax2.grid()
            plt.legend()

            plt.ion()  # Turn interactive mode back on
            plt.show()

        control = ego_vehicle.actor.get_control()
        normalized_steer = control.steer  # Normalize the steering angle to [-1, 1]
        max_steer_angle = 0.5  # Maximum steering angle
        delta = normalized_steer * max_steer_angle  # Steering angle in radians
        u = np.array([control.throttle*2, delta])

        # EKF prediction and update
        # get imu data
        imu_data = ego_vehicle.imu_data_queue.get_nowait()
        # ego_vehicle.ekf.predict(u=u)
        ego_vehicle.ekf.predict(u=imu_data, imu_prediction=True, dt=0.01)  # Predict the next state
        ego_vehicle.update_ekf()

        ekf_state_x = ego_vehicle.get_latest_state()

        # plotting the results
        ax1.scatter(ekf_state_x[0], ekf_state_x[1], color='r', marker='x', label='EKF Estimate')  # Plot EKF estimated position

        if i % 5 == 0:  # Only plot every 5 iterations to reduce clutter
            # Get position covariance from EKF
            plot_covariance_ellipse(
                ax1,
                ekf_state_x[:2],  # Just the position part [x,y]
                ego_vehicle.ekf.P_post,  # Full covariance matrix
                sigma=2,  # 2-sigma (95% confidence)
                color='r',
                alpha=0.1  # Make it transparent
            )
        # ax1.scatter(ego_vehicle.ekf.x_prior[0], ego_vehicle.ekf.x_prior[1], color='y', marker='o', label='EKF Prior')  # Plot EKF prior position
        ax1.scatter(ego_vehicle.actor.get_transform().location.x,
                    ego_vehicle.actor.get_transform().location.y,
                    color='b', label='Actual Position')
        # plot the GPS location
        ax1.scatter(ego_vehicle.carla_coords[0], ego_vehicle.carla_coords[1], color='g', label='GPS Location')

        ax2.scatter(i, ego_vehicle.ekf.x[2], color='r', marker='x', label='EKF Estimate')  # Plot EKF estimated yaw
        ax2.scatter(i, ego_vehicle.ekf.x_prior[2], color='y', marker='x', label='EKF prediction')  # Plot EKF estimated yaw
        ax2.scatter(i, np.deg2rad(ego_vehicle.actor.get_transform().rotation.yaw), color='b', label='Actual Yaw')  # Plot actual yaw

        i += 1
        # plot gps location
        ax1.scatter(ego_vehicle.carla_coords[0], ego_vehicle.carla_coords[1], color='g', label='GPS Location')

        # update spectator camera to follow the vehicle
        spectator_location = ego_vehicle.actor.get_transform()
        spectator_location.location.z += 5
        spectator_location.location.x += 5
        carla_manager.spectator.set_transform(spectator_location)  # Keep the spectator camera in the same location

        # update the world
        carla_manager.world.tick()
        plt.pause(.1)
        print('Ground Truth:', ego_vehicle.actor.get_transform().location)
        print('GT yaw:', ego_vehicle.actor.get_transform().rotation.yaw, np.deg2rad(ego_vehicle.actor.get_transform().rotation.yaw))
        print('vehicle velocity:', ego_vehicle.actor.get_velocity())
        if ego_vehicle.actor.get_transform().location.distance(initial_location) > 100:
            break
    plt.ioff()
    plt.show(block=True)  # Show the plot and block until closed
    carla_manager.__del__()  # Clean up the Carla manager and close the simulation
