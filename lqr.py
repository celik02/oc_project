import time
import numpy as np
import math
import scipy.linalg as la

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.basic_agent import BasicAgent
from vehicle import Vehicle  # Import the Vehicle class
import traceback
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
FORMAT = "[%(asctime)s.%(msecs)03d %(filename)15s:%(lineno)3s - %(funcName)17s() ] %(levelname)s %(message)s"
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True, format=FORMAT, datefmt='%H:%M:%S')
dt = 0.2  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

PRECEDING_SPEED = 15 / 3.6  # m/s, speed of the preceding vehicle

# synchronous_mode will make the simulation predictable
synchronous_mode = True


def bicycle_vehicle_dynamics(x, u, dt):
    """
    Bicycle kinematic model in global coordinates.

    State vector x = [x, y, psi, v, a]  # Position, heading, velocity, acceleration
    Control vector u = [u_a, delta]      # Acceleration command, steering angle

    Returns the next state after time dt using Euler integration.
    """
    # Vehicle parameters
    lr = 1.5    # Distance from CG to rear axle
    lf = 1.375  # Distance from CG to front axle
    L = lr + lf  # Wheelbase
    tau_a = 0.5  # Acceleration time constant

    # Extract states
    x_pos = x[0]    # X position in global coordinates
    y_pos = x[1]    # Y position in global coordinates
    psi = x[2]      # Heading angle
    v = x[3]        # Velocity
    a = x[4]        # Current acceleration

    # Extract control inputs
    u_a = u[0]      # Acceleration command
    delta = u[1]    # Steering angle control

    # Compute state derivatives based on the bicycle model equations
    x_dot = v * np.cos(psi)              # Rate of change of x-position
    y_dot = v * np.sin(psi)              # Rate of change of y-position
    psi_dot = v * np.tan(delta) / L      # Rate of change of heading
    v_dot = a                            # Rate of change of velocity equals acceleration
    a_dot = (u_a - a) / tau_a            # Acceleration dynamics with time constant

    # Euler integration to compute next state
    x_next = np.zeros(5)
    x_next[0] = x_pos + x_dot * dt
    x_next[1] = y_pos + y_dot * dt
    x_next[2] = psi + psi_dot * dt
    x_next[3] = v + v_dot * dt
    x_next[4] = a + a_dot * dt

    return x_next


def linearize_bicycle_model(x_ref, u_ref, dt):
    """
    Linearize the bicycle model around a reference point.

    Returns:
        A: State transition matrix
        B: Control input matrix
    """
    # Vehicle parameters
    lr = 1.5    # Distance from CG to rear axle
    lf = 1.375  # Distance from CG to front axle
    L = lr + lf  # Wheelbase
    tau_a = 0.5  # Acceleration time constant

    # Extract reference state
    x_ref_pos = x_ref[0]
    y_ref_pos = x_ref[1]
    psi_ref = x_ref[2]
    v_ref = x_ref[3]
    a_ref = x_ref[4]

    # Extract reference control
    u_a_ref = u_ref[0]
    delta_ref = u_ref[1]

    # Linearized continuous-time system matrices
    # State: [x, y, psi, v, a]
    # Control: [u_a, delta]

    # Jacobian of state derivatives with respect to state (A matrix)
    A = np.zeros((5, 5))
    A[0, 2] = -v_ref * np.sin(psi_ref)
    A[0, 3] = np.cos(psi_ref)
    A[1, 2] = v_ref * np.cos(psi_ref)
    A[1, 3] = np.sin(psi_ref)
    A[2, 3] = np.tan(delta_ref) / L
    A[3, 4] = 1.0
    A[4, 4] = -1.0 / tau_a

    # Jacobian of state derivatives with respect to control (B matrix)
    B = np.zeros((5, 2))
    B[2, 1] = v_ref / (L * np.cos(delta_ref)**2)  # d(psi_dot)/d(delta)
    B[4, 0] = 1.0 / tau_a                         # d(a_dot)/d(u_a)

    # Convert to discrete time (Euler approximation)
    Ad = np.eye(5) + A * dt
    Bd = B * dt

    return Ad, Bd


class BicycleLQRController:
    def __init__(self, dt=0.1, carla_manager=None):
        self.carla_manager = carla_manager
        self.dt = dt

        # Control constraints
        self.max_accel = 3.0
        self.min_accel = -3.0
        self.max_steer = np.pi / 4  # Max steering allowed (rad)
        self.min_steer = -np.pi / 4  # Min steering allowed (rad)

        # LQR cost matrices
        # State cost matrix - penalize deviation from reference state
        self.Q = np.diag([10.0, 10.0, 1.0, 5.0, 1.0])  # [x, y, psi, v, a]
        # Control cost matrix - penalize control effort
        self.R = np.diag([1.0, 10.0])  # [u_a, delta]

        # Reference trajectory
        self.x_ref = np.zeros(5)  # [x, y, psi, v, a]
        self.u_ref = np.zeros(2)  # [u_a, delta]

        # LQR gain matrix
        self.K = None

        # Prediction horizon for visualization
        self.horizon = 30

    def compute_lqr_gain(self, Ad, Bd):
        """
        Solve the discrete-time algebraic Riccati equation for LQR gain.

        Returns:
            K: Optimal feedback gain matrix
        """
        # Solve the discrete-time algebraic Riccati equation
        P = la.solve_discrete_are(Ad, Bd, self.Q, self.R)

        # Compute optimal feedback gain
        K = np.linalg.inv(self.R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad

        return K

    def run_step(self, ego_vehicle, preceding_vehicle, target_speed):
        """
        Execute one step of LQR control

        Args:
            ego_vehicle: Vehicle class instance for ego vehicle
            preceding_vehicle: Vehicle class instance for preceding vehicle
            target_speed: Desired speed in m/s

        Returns:
            carla.VehicleControl object
        """
        # Get ego vehicle state
        x0 = ego_vehicle.get_vehicle_state()
        print(f"Ego vehicle state: {x0}")

        # Get preceding vehicle position in ego coordinates
        preceding_location = preceding_vehicle.actor.get_location()
        preceding_pos = ego_vehicle.world_to_ego_coordinates(preceding_location)

        # Generate reference trajectory based on target speed and collision avoidance
        self.update_reference_trajectory(x0, preceding_pos, target_speed)

        # Linearize the model around the reference point
        Ad, Bd = linearize_bicycle_model(self.x_ref, self.u_ref, self.dt)

        # Compute LQR gain matrix
        self.K = self.compute_lqr_gain(Ad, Bd)

        # Compute control input based on state error
        state_error = x0 - self.x_ref
        u = self.u_ref - self.K @ state_error

        # Apply control constraints
        u[0] = np.clip(u[0], self.min_accel, self.max_accel)  # Limit acceleration
        u[1] = np.clip(u[1], self.min_steer, self.max_steer)  # Limit steering angle

        # Convert to CARLA control
        control = self.convert_to_control(u)

        return control

    def update_reference_trajectory(self, x0, preceding_pos, target_speed):
        """
        Update reference trajectory based on current state and preceding vehicle.

        For simplicity, we use a straight-line trajectory at target speed.
        The reference x position is shifted laterally if the preceding vehicle is close.
        """
        # Default reference is to maintain current position with desired speed
        self.x_ref = np.copy(x0)
        self.x_ref[3] = target_speed  # Set reference velocity to target
        self.x_ref[4] = 0.0           # Set reference acceleration to zero

        # Distance to preceding vehicle
        distance_to_preceding = preceding_pos[0]  # Longitudinal distance
        lateral_offset = preceding_pos[1]         # Lateral distance

        # If preceding vehicle is close, plan to move laterally
        safety_distance = 8.0  # Longitudinal safety distance
        if distance_to_preceding < safety_distance and abs(lateral_offset) < 2.0:
            # Plan to change lane (move to the left)
            self.x_ref[1] = -3.5  # Left lane position

        # Reference control is zero acceleration and zero steering
        self.u_ref = np.zeros(2)

    def predict_trajectory(self, x0, horizon):
        """
        Predict trajectory using LQR control law

        Args:
            x0: Initial state
            horizon: Prediction horizon

        Returns:
            Predicted trajectory [(x, y), ...]
        """
        trajectory = []
        x = np.copy(x0)

        for i in range(horizon):
            # Compute control using LQR gain
            state_error = x - self.x_ref
            u = self.u_ref - self.K @ state_error

            # Apply control constraints
            u[0] = np.clip(u[0], self.min_accel, self.max_accel)
            u[1] = np.clip(u[1], self.min_steer, self.max_steer)

            # Propagate state using bicycle model
            x = bicycle_vehicle_dynamics(x, u, self.dt)

            # Store position
            trajectory.append((x[0], x[1]))

        return trajectory

    def convert_to_control(self, u):
        """
        Convert control vector to CARLA control

        Args:
            u: Control vector [u_a, delta]

        Returns:
            carla.VehicleControl object
        """
        # Extract control inputs
        accel_cmd = u[0]  # Acceleration command
        steer = u[1]      # Steering angle

        # Create CARLA control
        control = carla.VehicleControl()

        # Convert acceleration to throttle/brake
        if accel_cmd >= 0:
            control.throttle = min(accel_cmd / self.max_accel, 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel_cmd / self.min_accel, 1.0)

        # Convert steering angle to CARLA steering [-1, 1]
        max_steering_angle = np.radians(69.99999237060547)  # CARLA max steering
        control.steer = steer / max_steering_angle
        control.steer = np.clip(control.steer, -1.0, 1.0)

        return control


class CarlaManager:
    _instance = None

    def __new__(cls, *args, **kwargs):  # to make this class singleton
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.client.load_world("Town04")
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.sampling_resolution = 0.5
        self.global_planner = GlobalRoutePlanner(self.map, self.sampling_resolution)
        self.fixed_delta_seconds = dt

        self.spectator = self.world.get_spectator()
        spectator_transform = carla.Transform(
            carla.Location(
                x=SPAWN_LOCATION[0] + 15, y=SPAWN_LOCATION[1], z=SPAWN_LOCATION[2]
            ),
            carla.Rotation(pitch=-37, yaw=-177, roll=0),
        )
        self.spectator.set_transform(spectator_transform)
        if synchronous_mode:
            self.settings = self.world.get_settings()
            self.settings.fixed_delta_seconds = dt
            self.settings.synchronous_mode = True
            self.world.apply_settings(self.settings)

    def __del__(self):
        if synchronous_mode:
            self.settings.synchronous_mode = False
            self.world.apply_settings(self.settings)

    def spawn_vehicle(self, blueprint_name, spawn_point):
        """
        spawn a vehicle at the given spawn_point
        """
        spawn_transform = self.map.get_waypoint(
            carla.Location(x=spawn_point[0], y=spawn_point[1], z=10),
            project_to_road=True,
        ).transform
        spawn_transform.location.z = spawn_transform.location.z + 0.5
        print("Spawn location:", spawn_transform.location)
        print("Spawn rotation:", spawn_transform.rotation)
        blueprint = self.blueprint_library.filter(blueprint_name)[0]
        vehicle = self.world.spawn_actor(blueprint, spawn_transform)
        return vehicle

    def visualize_trajectory(self, trajectory, color=carla.Color(r=0, g=255, b=0), lifetime=0.5):
        """
        Visualize a trajectory in the CARLA world
        """
        for i in range(len(trajectory)-1):
            # Transform from ego to world coordinates
            start = carla.Location(x=trajectory[i][0], y=trajectory[i][1], z=0.5)
            end = carla.Location(x=trajectory[i+1][0], y=trajectory[i+1][1], z=0.5)
            self.world.debug.draw_line(start, end, thickness=0.1, color=color, life_time=lifetime)

    def restart_world(self):
        """
        Restart the world by removing all actors and reloading the world
        """
        print("Cleaning up the world...")

        # First destroy all vehicles, sensors and other actors
        actor_list = self.world.get_actors()
        for actor in actor_list:
            # Check if actor is not already destroyed
            if actor.is_alive:
                # Only destroy vehicles and sensors (not static world objects)
                if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor'):
                    try:
                        actor.destroy()
                        print(f"Destroyed {actor.type_id}")
                    except Exception as e:
                        print(f"Error destroying {actor.type_id}: {e}")

        # Reset the simulation settings if in synchronous mode
        if synchronous_mode:
            self.settings = self.world.get_settings()
            self.settings.synchronous_mode = False
            self.world.apply_settings(self.settings)
            print("Reset simulation to asynchronous mode")

        # Reload the world (a lighter operation than restarting Carla)
        print("Reloading world...")
        self.client.reload_world()
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.sampling_resolution = 0.5
        self.global_planner = GlobalRoutePlanner(self.map, self.sampling_resolution)
        self.fixed_delta_seconds = dt

        # Reset spectator position
        self.spectator = self.world.get_spectator()
        spectator_transform = carla.Transform(
            carla.Location(
                x=SPAWN_LOCATION[0] + 15, y=SPAWN_LOCATION[1], z=SPAWN_LOCATION[2]
            ),
            carla.Rotation(pitch=-37, yaw=-177, roll=0),
        )
        self.spectator.set_transform(spectator_transform)

        # Restore synchronous mode if needed
        if synchronous_mode:
            self.settings = self.world.get_settings()
            self.settings.fixed_delta_seconds = dt
            self.settings.synchronous_mode = True
            self.world.apply_settings(self.settings)
            print("Restored synchronous mode")

        print("World has been successfully reset")


def run_simulation_with_lqr():
    """
    Run the main simulation with LQR controller
    """
    print("Starting simulation with LQR controller...")

    # Create CarlaManager instance
    carla_manager = CarlaManager()
    print("CarlaManager created")

    preceding_vehicle_actor = None
    ego_vehicle_actor = None

    try:
        # Spawn preceding vehicle
        preceding_vehicle_actor = carla_manager.spawn_vehicle(
            "vehicle.tesla.model3", SPAWN_LOCATION
        )

        preceding_vehicle_actor.set_autopilot(False)
        time.sleep(1)  # allow vehicle to spawn

        # Wrap with Vehicle class
        preceding_vehicle = Vehicle(preceding_vehicle_actor)

        # Spawn ego vehicle behind preceding vehicle
        spawn_loc_copy = SPAWN_LOCATION.copy()
        spawn_loc_copy[0] += 10  # 10 meters behind
        ego_vehicle_actor = carla_manager.spawn_vehicle("vehicle.tesla.model3", spawn_loc_copy)

        if synchronous_mode:
            carla_manager.world.tick()
        else:
            time.sleep(1)

        # Wrap with Vehicle class
        ego_vehicle = Vehicle(ego_vehicle_actor)

        # Save spawn transform and initialize transformation matrices
        ego_vehicle.transform_to_spawn = ego_vehicle_actor.get_transform()
        world_to_ego, ego_to_world = ego_vehicle.get_transform_matrices()

        # Create LQR controller
        lqr_controller = BicycleLQRController(dt=dt, carla_manager=carla_manager)

        # Target speed (m/s)
        target_speed = 20 / 3.6  # 20 km/h

        # Main control loop
        try:
            print("Starting simulation with LQR. Press Ctrl+C to exit...")
            preceding_speed_error_prev = 0.0  # Initialize previous speed error
            preceding_steer_error_prev = 0.0  # Initialize previous steering error
            kp_speed = 0.5
            kd_speed = 0.1
            kp_steer = 0.5
            kd_steer = 0.1

            while True:
                # Update preceding vehicle with PD control
                control_cmd = carla.VehicleControl()

                # Set a constant speed for the preceding vehicle using a PD controller
                error_speed = PRECEDING_SPEED - preceding_vehicle_actor.get_velocity().length()
                PD_preceding_control = kp_speed * error_speed + kd_speed * (error_speed - preceding_speed_error_prev) / dt
                preceding_speed_error_prev = error_speed

                # Apply control to preceding vehicle
                if PD_preceding_control > 0:
                    control_cmd.throttle = min(PD_preceding_control, 1.0)
                    control_cmd.brake = 0.0
                else:
                    control_cmd.throttle = 0.0
                    control_cmd.brake = min(-PD_preceding_control, 1.0)

                # Apply PD control for steering to keep the preceding vehicle in lane
                current_preceding_location = preceding_vehicle_actor.get_location()
                road_wp = carla_manager.map.get_waypoint(current_preceding_location, project_to_road=True)
                road_direction = road_wp.transform.get_forward_vector()
                vehicle_direction = preceding_vehicle_actor.get_transform().get_forward_vector()
                road_heading = math.atan2(road_direction.y, road_direction.x)
                vehicle_heading = math.atan2(vehicle_direction.y, vehicle_direction.x)
                heading_error = road_heading - vehicle_heading

                # Normalize heading error to [-1, 1]
                heading_error = heading_error / math.pi
                control_steer = kp_steer * heading_error + kd_steer * (heading_error - preceding_steer_error_prev) / dt
                preceding_steer_error_prev = heading_error

                # Apply steering control
                control_cmd.steer = np.clip(control_steer, -1.0, 1.0)
                preceding_vehicle_actor.apply_control(control_cmd)

                # Get LQR control for ego vehicle
                ego_control = lqr_controller.run_step(
                    ego_vehicle, preceding_vehicle, target_speed
                )

                # Apply control to ego vehicle
                ego_vehicle_actor.apply_control(ego_control)

                # Predict trajectory for visualization
                ego_state = ego_vehicle.get_vehicle_state()
                predicted_trajectory = lqr_controller.predict_trajectory(ego_state, 20)

                # Visualize predicted trajectory
                # To transform from ego to world coordinates, we need to:
                # 1. Get the current ego vehicle transform
                # 2. Apply the transform to the predicted trajectory points
                ego_transform = ego_vehicle_actor.get_transform()
                world_trajectory = []
                for point in predicted_trajectory:
                    # Convert from ego to world coordinates
                    world_point = ego_vehicle.ego_to_world_coordinates(
                        np.array([point[0], point[1], 0.0]), use_current_transform=True)
                    world_trajectory.append((world_point[0], world_point[1]))

                # Visualize trajectory in world coordinates
                for i in range(len(world_trajectory)-1):
                    start = carla.Location(x=world_trajectory[i][0], y=world_trajectory[i][1], z=0.5)
                    end = carla.Location(x=world_trajectory[i+1][0], y=world_trajectory[i+1][1], z=0.5)
                    carla_manager.world.debug.draw_line(start, end, thickness=0.1,
                                                       color=carla.Color(r=0, g=255, b=0),
                                                       life_time=0.1)

                # Advance simulation
                carla_manager.world.tick()

        except KeyboardInterrupt:
            print("\nSimulation terminated by user")

    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()

    finally:
        # Clean up
        if preceding_vehicle_actor and preceding_vehicle_actor.is_alive:
            preceding_vehicle_actor.destroy()

        if ego_vehicle_actor and ego_vehicle_actor.is_alive:
            ego_vehicle_actor.destroy()

        if synchronous_mode:
            settings = carla_manager.world.get_settings()
            settings.synchronous_mode = False
            carla_manager.world.apply_settings(settings)


if __name__ == "__main__":
    run_simulation_with_lqr()