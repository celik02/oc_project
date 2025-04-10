import sys
print('\n'.join(sys.path))
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.constant_velocity_agent import (
    ConstantVelocityAgent,
)  # just constant velocity no lateral control
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.controller import VehiclePIDController

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

dt = 0.01  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

# synchronous_mode will make the simulation predictable
synchronous_mode = True


def vehicle_dynamics(x, u, dt):
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

    # State update
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + x[3] * np.cos(x[2]) * dt
    x_next[1] = x[1] + x[3] * np.sin(x[2]) * dt
    x_next[2] = x[2] + x[3] * np.tan(u[1]) / L * dt
    x_next[3] = x[3] + x[4] * dt
    x_next[4] = x[4] + (u[0] - x[4]) / tau_a * dt

    return x_next


class CoordinateTransformManager:
    """
    Manages coordinate transformations between CARLA's global coordinate system
    and a local coordinate system centered at the ego vehicle's spawn position.
    """

    def __init__(self, ego_spawn_transform):
        """
        Initialize with the ego vehicle's spawn transform to establish the new origin.

        Args:
            ego_spawn_transform (carla.Transform): Spawn transform of the ego vehicle
        """
        self.origin = ego_spawn_transform
        print(f"Established new coordinate origin at: {self.origin.location}")

        # Cache the rotation matrices for efficiency
        self._setup_rotation_matrices()

    def _setup_rotation_matrices(self):
        """Pre-compute rotation matrices based on ego spawn orientation."""
        # Convert yaw angle to radians for rotation matrix
        yaw_rad = self.origin.rotation.yaw * np.pi / 180.0

        # Create rotation matrix (global to local)
        self.rotation_g2l = np.array([
            [np.cos(yaw_rad), np.sin(yaw_rad), 0],  # x-axis (forward in ego frame)
            [-np.sin(yaw_rad), np.cos(yaw_rad), 0], # y-axis (right in ego frame)
            [0, 0, 1]                               # z-axis (up remains up)
        ])

        # Create inverse rotation matrix (local to global)
        self.rotation_l2g = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])

    def global_to_local_location(self, global_location):
        """
        Convert a CARLA location from global to local coordinates.

        Args:
            global_location (carla.Location): Location in CARLA's global coordinates

        Returns:
            carla.Location: The same location in the local coordinate system
        """
        # Extract global location as numpy array
        global_pos = np.array([global_location.x, global_location.y, global_location.z])

        # Extract origin as numpy array
        origin_pos = np.array([
            self.origin.location.x,
            self.origin.location.y,
            self.origin.location.z
        ])

        # Translate to origin-centered coordinates
        relative_pos = global_pos - origin_pos

        # Apply rotation to align with ego vehicle's orientation
        local_pos = np.dot(self.rotation_g2l, relative_pos)

        # Return as a CARLA Location
        return carla.Location(x=local_pos[0], y=local_pos[1], z=local_pos[2])

    def local_to_global_location(self, local_location):
        """
        Convert a location from local coordinates back to CARLA's global coordinates.

        Args:
            local_location (carla.Location or np.array): Location in local coordinates

        Returns:
            carla.Location: The same location in CARLA's global coordinate system
        """
        # Convert input to numpy array if needed
        if isinstance(local_location, carla.Location):
            local_pos = np.array([local_location.x, local_location.y, local_location.z])
        else:
            local_pos = np.array(local_location)

        # Apply inverse rotation
        rotated_pos = np.dot(self.rotation_l2g, local_pos)

        # Translate back to global coordinates
        origin_pos = np.array([
            self.origin.location.x,
            self.origin.location.y,
            self.origin.location.z
        ])
        global_pos = rotated_pos + origin_pos

        # Return as a CARLA Location
        return carla.Location(x=global_pos[0], y=global_pos[1], z=global_pos[2])

    def global_to_local_transform(self, global_transform):
        """
        Convert a CARLA transform from global to local coordinates.

        Args:
            global_transform (carla.Transform): Transform in CARLA's global coordinates

        Returns:
            carla.Transform: The same transform in the local coordinate system
        """
        # Convert location
        local_location = self.global_to_local_location(global_transform.location)

        # Convert rotation (only yaw for now, as we're primarily concerned with 2D)
        global_yaw_rad = global_transform.rotation.yaw * np.pi / 180.0
        origin_yaw_rad = self.origin.rotation.yaw * np.pi / 180.0
        local_yaw_rad = global_yaw_rad - origin_yaw_rad

        # Normalize to range [-π, π]
        local_yaw_rad = ((local_yaw_rad + np.pi) % (2 * np.pi)) - np.pi
        local_yaw_deg = local_yaw_rad * 180.0 / np.pi

        # Create new transform
        local_rotation = carla.Rotation(
            pitch=global_transform.rotation.pitch,
            yaw=local_yaw_deg,
            roll=global_transform.rotation.roll
        )

        return carla.Transform(local_location, local_rotation)

    def local_to_global_transform(self, local_transform):
        """
        Convert a transform from local coordinates back to CARLA's global coordinates.

        Args:
            local_transform (carla.Transform): Transform in local coordinates

        Returns:
            carla.Transform: The same transform in CARLA's global coordinate system
        """
        # Convert location
        global_location = self.local_to_global_location(local_transform.location)

        # Convert rotation
        local_yaw_rad = local_transform.rotation.yaw * np.pi / 180.0
        origin_yaw_rad = self.origin.rotation.yaw * np.pi / 180.0
        global_yaw_rad = local_yaw_rad + origin_yaw_rad

        # Normalize to range [-π, π]
        global_yaw_rad = ((global_yaw_rad + np.pi) % (2 * np.pi)) - np.pi
        global_yaw_deg = global_yaw_rad * 180.0 / np.pi

        # Create new transform
        global_rotation = carla.Rotation(
            pitch=local_transform.rotation.pitch,
            yaw=global_yaw_deg,
            roll=local_transform.rotation.roll
        )

        return carla.Transform(global_location, global_rotation)

    def global_to_local_velocity(self, global_velocity):
        """
        Convert a velocity vector from global to local coordinates.
        Only rotation is applied (no translation).

        Args:
            global_velocity (carla.Vector3D): Velocity in global coordinates

        Returns:
            carla.Vector3D: Velocity in local coordinates
        """
        # Convert to numpy array
        global_vel = np.array([global_velocity.x, global_velocity.y, global_velocity.z])

        # Apply rotation (no translation for velocity vectors)
        local_vel = np.dot(self.rotation_g2l, global_vel)

        # Return as CARLA Vector3D
        return carla.Vector3D(x=local_vel[0], y=local_vel[1], z=local_vel[2])

    def local_to_global_velocity(self, local_velocity):
        """
        Convert a velocity vector from local to global coordinates.

        Args:
            local_velocity (carla.Vector3D): Velocity in local coordinates

        Returns:
            carla.Vector3D: Velocity in global coordinates
        """
        # Convert to numpy array
        local_vel = np.array([local_velocity.x, local_velocity.y, local_velocity.z])

        # Apply inverse rotation
        global_vel = np.dot(self.rotation_l2g, local_vel)

        # Return as CARLA Vector3D
        return carla.Vector3D(x=global_vel[0], y=global_vel[1], z=global_vel[2])


class MPCController:
    def __init__(self, horizon=10, dt=0.1, coordinate_manager=None):
        # Original initialization
        self.horizon = horizon
        self.dt = dt

        # Control constraints
        self.max_accel = 3.0
        self.min_accel = -5.0
        self.max_steer = 0.25
        self.min_steer = -0.25

        # Cost function weights
        self.w_progress = 2.0    # Forward progress reward
        self.w_lane = 50.0        # Lane centering reward
        self.w_speed = 1.0       # Target speed reward
        self.w_accel = 0.2       # Acceleration minimization
        self.w_steer = 0.5       # Steering minimization
        self.w_jerk = 0.1        # Jerk minimization

        # Store coordinate transform manager
        self.coordinate_manager = coordinate_manager

    def run_step(self, ego_vehicle, preceding_vehicle, target_speed):
        """Run a control step using local coordinates"""
        # Extract current state in global coordinates
        ego_transform_global = ego_vehicle.get_transform()
        ego_velocity_global = ego_vehicle.get_velocity()
        lead_transform_global = preceding_vehicle.get_transform()
        lead_velocity_global = preceding_vehicle.get_velocity()

        # Convert to local coordinates using our coordinate manager
        ego_transform = self.coordinate_manager.global_to_local_transform(ego_transform_global)
        ego_velocity = self.coordinate_manager.global_to_local_velocity(ego_velocity_global)
        lead_transform = self.coordinate_manager.global_to_local_transform(lead_transform_global)
        lead_velocity = self.coordinate_manager.global_to_local_velocity(lead_velocity_global)

        # Now all values are in the local coordinate system where:
        # - x is forward direction from ego vehicle's spawn position
        # - y is rightward direction from ego vehicle's spawn position
        # - z is upward

        # Construct current state vector in local coordinates
        x0 = np.array([
            ego_transform.location.x,
            ego_transform.location.y,
            ego_transform.rotation.yaw * np.pi / 180.0,  # Convert degrees to radians
            np.sqrt(ego_velocity.x**2 + ego_velocity.y**2),  # Speed magnitude
            0.0  # Estimated acceleration
        ])

        # Get lead vehicle state and speed in local coordinates
        lead_state = (lead_transform.location.x, lead_transform.location.y)
        lead_speed = np.sqrt(lead_velocity.x**2 + lead_velocity.y**2)

        # Generate lead vehicle prediction
        lead_vehicle_pred = self.get_lead_vehicle_prediction(lead_state, lead_speed)

        # Solve optimization problem using local coordinates
        u_optimal = self.solve_for_control(x0, lead_vehicle_pred, target_speed)

        # Convert to CARLA control - this doesn't change as control inputs
        # themselves are not affected by the coordinate transform
        control = self.convert_to_control(u_optimal[0])

        return control

    def solve_for_control(self, x0, lead_vehicle_pred, target_speed):
        # Initial guess - zeros for all controls
        u0 = np.zeros(self.horizon * 2)

        # Define bounds for controls
        bounds = []
        for i in range(self.horizon):
            bounds.append((self.min_accel, self.max_accel))
            bounds.append((self.min_steer, self.max_steer))

        # Solve optimization problem with constraints
        result = minimize(
            self.objective_function,
            u0,
            args=(x0, lead_vehicle_pred),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 100}
        )

        # Extract optimal control sequence
        u_optimal = result.x.reshape(self.horizon, 2)

        return u_optimal

    def get_lead_vehicle_prediction(self, lead_vehicle_state, lead_vehicle_speed):
        """
        Predict lead vehicle trajectory (constant velocity model)
        """
        predictions = []
        x_lead, y_lead = lead_vehicle_state

        for i in range(self.horizon):
            x_lead = x_lead + lead_vehicle_speed * self.dt
            predictions.append((x_lead, y_lead))

        return predictions

    def objective_function(self, u, x0, lead_vehicle_pred):
        """
        Objective function without explicit reference trajectory

        Instead of tracking reference points, we optimize for:
        1. Forward progress (longitudinal advancement)
        2. Lane centering (but allowing for lane changes)
        3. Target velocity
        4. Control effort minimization
        """
        # Reshape control sequence
        u = u.reshape(self.horizon, 2)

        # Initialize cost and current state
        cost = 0
        x = x0.copy()

        # Target speed (higher than lead vehicle)
        target_speed = 15.0  # m/s

        # Simulate trajectory and compute cost
        for i in range(self.horizon):
            # Get lead vehicle position at this step
            lead_x, lead_y = lead_vehicle_pred[i]

            # Compute longitudinal distance to lead vehicle in ego frame
            # This requires transforming from world to ego-vehicle coordinates
            relative_x = lead_x - x[0]
            relative_y = lead_y - x[1]
            ego_heading = x[2]

            # Project onto ego vehicle's longitudinal axis
            rel_long = relative_x * np.cos(ego_heading) + relative_y * np.sin(ego_heading)
            rel_lat = -relative_x * np.sin(ego_heading) + relative_y * np.cos(ego_heading)

            # 1. Progress reward - encourage moving forward
            # We negate this term because we want to maximize progress
            cost -= self.w_progress * x[0]

            # 2. Lane centering - encourage staying in the lane
            lane_y = 0  # Current lane center y-coordinate

            # When far from lead vehicle, prefer the original lane
            lateral_error = (x[1] - lane_y)**2

            cost += self.w_lane * lateral_error
            # 3. Target speed - encourage maintaining desired speed
            cost += self.w_speed * max(0, target_speed - x[3])**2

            # 4. Control effort
            cost += self.w_accel * u[i, 0]**2  # Acceleration
            cost += self.w_steer * u[i, 1]**2  # Steering

            # 6. Jerk minimization for comfort
            if i > 0:
                prev_accel = u[i-1, 0]
                cost += self.w_jerk * ((u[i, 0] - prev_accel) / self.dt)**2

            # Update state for next step
            x = vehicle_dynamics(x, u[i], self.dt)

        return cost

    def convert_to_control(self, u_optimal):
        """
        Convert MPC control output to CARLA vehicle control

        Args:
            u_optimal: Optimal control input [acceleration, steering]

        Returns:
            carla.VehicleControl object
        """
        # Extract control inputs
        acceleration = u_optimal[0]
        steering = u_optimal[1]

        # Convert to CARLA control
        control = carla.VehicleControl()

        # Convert acceleration to throttle/brake
        if acceleration >= 0:
            control.throttle = min(acceleration / 3.0, 1.0)  # Assuming max accel of 3 m/s^2
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-acceleration / 5.0, 1.0)  # Assuming max decel of 5 m/s^2

        control.steer = steering / 0.5  # Normalize to [-1, 1]

        return control


class CarlaManager:
    _instance = None

    def __new__(cls, *args, **kwargs):  # to make this class singleton
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
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
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.world.get_actors()]
        )

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

    def debug_waypoints(self, waypoints):
        # draw trace_route outputs
        i = 0
        for w in waypoints:
            if i % 10 == 0:
                self.world.debug.draw_string(
                    w.transform.location,
                    "O",
                    draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=120.0,
                    persistent_lines=True,
                )

    def restart_world(self):
        """
        Restart the world
        """
        self.client.reload_world()
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


if __name__ == "__main__":
    carla_manager = CarlaManager()
    print("CarlaManager is created")

    # Spawn preceding vehicle (in global coordinates)
    preceding_vehicle = carla_manager.spawn_vehicle(
        "vehicle.tesla.model3", SPAWN_LOCATION
    )
    preceding_vehicle.set_autopilot(False)
    time.sleep(1)  # allow the vehicle to spawn

    # Spawn ego vehicle (in global coordinates)
    SPAWN_LOCATION[0] += 20  # 20 meters behind preceding vehicle
    ego_vehicle = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
    time.sleep(1)  # allow the vehicle to spawn

    # Create coordinate transform manager using ego vehicle's spawn transform
    # This effectively makes the ego vehicle's spawn position the new origin
    ego_spawn_transform = ego_vehicle.get_transform()
    coord_manager = CoordinateTransformManager(ego_spawn_transform)

    # The lane center y-coordinate in local coordinates would be 0 if the ego vehicle
    # spawned perfectly centered in the lane. Otherwise we need to determine this value.
    # We could either:
    # 1. Get the nearest waypoint and transform its y-coordinate to local
    # 2. Make an explicit calculation based on the lane width/positioning

    # Get the nearest waypoint to get lane center
    waypoint = carla_manager.map.get_waypoint(ego_vehicle.get_location())
    lane_center_global = waypoint.transform.location
    lane_center_local = coord_manager.global_to_local_location(lane_center_global)
    lane_y_local = lane_center_local.y
    print(f"Lane center y-coordinate in local frame: {lane_y_local}")

    # Basic control for preceding vehicle (still using CARLA's global coordinates)
    agent = BasicAgent(preceding_vehicle, target_speed=10)

    # Set destination for the preceding vehicle (global coordinates)
    current_location = preceding_vehicle.get_location()
    current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)
    next_wps = current_wp.next(100.0)  # 100 meters ahead

    if next_wps:  # if there is a waypoint ahead
        destination = next_wps[0].transform.location
    else:
        destination = current_location
    agent.set_destination(destination)

    # Create MPC controller with coordinate manager
    mpc_controller = MPCController(horizon=100, dt=0.1, coordinate_manager=coord_manager)

    # Target speed for ego vehicle (km/h)
    target_speed = 15.0
    init_ego_control = carla.VehicleControl()
    init_ego_control.steer = 0.0
    init_ego_control.throttle = 1.0
    init_ego_control.brake = 0.0

    ego_vehicle.apply_control(init_ego_control)

    # Main control loop
    try:
        while True:
            # Control preceding vehicle (using CARLA's global coordinates)
            control_cmd = agent.run_step()
            preceding_vehicle.apply_control(control_cmd)

            # Control ego vehicle with MPC (will use local coordinates internally)
            ego_control = mpc_controller.run_step(
                ego_vehicle, preceding_vehicle, target_speed / 3.6
            )  # Convert to m/s
            ego_vehicle.apply_control(ego_control)
            print(f"Ego vehicle control: throttle={ego_control.throttle}, brake={ego_control.brake}, steer={ego_control.steer}")

            # Print current status in local coordinates for debugging
            ego_loc_global = ego_vehicle.get_location()
            lead_loc_global = preceding_vehicle.get_location()

            ego_loc_local = coord_manager.global_to_local_location(ego_loc_global)
            lead_loc_local = coord_manager.global_to_local_location(lead_loc_global)

            distance = np.sqrt(
                (ego_loc_local.x - lead_loc_local.x)**2 +
                (ego_loc_local.y - lead_loc_local.y)**2
            )

            carla_manager.world.tick()  # Step the world forward
            print(f"Ego vehicle global position: ({ego_loc_global.x:.2f}, {ego_loc_global.y:.2f}, {ego_loc_global.z:.2f})")
            print(f"Ego vehicle local position: ({ego_loc_local.x:.2f}, {ego_loc_local.y:.2f}, {ego_loc_local.z:.2f})")
            print(f"Lead vehicle local position: ({lead_loc_local.x:.2f}, {lead_loc_local.y:.2f}, {lead_loc_local.z:.2f})")
            print(f"Distance to lead vehicle: {distance:.2f} m")

    except KeyboardInterrupt:
        print("Simulation terminated by user")
        carla_manager.restart_world()
    finally:
        # Clean up
        if preceding_vehicle and ego_vehicle:
            preceding_vehicle.destroy()
            ego_vehicle.destroy()
        print("Vehicles destroyed")