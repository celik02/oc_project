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
import casadi as ca  # For optimization (you'll need to install this)



dt = 0.1  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

# synchronous_mode will make the simulation predictable
synchronous_mode = False


def vehicle_dynamics(x, u, dt):
    """
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
    L = 2.5  # Wheelbase
    tau_a = 0.5  # Acceleration time constant

    # State update
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + x[3] * np.cos(x[2]) * dt
    x_next[1] = x[1] + x[3] * np.sin(x[2]) * dt
    x_next[2] = x[2] + x[3] * np.tan(u[1]) / L * dt
    x_next[3] = x[3] + x[4] * dt
    x_next[4] = x[4] + (u[0] - x[4]) / tau_a * dt

    return x_next


class MPCController:
    def __init__(self, horizon=10, dt=0.1):
        """
        Simple MPC controller for vehicle overtaking

        Args:
            horizon: Prediction horizon length
            dt: Time step for discretization
        """
        self.horizon = horizon
        self.dt = dt

        # Control constraints
        self.max_accel = 3.0  # m/s^2
        self.min_accel = -5.0  # m/s^2
        self.max_steer = 0.5  # rad
        self.min_steer = -0.5  # rad

        # Cost function weights
        self.w_lat = 1.0  # Lateral position tracking
        self.w_vel = 0.5  # Velocity tracking
        self.w_accel = 0.1  # Minimize acceleration
        self.w_steer = 0.1  # Minimize steering
        self.w_jerk = 0.05  # Minimize jerk

        # Safety parameters
        self.safe_distance = 6.0  # Minimum safe distance to lead vehicle

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

    def objective_function(self, u, x0, ref_traj, lead_vehicle_pred):
        """
        MPC cost function

        Args:
            u: Flattened control sequence [u0, u1, ..., u_{N-1}]
            x0: Initial state
            ref_traj: Reference trajectory for lateral position and velocity
            lead_vehicle_pred: Predicted lead vehicle positions
        """
        # Reshape control sequence
        u = u.reshape(self.horizon, 2)

        # Initialize cost and current state
        cost = 0
        x = x0.copy()

        # Simulate trajectory and compute cost
        for i in range(self.horizon):
            # Extract reference values
            y_ref, v_ref = ref_traj[i]

            # Compute control inputs at step i
            accel = u[i, 0]
            steer = u[i, 1]

            # Compute cost components
            # 1. Reference tracking
            cost += self.w_lat * (x[1] - y_ref) ** 2  # Lateral position error
            cost += self.w_vel * (x[3] - v_ref) ** 2  # Velocity error

            # 2. Control effort
            cost += self.w_accel * accel**2
            cost += self.w_steer * steer**2

            # 3. Jerk (change in acceleration)
            if i > 0:
                prev_accel = u[i - 1, 0]
                cost += self.w_jerk * ((accel - prev_accel) / self.dt) ** 2

            # 4. Safety - soft constraint on distance to lead vehicle
            lead_x, lead_y = lead_vehicle_pred[i]
            distance = np.sqrt((x[0] - lead_x) ** 2 + (x[1] - lead_y) ** 2)

            # Heavily penalize being too close to lead vehicle
            if distance < self.safe_distance:
                cost += 100 * (self.safe_distance - distance) ** 2

            # Update state for next step
            x = vehicle_dynamics(x, u[i], self.dt)

        return cost

    def optimize_trajectory(self, x0, ref_traj, lead_vehicle_state, lead_vehicle_speed):
        """
        Solve the MPC optimization problem

        Returns:
            optimal_control: First control input from optimal sequence
        """
        # Generate lead vehicle prediction
        lead_vehicle_pred = self.get_lead_vehicle_prediction(
            lead_vehicle_state, lead_vehicle_speed
        )

        # Initial guess - zeros for all controls
        u0 = np.zeros(self.horizon * 2)

        # Define bounds for controls
        bounds = []
        for i in range(self.horizon):
            bounds.append((self.min_accel, self.max_accel))  # Acceleration bounds
            bounds.append((self.min_steer, self.max_steer))  # Steering bounds

        # Solve optimization problem
        result = minimize(
            self.objective_function,
            u0,
            args=(x0, ref_traj, lead_vehicle_pred),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 100},
        )

        # Extract optimal control sequence
        u_optimal = result.x.reshape(self.horizon, 2)

        # Return first control input
        return u_optimal[0]

    def generate_reference_trajectory(
        self, ego_vehicle, preceding_vehicle, target_speed, lane_width=3.5
    ):
        """
        Generate reference trajectory for overtaking

        Strategy:
        1. Start in current lane
        2. Move to adjacent lane when close to lead vehicle
        3. Accelerate to pass lead vehicle
        4. Return to original lane when sufficiently ahead
        """
        # Get current states
        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        lead_location = preceding_vehicle.get_location()

        # Extract positions
        ego_x, ego_y = ego_location.x, ego_location.y
        lead_x, lead_y = lead_location.x, lead_location.y

        # Compute longitudinal distance to lead vehicle
        heading = ego_transform.rotation.yaw * np.pi / 180.0
        dx = lead_x - ego_x
        dy = lead_y - ego_y

        # Convert to ego vehicle frame
        long_dist = dx * np.cos(heading) + dy * np.sin(heading)
        lat_dist = -dx * np.sin(heading) + dy * np.cos(heading)

        # Generate reference trajectory
        ref_traj = []

        for i in range(self.horizon):
            future_time = i * self.dt

            # Longitudinal position increases with time based on target speed
            future_long_pos = target_speed * future_time

            # Lateral position depends on overtaking phase
            if long_dist < 0:
                # Already passed the lead vehicle, merge back
                y_ref = 0  # Target is center of original lane
                v_ref = target_speed
            elif long_dist < 20:
                # Close to lead vehicle, move to adjacent lane
                y_ref = lane_width  # Target is center of adjacent lane
                v_ref = target_speed * 1.3  # Accelerate to pass
            else:
                # Far from lead vehicle, stay in lane
                y_ref = 0  # Target is center of original lane
                v_ref = target_speed

            ref_traj.append((y_ref, v_ref))

        return ref_traj

    def run_step(self, ego_vehicle, preceding_vehicle, target_speed):
        """
        Execute one step of MPC
        """
        # Get current state
        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        lead_transform = preceding_vehicle.get_transform()
        lead_velocity = preceding_vehicle.get_velocity()

        # Extract state components
        x = ego_transform.location.x
        y = ego_transform.location.y
        psi = ego_transform.rotation.yaw * np.pi / 180.0
        v = np.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

        # Estimate current acceleration (simplified)
        a = 0.0  # Could be improved with proper state estimation

        # Construct current state vector
        x0 = np.array([x, y, psi, v, a])

        # Get lead vehicle state and speed
        lead_x = lead_transform.location.x
        lead_y = lead_transform.location.y
        lead_speed = np.sqrt(lead_velocity.x**2 + lead_velocity.y**2)

        # Generate reference trajectory
        ref_traj = self.generate_reference_trajectory(
            ego_vehicle, preceding_vehicle, target_speed
        )

        # Solve MPC problem
        u_optimal = self.optimize_trajectory(x0, ref_traj, (lead_x, lead_y), lead_speed)

        # Extract control inputs
        acceleration = u_optimal[0]
        steering = u_optimal[1]

        # Convert to CARLA control
        control = carla.VehicleControl()

        # Convert acceleration to throttle/brake
        if acceleration >= 0:
            control.throttle = min(
                acceleration / 3.0, 1.0
            )  # Assuming max accel of 3 m/s^2
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(
                -acceleration / 5.0, 1.0
            )  # Assuming max decel of 5 m/s^2

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


def generate_overtake_waypoints(
    carla_manager,
    vehicle,
    direction="left",
    distance_current_lane=10.0,
    lane_change_step=2.0,
    overtake_distance=50.0,
    merge_distance=10.0,
):
    """
    Generate a list of waypoints for an overtaking maneuver.

    Args:
        carla_manager: Instance of CarlaManager containing the map.
        vehicle: The vehicle actor (rear vehicle).
        direction (str): "left" or "right" lane change direction.
        distance_current_lane (float): Distance to follow in current lane before lane change.
        lane_change_step (float): Interval (in meters) to sample waypoints.
        overtake_distance (float): Distance to follow in the adjacent lane.
        merge_distance (float): Distance to follow in merging lane after overtaking.

    Returns:
        List of carla.Waypoint objects forming the overtaking trajectory.
    """
    _map = carla_manager.map
    waypoints = []

    # 1. Start in current lane.
    current_wp = _map.get_waypoint(vehicle.get_location(), project_to_road=True)
    waypoints.append(current_wp)

    # Follow current lane for distance_current_lane
    traveled = 0.0
    last_wp = current_wp
    while traveled < distance_current_lane:
        next_wps = last_wp.next(lane_change_step)
        if not next_wps:
            break
        next_wp = next_wps[0]
        traveled += next_wp.transform.location.distance(last_wp.transform.location)
        waypoints.append(next_wp)
        last_wp = next_wp

    # 2. Change lane: get the adjacent lane from the last waypoint
    if direction == "left":
        adjacent_wp = last_wp.get_left_lane()
    else:
        adjacent_wp = last_wp.get_right_lane()

    if adjacent_wp is None:
        print("No adjacent lane available in direction", direction)
        return waypoints

    waypoints.append(adjacent_wp)
    last_wp = adjacent_wp

    # 3. Follow adjacent lane for overtaking distance.
    traveled = 0.0
    while traveled < overtake_distance:
        next_wps = last_wp.next(lane_change_step)
        if not next_wps:
            break
        next_wp = next_wps[0]
        traveled += next_wp.transform.location.distance(last_wp.transform.location)
        waypoints.append(next_wp)
        last_wp = next_wp

    # 4. Merge back to the original lane.
    if direction == "left":
        merging_wp = last_wp.get_right_lane()
    else:
        merging_wp = last_wp.get_left_lane()

    if merging_wp is not None:
        waypoints.append(merging_wp)
        last_wp = merging_wp
        traveled = 0.0
        while traveled < merge_distance:
            next_wps = last_wp.next(lane_change_step)
            if not next_wps:
                break
            next_wp = next_wps[0]
            traveled += next_wp.transform.location.distance(last_wp.transform.location)
            waypoints.append(next_wp)
            last_wp = next_wp

    return waypoints


if __name__ == "__main__":
    carla_manager = CarlaManager()
    print("CarlaManager is created")

    # Spawn preceding vehicle
    preceding_vehicle = carla_manager.spawn_vehicle(
        "vehicle.tesla.model3", SPAWN_LOCATION
    )
    preceding_vehicle.set_autopilot(False)
    time.sleep(1)  # allow the vehicle to spawn

    # Basic control for preceding vehicle
    agent = BasicAgent(preceding_vehicle, target_speed=10)

    # Set destination for the preceding vehicle
    current_location = preceding_vehicle.get_location()
    current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)
    next_wps = current_wp.next(100.0)  # 100 meters ahead

    if next_wps:  # if there is a waypoint ahead
        destination = next_wps[0].transform.location
    else:
        destination = current_location
    agent.set_destination(destination)

    # Spawn ego vehicle
    SPAWN_LOCATION[0] += 10  # 10 meters behind preceding vehicle
    ego_vehicle = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
    time.sleep(1)  # allow the vehicle to spawn

    # Create MPC controller for ego vehicle
    mpc_controller = MPCController(horizon=10, dt=0.1)

    # Target speed for ego vehicle (km/h)
    target_speed = 20

    # Main control loop
    try:
        while True:
            # Control preceding vehicle
            control_cmd = agent.run_step()
            preceding_vehicle.apply_control(control_cmd)

            # Control ego vehicle with MPC
            ego_control = mpc_controller.run_step(
                ego_vehicle, preceding_vehicle, target_speed / 3.6
            )  # Convert to m/s
            ego_vehicle.apply_control(ego_control)

            # Optional: Print current status
            ego_loc = ego_vehicle.get_location()
            lead_loc = preceding_vehicle.get_location()
            distance = np.sqrt(
                (ego_loc.x - lead_loc.x) ** 2 + (ego_loc.y - lead_loc.y) ** 2
            )
            print(f"Distance to lead vehicle: {distance:.2f} m")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Simulation terminated by user")
    finally:
        # Clean up
        if preceding_vehicle and ego_vehicle:
            preceding_vehicle.destroy()
            ego_vehicle.destroy()
        print("Vehicles destroyed")
