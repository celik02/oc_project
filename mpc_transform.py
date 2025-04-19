import time
import numpy as np

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.basic_agent import BasicAgent
from vehicle import Vehicle  # Import the Vehicle class
from carla_setup import get_next_waypoint_from_list
import traceback
import casadi as ca
import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
FORMAT = "[%(asctime)s.%(msecs)03d %(filename)15s:%(lineno)3s - %(funcName)17s() ] %(levelname)s %(message)s"
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True, format=FORMAT, datefmt='%H:%M:%S')
dt = 0.1  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

# synchronous_mode will make the simulation predictable
synchronous_mode = True


def frenet_vehicle_dynamics_casadi(x, u, dt):
    """
    Modified CasADi implementation of vehicle dynamics in Frenet coordinates.

    State vector x = [s, d, mu, v]  # Reduced state vector
    Control vector u = [a, delta]   # Direct control of acceleration and steering
    """
    # Vehicle parameters
    lr = 1.5    # Distance from CG to rear axle
    lf = 1.375  # Distance from CG to front axle
    # lr = 2.3
    # lf = 2.4
    L = lr + lf  # Wheelbase

    # Extract states
    s = x[0]      # Path position
    d = x[1]      # Lateral deviation
    mu = x[2]     # Heading error
    v = x[3]      # Velocity

    # Extract control inputs - now direct acceleration and steering
    a = u[0]      # Acceleration control
    delta = u[1]  # Steering angle control

    # Compute intermediate variables
    # Slip angle based on bicycle model
    beta = ca.arctan(lr/L * ca.tan(delta))

    # Curvature - set to 0 for straight road
    # try:
    #     kappa = 1 / delta
    # except FloatingPointError as e:
    #     kappa = 0
    #     print(e)
    kappa = 0

    # Compute dynamics based on equation (29) but with direct controls
    s_dot = v * ca.cos(mu + beta) / (1 - d * kappa)
    d_dot = v * ca.sin(mu + beta)
    mu_dot = v/lr * ca.sin(beta) - kappa * v * ca.cos(mu + beta) / (1 - d * kappa)
    v_dot = a  # Acceleration is directly controlled

    # Euler integration
    s_next = s + s_dot * dt
    d_next = d + d_dot * dt
    mu_next = mu + mu_dot * dt
    v_next = v + v_dot * dt

    # Return updated state vector
    return ca.vertcat(s_next, d_next, mu_next, v_next)


class FrenetMPCController:
    def __init__(self, horizon=10, dt=0.1, carla_manager=None):
        self.carla_manager = carla_manager

        self.horizon = horizon
        self.dt = dt

        # Control constraints
        self.max_accel = 3.0
        self.min_accel = -3.0
        self.max_steer = np.pi / 4  # Max steering allowed (rad)
        self.min_steer = -np.pi / 4  # Min steering allowed (rad)

        # Cost function weights
        self.w_progress = 30.0    # Forward progress reward
        self.w_lane = 0.0        # Lane centering reward
        self.w_speed = 0.0       # Target speed reward
        self.w_accel = 0.0       # Acceleration minimization
        self.w_steer_rate = 0.0  # Steering rate minimization
        self.w_jerk = 0.0        # Jerk minimization

        # Lane chang    e parameters
        self.lane_width = 1.75    # Standard lane width in meters
        self.current_lane = 0    # Middle lane (0), left lane (-1), right lane (1)
        self.target_lane = 0     # Initially target the current lane

        # Initialize CasADi solver (will be created the first time we run)
        self.solver = None
        self.casadi_setup_done = False

    def run_step(self, ego_vehicle, preceding_vehicle, target_speed, next_waypoint):
        """
        Execute one step of MPC control using Frenet coordinates

        Args:
            ego_vehicle: Vehicle class instance for ego vehicle
            preceding_vehicle: Vehicle class instance for preceding vehicle
            target_speed: Desired speed in m/s
            next_waypoint: Waypoint for Frenet coordinate calculation

        Returns:
            carla.VehicleControl object
        """

        # Get Frenet states for ego vehicle
        ego_frenet, waypoint = ego_vehicle.get_frenet_states(next_waypoint)

        # Extract ego states - now using the full state vector from Vehicle class
        s0, d0, mu0, v0, _, _ = ego_frenet

        # Then update x0 with the stabilized heading
        x0 = np.array([s0, d0, mu0, v0])

        print(f"d = {d0:.2f}")

        # Get Frenet states for preceding vehicle using the same waypoint reference
        lead_frenet, _ = preceding_vehicle.get_frenet_states(next_waypoint)
        lead_s, lead_d, _, lead_speed, _, _ = lead_frenet

        # Print diagnostic information about the Frenet coordinates
        # print(f"Ego Frenet: s={s0:.2f}, d={d0:.2f}, v={v0:.2f}")
        # print(f"Lead Frenet: s={lead_s:.2f}, d={lead_d:.2f}, v={lead_speed:.2f}")

        # Calculate relative distance in world coordinates
        ego_location = ego_vehicle.actor.get_location()
        lead_location = preceding_vehicle.actor.get_location()

        # Project the relative position onto the Frenet frame
        relative_position = carla.Location(
            lead_location.x - ego_location.x,
            lead_location.y - ego_location.y,
            lead_location.z - ego_location.z
        )

        # Get the road direction at the waypoint (tangent to the road)
        waypoint_yaw = np.radians(waypoint.transform.rotation.yaw)
        road_direction = np.array([np.cos(waypoint_yaw), np.sin(waypoint_yaw)])

        # Project the relative position onto the road direction for longitudinal distance (s)
        relative_s = relative_position.x * road_direction[0] + relative_position.y * road_direction[1]

        # Compare Frenet-derived distance with actual world distance
        # print(f"World-space relative s: {relative_s:.2f}")

        # Calculate desired lateral position based on target lane
        d_desired = self.target_lane * self.lane_width
        # print(f"Target lane: {self.target_lane}, d_desired: {d_desired:.2f}")

        # Generate lead vehicle prediction in Frenet coordinates
        lead_vehicle_pred = self.get_lead_vehicle_prediction(relative_s, lead_d, lead_speed)

        # Solve optimization problem with CasADi
        if not self.casadi_setup_done:
            self.setup_casadi_solver()
            self.casadi_setup_done = True

        # When calling solve_with_casadi, pass the updated state vector
        u_optimal = self.solve_with_casadi(x0, lead_vehicle_pred, target_speed, d_desired)

        # print("Control action", u_optimal[0])
        # Convert to CARLA control, adapting to use acceleration and steering rate
        control = self.convert_to_control(u_optimal[0])  # Pass current steering angle

        return control

    def setup_casadi_solver(self):
        """
        Setup the CasADi solver for MPC optimization with adapted state representation.
        """
        print("Setting up CasADi solver...")

        # State variables (symbolic) - now 6 dimensions
        x = ca.SX.sym('x', 4)  # [s, d, mu, v]

        # Control variables (symbolic) - now [accel, steering]
        u = ca.SX.sym('u', 2)  # [accel, steering]

        # Define parameters that will change each time step
        # Initial state
        x0 = ca.SX.sym('x0', 4)

        # Lead vehicle predicted positions (for all steps in horizon)
        lead_s_pred = ca.SX.sym('lead_s_pred', self.horizon)
        lead_d_pred = ca.SX.sym('lead_d_pred', self.horizon)

        # Target speed and desired lateral position
        target_speed = ca.SX.sym('target_speed', 1)
        d_desired = ca.SX.sym('d_desired', 1)

        # Define the dynamics function with adapted model
        dynamics_func = ca.Function('dynamics', [x, u], [frenet_vehicle_dynamics_casadi(x, u, self.dt)])

        # Initialize the cost function and constraints
        obj = 0
        g = []  # Constraints
        lbg = []  # Lower bounds for constraints
        ubg = []  # Upper bounds for constraints

        # State bounds for lane boundaries (simplified from the lane_boundary_constraint)
        lane_boundaries = [-self.lane_width*1.5, -self.lane_width*0.5, self.lane_width*0.5, self.lane_width*1.5]
        safe_margin = 0.3  # meters

        # Setup decision variables
        opt_vars = []
        opt_vars_lb = []
        opt_vars_ub = []

        # Initial conditions
        xk = x0

        # Define the optimization variables for each step
        for k in range(self.horizon):
            # Control at the current step
            uk = ca.SX.sym(f'u_{k}', 2)
            opt_vars.append(uk)
            opt_vars_lb.extend([self.min_accel, self.min_steer])
            opt_vars_ub.extend([self.max_accel, self.max_steer])

            # In the for k in range(self.horizon) loop:

            # 1. Progress reward (using distance along the lane)
            obj += self.w_progress * (xk[0]-30)**2  # Forward progress reward - bigger s is better

            # 2. Lane centering
            obj += self.w_lane * (xk[1] - d_desired)**2

            # 3. Target speed
            obj += self.w_speed * (target_speed - xk[3])**2

            # 4. Control effort - now penalizing acceleration and steering rate
            obj += self.w_accel * uk[0]**2  # Acceleration
            # obj += self.w_steer * uk[1]**2  # Steering angle
            # # penalize steering rate
            # obj += self.w_steer * (uk[1] - xk[4])**2  # Steering rate

            # 5. Jerk minimization - adapt if needed based on new control definition
            if k > 0:
                prev_uk = opt_vars[-3]  # Two items back in the list
                obj += self.w_jerk * ((uk[0] - prev_uk[0]) / self.dt)**2  # Jerk penalty (accel rate)
                obj += self.w_steer_rate * ((uk[1] - prev_uk[1]) / self.dt)**2  # Jerk penalty (accel rate)

            # Lead vehicle position at this step
            lead_s_k = lead_s_pred[k]
            lead_d_k = lead_d_pred[k]
            logger.debug(f"Lead vehicle position at step {k}: s={lead_s_k}, d={lead_d_k}")
            # Collision avoidance constraint (ellipsoid)
            # Define ellipsoid parameters
            a = 6.0  # Longitudinal semi-axis (front-back)
            b = 2.5  # Lateral semi-axis (side-to-side)

            # Calculate distances in Frenet coordinates
            ds = (23) - xk[0]  # Longitudinal distance
            dd = 0 - xk[1]  # Lateral distance
            logger.debug(f"Lead vehicle distance at step {k}: ds={ds}, dd={dd}")

            # Add collision avoidance constraint
            # We want (ds/a)² + (dd/b)² >= 1 (outside the ellipsoid)
            g.append((ds/a)**2 + (dd/b)**2)
            lbg.append(1.0)  # Lower bound: must be greater than 1 to stay outside
            ubg.append(ca.inf)  # Upper bound: infinity

            # # Lane boundary constraints
            # # Calculate distance to each lane boundary
            # for boundary in lane_boundaries:
            #     margin = ca.fabs(xk[1] - boundary)
            #     # Add a constraint that margin should be greater than safe_margin
            #     g.append(margin)
            #     lbg.append(safe_margin)
            #     ubg.append(ca.inf)

            # State propagation - get the next state
            xk_next = dynamics_func(xk, uk)

            # Add dynamics constraints (next state must follow the dynamics model)
            if k < self.horizon - 1:
                # When creating opt_vars:
                # For each state variable, we need to define a new symbolic variable for the next state
                xk_next_sym = ca.SX.sym(f'x_{k+1}', 4)
                opt_vars.append(xk_next_sym)

                # No bounds on states except for practical limits, adjusted for 6D state
                opt_vars_lb.extend([-ca.inf, -ca.inf, -ca.inf, 0])  # v >= 0
                opt_vars_ub.extend([ca.inf, ca.inf, ca.inf, 40])

                # Add the dynamics constraint: next state must equal dynamics model
                g.append(xk_next_sym - xk_next)
                lbg.extend([0, 0, 0, 0])  # Equality constraints for 4D state
                ubg.extend([0, 0, 0, 0])

                # Lane constraint with slack variable
                slack_lane = ca.SX.sym(f'slack_lane_{k}', 1)
                opt_vars.append(slack_lane)

                # Slack variable should be non-negative but can be large if needed
                opt_vars_lb.extend([0])
                opt_vars_ub.extend([5])  # Upper bound can be tuned
                g.append(xk_next_sym[1] - d_desired + slack_lane)  # Lower bound slack
                lbg.append(-self.lane_width + 1.2)  # Lower bound is -lane_width
                ubg.append(ca.inf)  # Upper bound is 0

                # Right lane boundary is strict (no slack)
                g.append(xk_next_sym[1] - d_desired)  # No slack for right lane
                lbg.append(-ca.inf)
                ubg.append(self.lane_width - 1.2)

                slack_weight = 280.0  # Tunable parameter
                obj += slack_weight * slack_lane**2

                # Update state for next iteration
                xk = xk_next_sym

        # Pack all optimization variables into a single vector
        opt_vars = ca.vertcat(*opt_vars)

        # Create the NLP problem
        nlp = {
            'x': opt_vars,
            'f': obj,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, lead_s_pred, lead_d_pred, target_speed, d_desired)
        }

        # Configure the solver
        opts = {
            'ipopt': {
                'print_level': 0,         # 0 for no output
                'max_iter': 10000,          # Maximum number of iterations
                'acceptable_tol': 1e-4,   # Tolerance
                'warm_start_init_point': 'yes',  # Use warm starting
            },
            'print_time': False
        }

        # Store constraint bounds when creating the problem
        self.lbg = lbg  # Store lower bounds for constraints
        self.ubg = ubg  # Store upper bounds for constraints

        # Create the solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Store information about the problem dimensions
        self.num_states = 4
        self.num_controls = 2
        self.opt_vars_size = opt_vars.size1()

        print("CasADi solver setup complete!")

    def solve_with_casadi(self, x0, lead_vehicle_pred, target_speed, d_desired):
        """
        Solve the MPC problem using the CasADi solver

        Args:
            x0: Initial state [s, d, mu, v, a]
            lead_vehicle_pred: Predicted lead vehicle positions [(s,d), ...]
            target_speed: Desired speed in m/s
            d_desired: Desired lateral position

        Returns:
            Optimal control sequence
        """
        try:
            # Extract lead vehicle prediction data
            lead_s_pred = np.array([pred[0] for pred in lead_vehicle_pred])
            lead_d_pred = np.array([pred[1] for pred in lead_vehicle_pred])
            # print(f"Lead vehicle predictions: {lead_s_pred}, {lead_d_pred}")
            # Define initial guess (all zeros)
            x_init = np.zeros(self.opt_vars_size)

            # Pack parameters with 6D state
            p = np.concatenate([
                x0.flatten(),  # Now 6D
                lead_s_pred.flatten(),
                lead_d_pred.flatten(),
                [target_speed],
                [d_desired]
            ])

            # Lower and upper bounds for variables, adjusted for 6D state
            lbx = []
            ubx = []

            # For each time step, add control bounds
            for _ in range(self.horizon):
                lbx.extend([self.min_accel, self.min_steer])
                ubx.extend([self.max_accel, self.max_steer])

                # If not the last step, add state bounds
                if _ < self.horizon - 1:
                    # Bounds for 6D state
                    lbx.extend([-np.inf, -np.inf, -np.inf, 0])
                    ubx.extend([np.inf, np.inf, np.inf, 40])

                    # for slack variables
                    lbx.extend([0])  # Non-negative slack
                    ubx.extend([10])  # Upper bound for slack

            # Solve the optimization problem with stored constraint bounds
            sol = self.solver(
                x0=x_init,
                lbx=lbx,
                ubx=ubx,
                lbg=self.lbg,  # Use stored lower bounds
                ubg=self.ubg,  # Use stored upper bounds
                p=p
            )

            stats = self.solver.stats()
            if stats['return_status'] in ['Infeasible_Problem_Detected', 'Maximum_CpuTime_Exceeded',
                                      'Maximum_Iterations_Exceeded', 'Restoration_Failed']:
                print(f"MPC problem is INFEASIBLE! Status: {stats['return_status']}")
                print(f"Iteration count: {stats['iter_count']}")

            # Extract the optimal control sequence
            x_opt = sol['x'].full().flatten()

            slack_values = []
            for i in range(self.horizon-1):  # Only horizon-1 slack variables
                # Each timestep has controls + states + slack (except last)
                variables_per_timestep = self.num_controls + self.num_states + 1

                # First get to the right timestep, then skip controls and states
                slack_idx = i * variables_per_timestep + self.num_controls + self.num_states

                # Now you're at the slack variable
                slack_value = x_opt[slack_idx]
                slack_values.append(slack_value)
                print(f"Slack variable at step {i}: {slack_value:.4f}")

            # input('Press Enter to continue...')

            # Reshape to get control sequence - adjusted for the 6+2 dimensionality
            u_optimal = []
            for i in range(self.horizon):
                idx = i * (self.num_controls + self.num_states) if i < self.horizon - 1 else i * 2
                u_i = x_opt[idx:idx+2]
                u_optimal.append(u_i)

            return np.array(u_optimal)

        except Exception as e:
            print(f"CasADi optimization failed: {e}")
            traceback.print_exc()

    def get_lead_vehicle_prediction(self, lead_s, lead_d, lead_speed):
        """
        Predict lead vehicle trajectory in Frenet coordinates (constant velocity model)
        """
        predictions = []
        s_lead = lead_s
        d_lead = lead_d

        for i in range(self.horizon):
            # Assume constant velocity in s direction, constant d position
            s_lead = s_lead + lead_speed * self.dt
            predictions.append((s_lead, d_lead))
            logger.debug(f"Lead vehicle prediction at step {i}: s={s_lead}, d={d_lead}")

        return predictions

    def convert_to_control(self, u_optimal):
        """
        Convert MPC control output to CARLA control

        Args:
            u_optimal: Optimal control input [acceleration, steering]
            current_steering: Current steering angle

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
            control.throttle = min(acceleration / self.max_accel, 1.0)  # Assuming max accel of 3 m/s^2
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-acceleration / self.max_accel, 1.0)  # Assuming max decel of 3 m/s^2

        # Apply steering rate by converting to steering command
        # max steering angle is 69.99999237060547 degrees
        # max steering from degrees to radians
        max_steering_angle = np.radians(69.99999237060547)
        # Note: CARLA expects steering in [-1, 1] range
        control.steer = steering
        # map the steering from radians to [-1, 1] range
        control.steer = control.steer / max_steering_angle
        control.steer = np.clip(control.steer, -1.0, 1.0)  # Clip to valid range

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


def run_simulation_with_casadi():
    """
    Run the main simulation with CasADi-based MPC controller
    """
    print("Starting simulation with CasADi MPC controller...")

    # Create CarlaManager instance
    carla_manager = CarlaManager()
    # carla_manager.restart_world()  # Ensure the world is clean before starting
    print("CarlaManager is created")

    preceding_vehicle_actor = None
    ego_vehicle_actor = None

    try:
        # Spawn preceding vehicle (stationary)
        preceding_vehicle_actor = carla_manager.spawn_vehicle(
            "vehicle.tesla.model3", SPAWN_LOCATION
        )

        preceding_vehicle_actor.set_autopilot(False)
        time.sleep(1)  # allow the vehicle to spawn

        # Wrap with Vehicle class
        preceding_vehicle = Vehicle(preceding_vehicle_actor)

        # Basic control for preceding vehicle (zero speed to keep it stationary)
        agent = BasicAgent(preceding_vehicle_actor, target_speed=0)

        # Set destination for the preceding vehicle
        current_location = preceding_vehicle_actor.get_location()
        current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)
        next_wps = current_wp.next(30.0)  # 100 meters ahead

        if next_wps:  # if there is a waypoint ahead
            destination = next_wps[0].transform.location
        else:
            destination = current_location
        agent.set_destination(destination)

        # Spawn ego vehicle
        spawn_loc_copy = SPAWN_LOCATION.copy()  # Make a copy to avoid modifying the original
        spawn_loc_copy[0] += 20  # 20 meters behind preceding vehicle
        ego_vehicle_actor = carla_manager.spawn_vehicle("vehicle.tesla.model3", spawn_loc_copy)
        if synchronous_mode:
            carla_manager.world.tick()  # Synchronize the world to ensure the vehicle is spawned
        else:
            time.sleep(1)  # allow the vehicle to spawn

        # Wrap with Vehicle class
        ego_vehicle = Vehicle(ego_vehicle_actor)

        # Create Frenet MPC controller with CasADi for ego vehicle
        mpc_controller = FrenetMPCController(horizon=30, dt=dt, carla_manager=carla_manager)

        # Target speed for ego vehicle (m/s)
        target_speed = 15 / 3.6  # Convert from km/h to m/s

        # Main control loop
        try:
            current_wp_index = 0
            current_location = ego_vehicle_actor.get_location()
            next_waypoint = carla_manager.map.get_waypoint(current_location, project_to_road=True)
            # create a list of waypoints each 0.1 meters apart until 200 m ahead
            list_of_waypoints = []
            for i in range(200):
                next_waypoint = next_waypoint.next(0.5)[0]
                list_of_waypoints.append(next_waypoint)
                # show the next waypoint on the carla map
                carla_manager.world.debug.draw_string(
                    next_waypoint.transform.location,
                    "O",
                    draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=0),
                    life_time=120.0,
                    persistent_lines=True,
                )

            print("Starting simulation with CasADi MPC. Press Ctrl+C to exit...")

            while True:
                # Get next waypoint for Frenet coordinates
                next_waypoint, current_wp_index = get_next_waypoint_from_list(list_of_waypoints, ego_vehicle_actor, current_wp_index, threshold=5.0)

                # # Get next waypoint for Frenet coordinates
                # current_location = ego_vehicle_actor.get_location()
                # next_waypoint = carla_manager.map.get_waypoint(current_location, project_to_road=True)

                carla_manager.world.debug.draw_string(
                    next_waypoint.transform.location,
                    "O",
                    draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=120.0,
                    persistent_lines=True,
                )

                # Control preceding vehicle (keep stationary)
                control_cmd = agent.run_step()
                preceding_vehicle_actor.apply_control(control_cmd)

                # Control ego vehicle with CasADi Frenet MPC
                ego_control = mpc_controller.run_step(
                    ego_vehicle, preceding_vehicle, target_speed, next_waypoint
                )

                # apply a control to the left just driving the car into the left fence
                # ego_control.steer = 0.5  # Adjust steering to the left

                print(f"Control action: {ego_control.throttle:.2f}, {ego_control.steer:.2f}")

                ego_vehicle_actor.apply_control(ego_control)

                # Print current status
                ego_frenet, _ = ego_vehicle.get_frenet_states(next_waypoint)
                lead_frenet, _ = preceding_vehicle.get_frenet_states(next_waypoint)

                # Calculate real-world distance between vehicles
                ego_loc = ego_vehicle_actor.get_location()
                lead_loc = preceding_vehicle_actor.get_location()

                # time.sleep(dt - 0.02)
                # # Add this line before or instead of time.sleep()
                carla_manager.world.tick()

        except KeyboardInterrupt:
            print("\nSimulation terminated by user")

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up vehicles first (if they exist and are alive)
        if preceding_vehicle_actor and preceding_vehicle_actor.is_alive:
            try:
                preceding_vehicle_actor.destroy()
                print("Lead vehicle destroyed")
            except Exception as e:
                print(f"Error destroying lead vehicle: {e}")

        if ego_vehicle_actor and ego_vehicle_actor.is_alive:
            try:
                ego_vehicle_actor.destroy()
                print("Ego vehicle destroyed")
            except Exception as e:
                print(f"Error destroying ego vehicle: {e}")

        # Reset the world after cleaning up specific vehicles
        print("Resetting the Carla world...")
        # carla_manager.restart_world()
        if synchronous_mode:
            settings = carla_manager.world.get_settings()
            settings.synchronous_mode = False
            carla_manager.world.apply_settings(settings)


if __name__ == "__main__":
    run_simulation_with_casadi()
