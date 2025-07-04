import time
import numpy as np

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.basic_agent import BasicAgent
from vehicle import Vehicle  # Import the Vehicle class
from pid import get_next_waypoint_from_list
import traceback
import casadi as ca
import logging
import sys
from hydra import initialize, compose
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
FORMAT = "[%(asctime)s.%(msecs)03d %(filename)15s:%(lineno)3s - %(funcName)17s() ] %(levelname)s %(message)s"
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True, format=FORMAT, datefmt='%H:%M:%S')
dt = 0.1  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

PRECEDING_SPEED = 10 / 3.6  # m/s, speed of the preceding vehicle

# synchronous_mode will make the simulation predictable
synchronous_mode = True
EKF = False
measurement_dim = 2  # Measurement dimension
# P0 = np.array([1., 1., 1., 1.])  # Initial covariance
P0 = np.array([1.5, 1.5, 1.5, 1.])

# Q = np.eye(state_dim) * 0.1  # Process noise covariance
Q = np.diag([0.1, 0.1, 0.005, 0.5])  # Process noise covariance

R = np.eye(measurement_dim) * 2  # Measurement noise covariance
# R = np.diag([1, 1])  # Measurement noise covariance


def bicycle_vehicle_dynamics_casadi(x, u, dt):
    """
    CasADi implementation of bicycle kinematic model in global coordinates.
    State vector x = [x, y, psi, v, a]  # Position, heading, velocity, acceleration
    Control vector u = [delta, u_a]      # Steering angle, acceleration command
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
    x_dot = v * ca.cos(psi)              # Rate of change of x-position
    y_dot = v * ca.sin(psi)              # Rate of change of y-position
    psi_dot = v * ca.tan(delta) / L      # Rate of change of heading
    v_dot = a                            # Rate of change of velocity equals acceleration
    a_dot = (u_a - a) / tau_a            # Acceleration dynamics with time constant

    # Euler integration to compute next state
    x_next = x_pos + x_dot * dt
    y_next = y_pos + y_dot * dt
    psi_next = psi + psi_dot * dt
    v_next = v + v_dot * dt
    a_next = a + a_dot * dt

    # Return updated state vector
    return ca.vertcat(x_next, y_next, psi_next, v_next, a_next)


class BicycleMPCController:
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
        self.w_path = 30.0      # Path following reward
        self.w_speed = 20.0     # Target speed reward
        self.w_accel = 1.0      # Acceleration minimization
        self.w_steer = 1.0      # Steering minimization
        self.w_jerk = 5.0       # Jerk minimization (change in acceleration)
        self.w_steer_rate = 10.0  # Steering rate minimization

        # Initialize CasADi solver
        self.solver = None
        self.casadi_setup_done = False

    def run_step(self, ego_vehicle: Vehicle, preceding_vehicle, target_speed):
        """
        Execute one step of MPC control using bicycle model

        Args:
            ego_vehicle: Vehicle class instance for ego vehicle
            preceding_vehicle: Vehicle class instance for preceding vehicle
            target_speed: Desired speed in m/s

        Returns:
            carla.VehicleControl object
        """
        # clear all drawned objects in CARLA
        if self.carla_manager:
            self.carla_manager.world.debug.draw_string(
                carla.Location(x=0, y=0, z=0),
                "",
                draw_shadow=False,
                color=carla.Color(r=0, g=0, b=0),
                life_time=120.0,
                persistent_lines=True,
            )

        # Get ego vehicle state
        if not EKF:
            x0 = ego_vehicle.get_vehicle_state()
        else:
            x0 = ego_vehicle.get_vehicle_state_kalman()  # Get the latest EKF state

        # Get preceding vehicle position in ego coordinates
        preceding_location = preceding_vehicle.actor.get_location()
        preceding_pos_in_ego = ego_vehicle.world_to_ego_coordinates(preceding_location)

        # Generate preceding vehicle prediction
        # TODO: Right now we are assuming we know the initial speed and the orientation of the preceding vehicle
        preceding_vehicle_pred = self.predict_vehicle_trajectory(
            preceding_pos_in_ego, PRECEDING_SPEED)


        # Generate reference path in ego coordinates
        reference_path = self.generate_reference_path(x0[0])

        # Visualize reference path in CARLA but after reverting it back to the # world coordinates
        if self.carla_manager:
            # Convert reference path to world coordinates while padding the z co-ordinates with zero
            reference_path_world = [
                ego_vehicle.ego_to_world_coordinates((point[0], point[1], 0.0))
                for point in reference_path
            ]
            # Draw the reference path in CARLA
            # convert to carla location objects
            reference_path_world = [
                carla.Location(x=point[0], y=point[1], z=point[2])
                for point in reference_path_world
            ]
            for point in reference_path_world:
                self.carla_manager.world.debug.draw_string(
                    point,
                    "O",
                    draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=0),
                    life_time=120.0,
                    persistent_lines=True,
                )

        # Initialize or update CasADi solver
        if not self.casadi_setup_done:
            self.setup_casadi_solver()
            self.casadi_setup_done = True

        # Solve optimization problem
        u_optimal = self.solve_with_casadi(
            x0, preceding_vehicle_pred, target_speed, reference_path)

        # Convert to CARLA control
        control = self.convert_to_control(u_optimal[0])

        return control, u_optimal

    def predict_vehicle_trajectory(self, position, speed, heading=0.0):
        """
        Predict trajectory using constant velocity model
        """
        predictions = []
        x, y = position[0], position[1]
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        for i in range(self.horizon):
            # Simple constant velocity prediction
            x += vx * self.dt
            y += vy * self.dt
            predictions.append((x, y))

        return predictions

    def generate_reference_path(self, x_start):
        """
        Generate a reference path in ego coordinates
        (straight line ahead by default)
        """
        path = []
        for i in range(self.horizon):
            # Points spaced at 1m intervals directly ahead
            x = (i + 1) * 1.0 + x_start  # 1m ahead for each step
            y = 0.0 - i*0.1  # Center of lane
            path.append((x, y))

        return path

    def setup_casadi_solver(self):
        """
        Setup the CasADi solver for MPC optimization with bicycle model.
        """
        print("Setting up CasADi solver...")

        # State variables (symbolic)
        x = ca.SX.sym('x', 5)  # [x, y, psi, v, a]

        # Control variables (symbolic)
        u = ca.SX.sym('u', 2)  # [u_a, delta]

        # Parameters
        x0 = ca.SX.sym('x0', 5)  # Initial state
        preceding_x = ca.SX.sym('preceding_x', self.horizon)  # Preceding vehicle x positions
        preceding_y = ca.SX.sym('preceding_y', self.horizon)  # Preceding vehicle y positions
        ref_path_x = ca.SX.sym('ref_path_x', self.horizon)    # Reference path x positions
        ref_path_y = ca.SX.sym('ref_path_y', self.horizon)    # Reference path y positions
        target_v = ca.SX.sym('target_v', 1)  # Target velocity

        # Dynamics function
        dynamics_func = ca.Function('dynamics', [x, u], [bicycle_vehicle_dynamics_casadi(x, u, self.dt)])

        # Initialize optimization problem
        obj = 0  # Objective function
        g = []   # Constraints
        lbg = [] # Lower bounds for constraints
        ubg = [] # Upper bounds for constraints

        # Variables
        opt_vars = []
        opt_vars_lb = []
        opt_vars_ub = []

        # Initial state
        xk = x0

        # Control trajectory optimization
        for k in range(self.horizon):
            # Control variables at step k
            uk = ca.SX.sym(f'u_{k}', 2)
            opt_vars.append(uk)
            opt_vars_lb.extend([self.min_accel, self.min_steer])
            opt_vars_ub.extend([self.max_accel, self.max_steer])

            # Path following cost
            path_error = (xk[0] - ref_path_x[k])**2 + (xk[1] - ref_path_y[k])**2
            obj += self.w_path * path_error

            # Speed tracking cost
            speed_error = (xk[3] - target_v)**2
            obj += self.w_speed * speed_error

            # Control effort costs
            obj += self.w_accel * uk[0]**2  # Acceleration command
            obj += self.w_steer * uk[1]**2  # Steering angle

            # Control rate costs (if not first step)
            if k > 0:
                prev_uk = opt_vars[-3]  # Previous control
                accel_rate = (uk[0] - prev_uk[0])/self.dt
                steer_rate = (uk[1] - prev_uk[1])/self.dt
                obj += self.w_jerk * accel_rate**2
                obj += self.w_steer_rate * steer_rate**2

            # Collision avoidance with preceding vehicle
            # Define ellipsoid safety zone
            a = 7.0  # Longitudinal semi-axis
            b = 3.5  # Lateral semi-axis

            # Original ellipsoidal distance calculation
            dx = xk[0] - preceding_x[k]
            dy = xk[1] - preceding_y[k]
            ellipsoid_distance = (dx/a)**2 + (dy/b)**2

            # When ego is behind: constraint is "ellipsoid_distance >= 1.0"
            # When ego is ahead by 3.0: constraint is "0 >= 0" (automatically satisfied)
            safety_constraint = ca.if_else(dx < 1.0,
                                        ellipsoid_distance - 1.0,  # When behind: must stay outside ellipsoid
                                        0.0)                       # When ahead: constraint inactive (0 >= 0)

            g.append(safety_constraint)
            lbg.append(0.0)  # Constraint must be >= 0
            ubg.append(ca.inf)


            # State propagation
            xk_next = dynamics_func(xk, uk)

            # State constraints for next step
            if k < self.horizon - 1:
                # New state variables
                xk_next_var = ca.SX.sym(f'x_{k+1}', 5)
                opt_vars.append(xk_next_var)

                # State bounds
                opt_vars_lb.extend([-ca.inf, -ca.inf, -ca.inf, 0, self.min_accel])  # v >= 0, a >= -5
                opt_vars_ub.extend([ca.inf, ca.inf, ca.inf, 40, self.max_accel])     #self.max_accel<= 40, a <= 5

                # Dynamics constraints (next state follows dynamics model)
                g.append(xk_next_var - xk_next)
                lbg.extend([0, 0, 0, 0, 0])  # Equality constraints
                ubg.extend([0, 0, 0, 0, 0])

                # Update for next iteration
                xk = xk_next_var

        # Create optimization vector
        opt_vars = ca.vertcat(*opt_vars)

        # NLP problem formulation
        nlp = {
            'x': opt_vars,
            'f': obj,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, preceding_x, preceding_y, ref_path_x, ref_path_y, target_v)
        }

        # Solver options
        opts = {
                'ipopt': {
                    'print_level': 0,                # Minimal output
                    'max_iter': 500,                 # Reduced from 500 for speed
                    'acceptable_tol': 1e-4,          # Slightly relaxed tolerance
                    'warm_start_init_point': 'yes',
                },
                'print_time': False
            }

        # Store constraint bounds
        self.lbg = lbg
        self.ubg = ubg

        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Store problem dimensions
        self.num_states = 5
        self.num_controls = 2
        self.opt_vars_size = opt_vars.size1()

        print("CasADi solver setup complete!")

    def solve_with_casadi(self, x0, preceding_vehicle_pred, target_speed, reference_path):
        """
        Solve the MPC optimization problem
        """
        try:
            # Extract preceding vehicle prediction data
            preceding_x = np.array([pred[0] for pred in preceding_vehicle_pred])
            preceding_y = np.array([pred[1] for pred in preceding_vehicle_pred])

            # Extract reference path
            ref_x = np.array([point[0] for point in reference_path])
            ref_y = np.array([point[1] for point in reference_path])

            # Ensure arrays are of correct length
            if len(ref_x) < self.horizon:
                # Pad if reference path is too short
                ref_x = np.pad(ref_x, (0, self.horizon - len(ref_x)), 'edge')
                ref_y = np.pad(ref_y, (0, self.horizon - len(ref_y)), 'edge')

            # Initial guess (all zeros)
            x_init = np.zeros(self.opt_vars_size)

            # Pack parameters
            p = np.concatenate([
                x0.flatten(),
                preceding_x.flatten(),
                preceding_y.flatten(),
                ref_x.flatten(),
                ref_y.flatten(),
                [target_speed]
            ])

            # Bounds for variables
            lbx = []
            ubx = []

            for i in range(self.horizon):
                # Control bounds
                lbx.extend([self.min_accel, self.min_steer])
                ubx.extend([self.max_accel, self.max_steer])

                # State bounds (except for last step)
                if i < self.horizon - 1:
                    lbx.extend([-ca.inf, -ca.inf, -ca.inf, 0, self.min_accel])
                    ubx.extend([ca.inf, ca.inf, ca.inf, 40, self.max_accel])

            # Solve optimization
            sol = self.solver(
                x0=x_init,
                lbx=lbx,
                ubx=ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )

            # Check solution status
            stats = self.solver.stats()
            if stats['return_status'] not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
                print(f"Warning: Solver status: {stats['return_status']}")
                # Return zero steering and negative acceleration as fallback
                control = np.zeros((self.horizon, self.num_controls))
                # control[:, 0] = self.min_accel / 100  # Negative acceleration
                return control
            else:
                print(f"Solver succeeded: {stats['return_status']}")

            # Extract solution
            x_opt = sol['x'].full().flatten()

            # Extract control sequence
            u_optimal = []
            for i in range(self.horizon):
                # Index calculation depends on problem structure
                if i < self.horizon - 1:
                    # Each timestep has: u_k (2 controls) + x_k+1 (5 states)
                    idx = i * (self.num_controls + self.num_states)
                else:
                    # Last timestep only has u_k
                    idx = (self.horizon-1) * (self.num_controls + self.num_states) + \
                        (i - (self.horizon-1)) * self.num_controls

                # Extract controls for this timestep
                u_i = x_opt[idx:idx+self.num_controls]
                u_optimal.append(u_i)

            return np.array(u_optimal)

        except Exception as e:
            print(f"CasADi optimization failed: {e}")
            traceback.print_exc()
            # Return zero control as fallback
            return np.zeros((self.horizon, self.num_controls))

    def convert_to_control(self, u_optimal):
        """
        Convert optimal control to CARLA control
        """
        # Extract control values
        accel_cmd = u_optimal[0]  # Acceleration command
        steer = u_optimal[1]      # Steering angle

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

    def spawn_vehicle_sensor(self, blueprint_name, spawn_point, role_name='default', GPS=False, IMU=False):
        '''
        spawn a vehicle at the given spawn_point
        '''
        spawn_transform = self.map.get_waypoint(
            carla.Location(x=spawn_point[0], y=spawn_point[1], z=10),
            project_to_road=True).transform
        spawn_transform.location.z = spawn_transform.location.z + 0.5
        print("Spawn location:", spawn_transform.location)
        print("Spawn rotation:", spawn_transform.rotation)
        blueprint = self.blueprint_library.filter(blueprint_name)[0]
        blueprint.set_attribute("role_name", role_name)
        vehicle = self.world.spawn_actor(blueprint, spawn_transform)

        if IMU or GPS:
            with initialize(version_base='1.1', config_path="configs", job_name="carla-manager"):
                cfg = compose(config_name="config")

        if IMU:
            imu_bp = self.blueprint_library.find('sensor.other.imu')

            # add noise to IMU if specified in the config
            imu_bp.set_attribute('noise_accel_stddev_x', str(cfg.imu.noise_accel_stddev_x))
            imu_bp.set_attribute('noise_gyro_stddev_z', str(cfg.imu.noise_gyro_stddev_z))
            # add noise seed if specified in the config (for reproducibility)
            imu_bp.set_attribute('noise_seed', str(cfg.imu.noise_seed))

            imu_transform = carla.Transform(carla.Location(x=cfg.imu.location.x, y=cfg.imu.location.y, z=cfg.imu.location.z))
            imu = self.world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)

        if GPS:
            gnss_bp = self.blueprint_library.find('sensor.other.gnss')

            # add noise to GNSS if specified in the config
            gnss_bp.set_attribute('noise_lat_stddev', str(cfg.gnss.noise_lat_stddev))
            gnss_bp.set_attribute('noise_lon_stddev', str(cfg.gnss.noise_lon_stddev))
            # add noise seed if specified in the config (for reproducibility)
            gnss_bp.set_attribute('noise_seed', str(cfg.gnss.noise_seed))

            gnss_transform = carla.Transform(carla.Location(x=cfg.gnss.location.x, y=cfg.gnss.location.y, z=cfg.gnss.location.z))
            gnss = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)

        if IMU or GPS:
            return vehicle, imu, gnss
        else:
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
    Run the main simulation with bicycle model MPC controller
    """
    print("Starting simulation with bicycle model MPC controller...")

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

        # Basic agent for preceding vehicle
        preceding_agent = BasicAgent(preceding_vehicle_actor, target_speed=PRECEDING_SPEED)

        # Set destination
        current_location = preceding_vehicle_actor.get_location()
        # Create a sequence of waypoints for the preceding vehicle to follow
        waypoints = []
        current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)

        # Get the next 20 waypoints, 5 meters apart
        next_wp = current_wp
        for _ in range(20):
            next_waypoints = next_wp.next(5.0)
            if next_waypoints:
                next_wp = next_waypoints[0]
                waypoints.append(next_wp.transform.location)
            else:
                break

        # Set the first waypoint as the initial destination
        if waypoints:
            preceding_agent.set_destination(waypoints[0])

        # Spawn ego vehicle behind preceding vehicle
        spawn_loc_copy = SPAWN_LOCATION.copy()
        spawn_loc_copy[0] += 20  # 20 meters behind

        if EKF:
            # Spawn ego vehicle with IMU and GPS sensors
            ego_vehicle_actor, IMU, GPS = carla_manager.spawn_vehicle_sensor(
                "vehicle.tesla.model3", spawn_loc_copy, GPS=True, IMU=True
            )
        else:
            ego_vehicle_actor = carla_manager.spawn_vehicle("vehicle.tesla.model3", spawn_loc_copy)

        if synchronous_mode:
            carla_manager.world.tick()
        else:
            time.sleep(1)

        # Wrap with Vehicle class
        if EKF:
            ego_vehicle = Vehicle(ego_vehicle_actor)
            ego_vehicle.attach_sensors(GPS=GPS, IMU=IMU)
            # to get initial sensor data
            if synchronous_mode:
                carla_manager.world.tick()
            else:
                time.sleep(1)

            ego_vehicle.initialize_ekf(P0, Q, R)
        else:
            ego_vehicle = Vehicle(ego_vehicle_actor)

        # Save spawn transform and initialize transformation matrices
        ego_vehicle.transform_to_spawn = ego_vehicle_actor.get_transform()
        world_to_ego, ego_to_world = ego_vehicle.get_transform_matrices()

        # Create bicycle model MPC controller
        mpc_controller = BicycleMPCController(horizon=50, dt=dt, carla_manager=carla_manager)

        # Target speed (m/s)
        target_speed = 25 / 3.6  # 20 km/h

        # Main control loop
        try:
            print("Starting simulation with bicycle MPC. Press Ctrl+C to exit...")
            preceding_speed_error_prev = 0.0  # Initialize previous speed error
            preceding_steer_error_prev = 0.0  # Initialize previous steering error
            kp_speed = 0.5
            kd_speed = 0.1
            kp_steer = 0.5
            kd_steer = 0.1
            while True:

                # In your main simulation loop
                current_preceding_location = preceding_vehicle_actor.get_location()

                # Update preceding vehicle
                # control_cmd = preceding_agent.run_step()
                control_cmd = carla.VehicleControl()
                # Set a constant speed for the preceding vehicle using a PD controller on the vehicle's speed
                error_speed = PRECEDING_SPEED - preceding_vehicle_actor.get_velocity().length()
                # PD control for speed
                PD_preceding_control = kp_speed * error_speed + kd_speed * (error_speed - preceding_speed_error_prev) / dt
                preceding_speed_error_prev = error_speed
                # Apply control to preceding vehicle
                if PD_preceding_control > 0:
                    control_cmd.throttle = min(PD_preceding_control, 1.0)
                    control_cmd.brake = 0.0
                else:
                    control_cmd.throttle = 0.0
                    control_cmd.brake = min(-PD_preceding_control, 1.0)

                # Apply another PD control for the steering angle to keep the preceding lane following
                # get the error of the vehicle's heading compared to the road direction
                road_wp = carla_manager.map.get_waypoint(current_preceding_location, project_to_road=True)
                road_direction = road_wp.transform.get_forward_vector()
                vehicle_direction = preceding_vehicle_actor.get_transform().get_forward_vector()
                road_heading = np.arctan2(road_direction.y, road_direction.x)
                vehicle_heading = np.arctan2(vehicle_direction.y, vehicle_direction.x)
                heading_error = road_heading - vehicle_heading
                # normalize heading error to [-1, 1]
                heading_error = heading_error / np.pi  # Normalize to [-1, 1]
                control_steer = kp_steer * heading_error + kd_steer * (heading_error - preceding_steer_error_prev) / dt
                # clip control steer to [-1, 1]
                control_steer = np.clip(control_steer, -1.0, 1.0)
                control_cmd.steer = control_steer
                # Apply control to preceding vehicle
                preceding_vehicle_actor.apply_control(control_cmd)

                # Get MPC control for ego vehicle
                ego_control, u_optimal = mpc_controller.run_step(
                    ego_vehicle, preceding_vehicle, target_speed
                )

                # Apply control to ego vehicle
                ego_vehicle_actor.apply_control(ego_control)

                # Save the state and control input as a csv file for plotting later
                # Initialize data collection if first iteration
                if not hasattr(run_simulation_with_casadi, 'simulation_data'):
                    run_simulation_with_casadi.simulation_data = []
                    run_simulation_with_casadi.sim_time = 0.0
                    import os
                    os.makedirs('results', exist_ok=True)
                    print("Data collection initialized in ego coordinate frame")

                # Update simulation time
                run_simulation_with_casadi.sim_time += dt

                # Get ego vehicle state in ego coordinates [x, y, psi, v, a]
                ego_state = ego_vehicle.get_vehicle_state()

                # Get preceding vehicle position in ego coordinates
                preceding_location = preceding_vehicle.actor.get_location()
                preceding_pos_in_ego = ego_vehicle.world_to_ego_coordinates(preceding_location)

                # Extract MPC control commands
                # The raw output from the MPC solver contains acceleration and steering commands
                mpc_accel_cmd = u_optimal[0][0] if len(u_optimal) > 0 else 0
                mpc_steer_cmd = u_optimal[0][1] if len(u_optimal) > 0 else 0

                # Create data record in ego coordinate frame
                data_point = {
                    'time': run_simulation_with_casadi.sim_time,
                    'ego_x': ego_state[0],  # x-position in ego frame
                    'ego_y': ego_state[1],  # y-position in ego frame
                    'ego_heading': ego_state[2],  # heading in ego frame
                    'ego_speed': ego_state[3],  # speed
                    'ego_accel': ego_state[4],  # acceleration
                    'preceding_x': preceding_pos_in_ego[0],  # x-position in ego frame
                    'preceding_y': preceding_pos_in_ego[1],  # y-position in ego frame
                    'control_throttle': ego_control.throttle,
                    'control_brake': ego_control.brake,
                    'control_steer': ego_control.steer,
                    'mpc_accel_cmd': mpc_accel_cmd,
                    'mpc_steer_cmd': mpc_steer_cmd
                }

                # Append to data collection
                run_simulation_with_casadi.simulation_data.append(data_point)

                # Check if target position reached (100m in ego x-direction)
                if ego_state[0] >= 100.0:
                    print(f"Target position reached at {ego_state[0]:.2f}m in ego frame. Ending simulation.")

                    # Save data to CSV
                    import csv
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f'results/overtaking_simulation_{timestamp}.csv'

                    with open(csv_filename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=data_point.keys())
                        writer.writeheader()
                        for data_row in run_simulation_with_casadi.simulation_data:
                            writer.writerow(data_row)

                    print(f"Data saved to {csv_filename}")
                    break

                # Advance simulation
                carla_manager.world.tick()

        except KeyboardInterrupt:
            carla_manager.__del__()
            print("\nSimulation terminated by user")

    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()

    finally:
        # Clean up actors
        if preceding_vehicle_actor and preceding_vehicle_actor.is_alive:
            preceding_vehicle_actor.destroy()

        if ego_vehicle_actor and ego_vehicle_actor.is_alive:
            ego_vehicle_actor.destroy()

        if synchronous_mode:
            settings = carla_manager.world.get_settings()
            settings.synchronous_mode = False
            carla_manager.world.apply_settings(settings)

        # Save collected data to CSV if simulation was interrupted
        if hasattr(run_simulation_with_casadi, 'simulation_data') and run_simulation_with_casadi.simulation_data:
            import csv
            import os
            from datetime import datetime

            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'results/overtaking_simulation_{timestamp}.csv'

            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=run_simulation_with_casadi.simulation_data[0].keys())
                writer.writeheader()
                for data_row in run_simulation_with_casadi.simulation_data:
                    writer.writerow(data_row)

            print(f"Simulation data saved to {csv_filename}")


if __name__ == "__main__":
    run_simulation_with_casadi()
