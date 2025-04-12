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


class MPCController:
    def __init__(self, horizon=10, dt=0.1):
        self.horizon = horizon
        self.dt = dt
        
        # Control constraints
        self.max_accel = 3.0
        self.min_accel = -5.0
        self.max_steer = 0.5 / 2
        self.min_steer = -0.5 / 2
                
        # Cost function weights
        self.w_progress = 2.0    # Forward progress reward
        self.w_lane = 2.0        # Lane centering reward
        self.w_speed = 0.0       # Target speed reward
        self.w_accel = 0.0       # Acceleration minimization
        self.w_steer = 0.0       # Steering minimization
        self.w_jerk = 0.0       # Jerk minimization

    def run_step(self, ego_vehicle, preceding_vehicle, target_speed):
        # Extract current state
        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        lead_transform = preceding_vehicle.get_transform()
        lead_velocity = preceding_vehicle.get_velocity()
        
        # Construct current state vector
        x0 = np.array([
            ego_transform.location.x,
            ego_transform.location.y,
            ego_transform.rotation.yaw * np.pi / 180.0,
            np.sqrt(ego_velocity.x**2 + ego_velocity.y**2),
            0.0  # Estimated acceleration
        ])
        
        # Get lead vehicle state and speed
        lead_state = (lead_transform.location.x, lead_transform.location.y)
        lead_speed = np.sqrt(lead_velocity.x**2 + lead_velocity.y**2)
        
        # Generate lead vehicle prediction
        lead_vehicle_pred = self.get_lead_vehicle_prediction(lead_state, lead_speed)
        
        # Solve optimization problem
        u_optimal = self.control_optimization(x0, lead_vehicle_pred, target_speed)
        
        # Convert to CARLA control
        control = self.convert_to_control(u_optimal[0])
        
        return control
    
    def control_optimization(self, x0, lead_vehicle_pred, target_speed):
        # Initial guess - zeros for all controls
        u0 = np.zeros(self.horizon * 2)
        
        # Define bounds for controls
        bounds = []
        for i in range(self.horizon):
            bounds.append((self.min_accel, self.max_accel))
            bounds.append((self.min_steer, self.max_steer))
        
        # Define ellipsoid constraint for all prediction steps
        constraints = []
        for i in range(self.horizon):
            constraints.append({
                'type': 'ineq',
                'fun': lambda u, i=i, x0=x0, lead_vehicle_pred=lead_vehicle_pred: 
                    self.ellipsoid_constraint(u, x0, lead_vehicle_pred, i)
            })
        
        # Solve optimization problem with constraints
        result = minimize(
            self.objective_function,
            u0,
            args=(x0, lead_vehicle_pred),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100}
        )
        
        # Extract optimal control sequence
        u_optimal = result.x.reshape(self.horizon, 2)
        
        return u_optimal

    def ellipsoid_constraint(self, u, x0, lead_vehicle_pred, step_index):
        """
        Ellipsoid constraint function for collision avoidance
        
        Returns positive value when constraint is satisfied (no collision)
        Returns negative value when constraint is violated (collision)
        
        Args:
            u: Flattened control sequence
            x0: Initial state
            lead_vehicle_pred: Predicted lead vehicle positions
            step_index: Index in the prediction horizon to evaluate
            
        Returns:
            Constraint value (positive when satisfied)
        """
        # Reshape control sequence
        u_reshaped = u.reshape(-1, 2)
        
        # Simulate trajectory up to the specified step
        x = x0.copy()
        for i in range(step_index + 1):
            if i < len(u_reshaped):
                x = vehicle_dynamics(x, u_reshaped[i], self.dt)
        
        # Get lead vehicle position at this step
        lead_x, lead_y = lead_vehicle_pred[step_index]
        
        # Define ellipsoid parameters
        a = 4  # Longitudinal semi-axis (front-back)
        b = 2  # Lateral semi-axis (side-to-side)
        
        # Apply vehicle orientation to ellipsoid
        dx = x[0] - lead_x
        dy = x[1] - lead_y
        
        # Transform to lead vehicle's coordinate frame
        # Assuming lead vehicle is aligned with the road
        # (In a more complete implementation, you would use the lead vehicle's heading)
        theta = 0  # Simplified for this example
        
        # Rotated coordinates
        dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
        dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)
        
        # Ellipsoid constraint: (x/a)² + (y/b)² >= 1 means OUTSIDE the ellipsoid
        # We use >= 1 because we want the ego vehicle to stay outside the ellipsoid
        constraint_value = (dx_rot/a)**2 + (dy_rot/b)**2 - 1
        
        return constraint_value
    
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
            cost += self.w_progress * x[0]  
        
            # 2. Lane centering - encourage staying in the lane
            lane_y = 13.13945484161377  # Current lane center y-coordinate

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
    # import subprocess
    # # ./CarlaUE4.sh -quality-level=Low

    # subprocess.Popen(["/home/abdelrahman/CARLA_0.9.15/CarlaUE4.sh", "-quality-level=Low"])
    # time.sleep(5)  # wait for Carla to start

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
    SPAWN_LOCATION[0] += 20  # 15 meters behind preceding vehicle
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
        carla_manager.restart_world()
    finally:
        # Clean up
        if preceding_vehicle and ego_vehicle:
            preceding_vehicle.destroy()
            ego_vehicle.destroy()
        print("Vehicles destroyed")