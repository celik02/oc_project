import time
import numpy as np

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.basic_agent import BasicAgent
from vehicle import Vehicle  # Import the Vehicle class

import casadi as ca

dt = 0.1  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

# synchronous_mode will make the simulation predictable
synchronous_mode = False


def frenet_vehicle_dynamics(x, u, dt):
    """ Vehicle dynamics in Frenet coordinates using numpy instead of CasADi
    Vehicle dynamics in Frenet coordinates
    x: state vector [s, d, mu, v, delta, kappa]
    u: control input [accel, steer]
    dt: time step
    """
    # Vehicle parameters
    L = 3.0  # Wheelbase
    
    # State update using numpy operations
    s_next = x[0] + x[3] * np.cos(x[2]) * dt  # Longitudinal position
    d_next = x[1] + x[3] * np.sin(x[2]) * dt  # Lateral position
    mu_next = x[2] + x[3] * np.tan(x[4]) / L * dt  # Heading angle
    v_next = x[3] + u[0] * dt 
    delta_next = x[4] + u[1] * dt 
    kappa_next = x[5]  # we do not use curvature in our work
    
    # Return next state as a numpy array
    return np.array([s_next, d_next, mu_next, v_next, delta_next, kappa_next])
    

def frenet_vehicle_dynamics_casadi(x, u, dt):
    """
    Adapted CasADi version for the modified state representation
    """
    # Vehicle parameters
    L = 3.0  # Wheelbase
    
    # State update using CasADi expressions
    s_next = x[0] + x[3] * ca.cos(x[2]) * dt # Longitudinal position
    d_next = x[1] + x[3] * ca.sin(x[2]) * dt # Lateral position
    mu_next = x[2] + x[3] * ca.tan(x[4]) / L * dt # Heading angle
    v_next = x[3] + u[0] * dt 
    delta_next = x[4] + u[1] * dt 
    kappa_next = x[5]  # we do not use curvature in our work
    
    # Return next state as a CasADi column vector
    return ca.vertcat(s_next, d_next, mu_next, v_next, delta_next, kappa_next)


class FrenetMPCController:
    def __init__(self, horizon=10, dt=0.1, carla_manager=None):
        self.carla_manager = carla_manager

        self.horizon = horizon
        self.dt = dt
        
        # Control constraints
        self.max_accel = 3.0
        self.min_accel = -5.0
        self.max_steer_rate = 0.1  # Max steering rate (rad/s)
        self.min_steer_rate = -0.1  # Min steering rate (rad/s)
                
        # Cost function weights
        self.w_progress = 5.0    # Forward progress reward
        self.w_lane = 2.0        # Lane centering reward
        self.w_speed = 0.0       # Target speed reward
        self.w_accel = 0.0       # Acceleration minimization
        self.w_steer = 0.0       # Steering minimization
        self.w_jerk = 0.0        # Jerk minimization

        # Lane change parameters
        self.lane_width = 3.5    # Standard lane width in meters
        self.current_lane = 0    # Middle lane (0), left lane (-1), right lane (1)
        self.target_lane = 0     # Initially target the current lane
        self.lane_change_threshold = 2.0  # Threshold to trigger lane change decision
        
        # Initialize CasADi solver (will be created the first time we run)
        self.solver = None
        self.casadi_setup_done = False

        
    def frenet_to_world(self, s, d, reference_waypoint, ego_s, ego_d):
        """
        Convert Frenet coordinates to world coordinates more accurately.
        
        Args:
            s: Longitudinal position in Frenet coordinates
            d: Lateral position in Frenet coordinates
            reference_waypoint: Reference waypoint for the conversion
            ego_s: Current s-coordinate of the ego vehicle
            ego_d: Current d-coordinate of the ego vehicle
            
        Returns:
            carla.Location: World coordinates corresponding to the Frenet coordinates
        """
        # Calculate the relative s displacement from ego vehicle
        ds = s - ego_s
        
        # Find a waypoint at the desired s-position
        if ds > 0:
            # Moving forward along the road
            waypoints = reference_waypoint.next(ds)
            if not waypoints:
                # If we can't find waypoints that far ahead, use the reference waypoint
                target_waypoint = reference_waypoint
            else:
                target_waypoint = waypoints[0]
        elif ds < 0:
            # Moving backward along the road
            waypoints = reference_waypoint.previous(-ds)
            if not waypoints:
                # If we can't find waypoints that far behind, use the reference waypoint
                target_waypoint = reference_waypoint
            else:
                target_waypoint = waypoints[0]
        else:
            # At the same s position
            target_waypoint = reference_waypoint
        
        # Get the base location on the road
        world_loc = target_waypoint.transform.location
        
        # Calculate lateral displacement using the right vector at this specific waypoint
        right_vector = target_waypoint.transform.get_right_vector()
        
        # Calculate absolute lateral position (take into account ego's current d)
        lateral_displacement = d - ego_d
        
        # Apply lateral displacement
        world_loc.x += right_vector.x * lateral_displacement
        world_loc.y += right_vector.y * lateral_displacement
        
        return world_loc

    def draw_predicted_trajectory(self, ego_vehicle, predicted_trajectory, waypoint):
        """
        Visualize the predicted trajectory using improved Frenet-to-world conversion.
        
        Args:
            ego_vehicle: Vehicle class instance for ego vehicle
            predicted_trajectory: Array of predicted states [s, d, mu, v, delta, kappa]
            waypoint: Reference waypoint for Frenet coordinate calculations
        """
        if self.carla_manager is None:
            return
            
        # Get the initial Frenet coordinates
        ego_frenet, _ = ego_vehicle.get_frenet_states(waypoint)
        ego_s, ego_d = ego_frenet[0], ego_frenet[1]
        
        # Small vertical offset for better visibility
        z_offset = 0.5
        
        # Store all trajectory points for drawing lines
        trajectory_points = []
        
        # Draw each point in the predicted trajectory
        for i in range(len(predicted_trajectory)):
            # Get absolute Frenet coordinates
            s = predicted_trajectory[i][0]
            d = predicted_trajectory[i][1]
            
            # Convert to world coordinates using the improved method
            world_pos = self.frenet_to_world(s, d, waypoint, ego_s, ego_d)
            
            # Add z-offset for visibility
            world_pos.z += z_offset
            
            trajectory_points.append(world_pos)
            
            # Color based on position in trajectory (red → green gradient)
            progress = i / (len(predicted_trajectory) - 1) if len(predicted_trajectory) > 1 else 0
            point_color = carla.Color(
                r=int(255 * (1 - progress)),
                g=int(255 * progress),
                b=50  # Add some blue for better visibility
            )
            
            # Draw point
            self.carla_manager.world.debug.draw_point(
                world_pos,
                size=0.05,
                color=point_color,
                life_time=0.3
            )
            
            # Draw connecting lines between points
            if i > 0:
                self.carla_manager.world.debug.draw_line(
                    trajectory_points[i-1],
                    world_pos,
                    thickness=0.05,
                    color=point_color,
                    life_time=0.3
                )
                
        # Draw a special marker at the end of the trajectory
        if trajectory_points:
            end_point = trajectory_points[-1]
            self.carla_manager.world.debug.draw_point(
                end_point,
                size=0.1,
                color=carla.Color(0, 255, 0),  # Green
                life_time=0.3
            )

            # Display predicted speed at the end point
            final_speed = predicted_trajectory[-1][3]  # v component
            speed_text = f"{final_speed * 3.6:.1f} km/h"  # Convert m/s to km/h
            
            self.carla_manager.world.debug.draw_string(
                end_point + carla.Location(z=0.5),  # Position text above the point
                speed_text,
                draw_shadow=True,
                color=carla.Color(255, 255, 255),  # White text
                life_time=0.3
            )

    def visualize_elliptical_constraint(self, ego_vehicle, preceding_vehicle, waypoint, lead_frenet):
        """
        Visualize the elliptical constraint using improved Frenet-to-world conversion.
        
        Args:
            ego_vehicle: Vehicle class instance for ego vehicle
            preceding_vehicle: Vehicle class instance for preceding vehicle
            waypoint: Reference waypoint for Frenet coordinate calculations
            lead_frenet: Frenet coordinates of the lead vehicle
        """
        if self.carla_manager is None:
            return
            
        # Get ego Frenet coordinates
        ego_frenet, _ = ego_vehicle.get_frenet_states(waypoint)
        ego_s, ego_d = ego_frenet[0], ego_frenet[1]
        
        # Get parameters used in constraint
        a = 4.0  # Longitudinal semi-axis of ellipsoid
        b = 2.0  # Lateral semi-axis of ellipsoid
        ellipsoid_offset = 20.0  # The offset used in the MPC constraint
        
        # Calculate the center of the ellipsoid in Frenet coordinates
        lead_s, lead_d = lead_frenet[0], lead_frenet[1]
        
        # Since lead_s is in lead vehicle's frame, convert to ego vehicle's frame
        lead_s_ego_frame = lead_s + ellipsoid_offset
        
        # Calculate ellipse center in Frenet coordinates
        ellipsoid_center_s = lead_s_ego_frame
        ellipsoid_center_d = lead_d
        
        # Draw the ellipsoid in the Frenet frame, transformed to world coordinates
        num_points = 36  # Number of points to draw around the ellipse
        theta = np.linspace(0, 2*np.pi, num_points)
        
        # Calculate ellipse boundary points in Frenet coordinates
        boundary_s = ellipsoid_center_s + a * np.cos(theta)
        boundary_d = ellipsoid_center_d + b * np.sin(theta)
        
        # Transform to world coordinates using improved method
        ellipse_points = []
        for i in range(num_points):
            # Convert to world coordinates
            ellipse_point = self.frenet_to_world(
                boundary_s[i], boundary_d[i], waypoint, ego_s, ego_d
            )
            
            # Add small z-offset for visibility
            ellipse_point.z += 0.3
            
            ellipse_points.append(ellipse_point)
            
        # Draw ellipse boundary
        for i in range(num_points):
            # Draw point
            self.carla_manager.world.debug.draw_point(
                ellipse_points[i],
                size=0.05,
                color=carla.Color(255, 0, 255),  # Purple for ellipse
                life_time=0.3
            )
            
            # Draw connecting line to next point
            next_i = (i + 1) % num_points
            self.carla_manager.world.debug.draw_line(
                ellipse_points[i],
                ellipse_points[next_i],
                thickness=0.05,
                color=carla.Color(255, 0, 255),  # Purple for ellipse
                life_time=0.3
            )
        
        # Draw ellipsoid center
        ellipse_center = self.frenet_to_world(
            ellipsoid_center_s, ellipsoid_center_d, waypoint, ego_s, ego_d
        )
        ellipse_center.z += 0.5  # Slightly above for visibility
        
        self.carla_manager.world.debug.draw_point(
            ellipse_center,
            size=0.2,
            color=carla.Color(255, 0, 0),  # Red for center
            life_time=0.3
        )
        
        # Add diagnostic text at ellipse center
        diagnostic_text = f"Ellipse center: s={ellipsoid_center_s-ego_s:.1f}m, d={ellipsoid_center_d-ego_d:.1f}m"
        self.carla_manager.world.debug.draw_string(
            ellipse_center + carla.Location(z=0.5),
            diagnostic_text,
            draw_shadow=True,
            color=carla.Color(255, 255, 255),  # White
            life_time=0.3
        )
        
        # Compare with actual lead vehicle position for verification
        lead_actual_pos = preceding_vehicle.actor.get_location()
        lead_actual_pos.z += 0.3  # Offset for visibility
        
        self.carla_manager.world.debug.draw_point(
            lead_actual_pos,
            size=0.2,
            color=carla.Color(0, 0, 255),  # Blue for actual lead position
            life_time=0.3
        )
        
        # Draw a line connecting actual lead position to computed ellipse center
        self.carla_manager.world.debug.draw_line(
            lead_actual_pos,
            ellipse_center,
            thickness=0.05,
            color=carla.Color(255, 255, 0),  # Yellow
            life_time=0.3
        )
        
        # Calculate and display distance between actual and computed positions
        distance = lead_actual_pos.distance(ellipse_center)
        distance_text = f"Offset: {distance:.1f}m"
        mid_point = carla.Location(
            (lead_actual_pos.x + ellipse_center.x) / 2,
            (lead_actual_pos.y + ellipse_center.y) / 2,
            (lead_actual_pos.z + ellipse_center.z) / 2 + 0.5
        )
        
        self.carla_manager.world.debug.draw_string(
            mid_point,
            distance_text,
            draw_shadow=True,
            color=carla.Color(255, 255, 0),  # Yellow
            life_time=0.3
        )

    def visualize_lanes_and_states(self, ego_vehicle, preceding_vehicle, carla_manager, d0, lead_d):
        """
        Draw visual indicators for lane centers and vehicle states in CARLA.
        
        Args:
            ego_vehicle: The ego vehicle instance
            preceding_vehicle: The lead vehicle instance
            carla_manager: The CarlaManager instance for drawing
            d0: Current lateral position of ego vehicle
            lead_d: Current lateral position of lead vehicle
        """
        # Get current waypoint and road features
        ego_location = ego_vehicle.actor.get_location()
        waypoint = carla_manager.map.get_waypoint(ego_location, project_to_road=True)
        
        # Draw lane centers (extending 20m forward and backward)
        forward_vec = waypoint.transform.get_forward_vector()
        right_vec = waypoint.transform.get_right_vector()
        
        # Center lane (d=0)
        center_lane_start = ego_location + forward_vec * (-20)  # 20m behind
        center_lane_end = ego_location + forward_vec * 40  # 40m ahead
        carla_manager.world.debug.draw_line(
            center_lane_start, center_lane_end,
            thickness=0.1, color=carla.Color(0, 255, 0),  # Green
            life_time=0.3  # Short-lived visualization updated frequently
        )
        
        # Left lane (d=-3.5)
        left_lane_start = center_lane_start - right_vec * self.lane_width
        left_lane_end = center_lane_end - right_vec * self.lane_width
        carla_manager.world.debug.draw_line(
            left_lane_start, left_lane_end,
            thickness=0.1, color=carla.Color(0, 0, 255),  # Blue
            life_time=0.3
        )
        
        # Right lane (d=3.5)
        right_lane_start = center_lane_start + right_vec * self.lane_width
        right_lane_end = center_lane_end + right_vec * self.lane_width
        carla_manager.world.debug.draw_line(
            right_lane_start, right_lane_end,
            thickness=0.1, color=carla.Color(255, 0, 0),  # Red
            life_time=0.3
        )
        
        # Draw target lane indicator
        target_d = self.target_lane * self.lane_width
        target_point = ego_location + forward_vec * 20  # 20m ahead
        target_point = target_point - right_vec * target_d  # Adjust for desired lateral position
        
        carla_manager.world.debug.draw_point(
            target_point, size=0.2,
            color=carla.Color(255, 255, 0),  # Yellow
            life_time=0.3
        )

        text_location = ego_location + carla.Location(z=3.0)  # Above the vehicle
        
        # Draw lateral position indicators
        d_text = f"Ego d: {d0:.2f}, Target d: {target_d:.2f}"
        d_location = text_location + carla.Location(z=0.5)
        carla_manager.world.debug.draw_string(
            d_location, d_text,
            draw_shadow=True, color=carla.Color(200, 200, 200),
            life_time=0.3
        )

    def display_state_in_corner(self, carla_manager):
        """
        Displays the current state of the controller in the top-right corner of the screen.
        
        Args:
            carla_manager: The CarlaManager instance with access to debug drawing tools
        """        
        # Get the current camera view transform from the spectator
        spectator_transform = carla_manager.spectator.get_transform()
        
        # Calculate a position in 3D space that will appear in the top-right corner of the screen
        # This requires understanding the camera view frustum geometry
        forward_vector = spectator_transform.get_forward_vector()
        right_vector = spectator_transform.get_right_vector()
        up_vector = spectator_transform.get_up_vector()
        
        # Position the text relative to the camera
        # The specific values (15, 8, 5) represent offsets that place the text in the corner
        # These values may need adjustment based on your specific camera setup
        position = (spectator_transform.location + 
                    forward_vector * 15 +  # Project forward into visible space
                    right_vector * 8 +     # Move to the right side of the screen
                    up_vector * 5)         # Move to the top portion of the screen
        
        
        
        # Add additional diagnostic information below the state
        info_position = carla.Location(position.x, position.y, position.z - 0.5)  # Position slightly below
        
        # Add target lane information
        lane_text = f"Target Lane: {self.target_lane} (d={self.target_lane * self.lane_width:.1f}m)"
        carla_manager.world.debug.draw_string(
            info_position,
            lane_text,
            draw_shadow=True,
            color=carla.Color(200, 200, 200),  # Light gray for supplementary info
            life_time=0.5
        )
        # Display ellipsoid constraint information in the corner
        ellipsoid_info_position = info_position + carla.Location(z=-1.0)  # Position below lane info
        ellipsoid_text = f"Ellipsoid: a={4.0}m, b={2.0}m, offset={20.0}m"
        carla_manager.world.debug.draw_string(
            ellipsoid_info_position,
            ellipsoid_text,
            draw_shadow=True,
            color=carla.Color(255, 200, 255),  # Light purple for ellipsoid info
            life_time=0.5
        )
        
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
        s0, d0, mu0, v0, steering0, curvature0 = ego_frenet
        
        # Stabilize heading to prevent runaway rotation
        if abs(mu0) > np.pi/2:  # If heading error exceeds 90 degrees
            print(f"WARNING: Extreme heading error: {np.degrees(mu0):.2f}° - stabilizing")
            # Gradually limit extreme heading errors to prevent instability
            mu0 = np.sign(mu0) * np.pi/4  # Cap at 45 degrees while preserving direction
            
        # Then update x0 with the stabilized heading
        x0 = np.array([s0, d0, mu0, v0, steering0, curvature0])
        
        # Get Frenet states for preceding vehicle using the same waypoint reference
        lead_frenet, _ = preceding_vehicle.get_frenet_states(next_waypoint)
        lead_s, lead_d, _, lead_speed, _, _ = lead_frenet

        # Print diagnostic information about the Frenet coordinates
        print(f"Ego Frenet: s={s0:.2f}, d={d0:.2f}, v={v0:.2f}")
        print(f"Lead Frenet: s={lead_s:.2f}, d={lead_d:.2f}, v={lead_speed:.2f}")
        print(f"Lead s in ego frame: s={lead_s + 20:.2f}")
        
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
        frenet_relative_s = (lead_s + 20) - s0
        print(f"World-space relative s: {relative_s:.2f}")
        print(f"Frenet-space relative s: {frenet_relative_s:.2f}")
        print(f"Difference: {abs(relative_s - frenet_relative_s):.2f}")
        
        # Display state machine status in corner
        if self.carla_manager is not None:
            self.display_state_in_corner(self.carla_manager)
                
        # Calculate desired lateral position based on target lane
        d_desired = self.target_lane * self.lane_width
        print(f"Target lane: {self.target_lane}, d_desired: {d_desired:.2f}")
        
        # Generate lead vehicle prediction in Frenet coordinates
        lead_vehicle_pred = self.get_lead_vehicle_prediction(relative_s, lead_d, lead_speed)
        
        # Solve optimization problem with CasADi
        if not self.casadi_setup_done:
            self.setup_casadi_solver()
            self.casadi_setup_done = True
            
        # When calling solve_with_casadi, pass the updated state vector
        u_optimal = self.solve_with_casadi(x0, lead_vehicle_pred, target_speed, d_desired)

        # Generate predicted trajectory
        predicted_trajectory = [x0]
        for u in u_optimal:
            next_state = frenet_vehicle_dynamics(predicted_trajectory[-1], u, self.dt)
            predicted_trajectory.append(next_state)
        predicted_trajectory = np.array(predicted_trajectory)
        
        # Use the improved visualization methods
        if self.carla_manager is not None:
            # Draw predicted trajectory with improved Frenet-to-world conversion
            self.draw_predicted_trajectory(ego_vehicle, predicted_trajectory, waypoint)
            
            # Visualize elliptical constraint with improved conversion
            self.visualize_elliptical_constraint(ego_vehicle, preceding_vehicle, waypoint, lead_frenet)
            
            # Visualize lane boundaries
            self.visualize_lanes_and_states(ego_vehicle, preceding_vehicle, self.carla_manager, d0, lead_d)

        print("Control action", u_optimal[0])
        # Convert to CARLA control, adapting to use acceleration and steering rate
        control = self.convert_to_control(u_optimal[0], x0[4])  # Pass current steering angle
        
        return control
    
    def setup_casadi_solver(self):
        """
        Setup the CasADi solver for MPC optimization with adapted state representation.
        """
        print("Setting up CasADi solver...")
        
        # State variables (symbolic) - now 6 dimensions
        x = ca.SX.sym('x', 6)  # [s, d, mu, v, steering, curvature]
        
        # Control variables (symbolic) - now [accel, steering_rate]
        u = ca.SX.sym('u', 2)  # [accel, steering_rate]
        
        # Define parameters that will change each time step
        # Initial state
        x0 = ca.SX.sym('x0', 6)
        
        # Lead vehicle predicted positions (for all steps in horizon)
        lead_s_pred = ca.SX.sym('lead_s_pred', self.horizon)
        lead_d_pred = ca.SX.sym('lead_d_pred', self.horizon)
        
        # Target speed and desired lateral position
        target_speed = ca.SX.sym('target_speed', 1)
        d_desired = ca.SX.sym('d_desired', 1)
        
        # Define the dynamics function with adapted model
        dynamics_func = ca.Function('dynamics', [x, u], 
                                [frenet_vehicle_dynamics_casadi(x, u, self.dt)])
        
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
            opt_vars_lb.extend([self.min_accel, self.min_steer_rate])
            opt_vars_ub.extend([self.max_accel, self.max_steer_rate])
            
            # In the for k in range(self.horizon) loop:

            # 1. Progress reward (using distance along the lane)
            obj -= self.w_progress * xk[0]  # Forward progress reward

            # 2. Lane centering
            obj += self.w_lane * (xk[1] - d_desired)**2

            # 3. Target speed
            obj += self.w_speed * (target_speed - xk[3])**2

            # 4. Control effort - now penalizing acceleration and steering rate
            obj += self.w_accel * uk[0]**2  # Acceleration
            obj += self.w_steer * uk[1]**2  # Steering rate (not steering angle)

            # 5. Jerk minimization - adapt if needed based on new control definition
            if k > 0:
                prev_uk = opt_vars[-3]  # Two items back in the list
                obj += self.w_jerk * ((uk[0] - prev_uk[0]) / self.dt)**2  # Jerk penalty (accel rate)     

            # Lead vehicle position at this step
            lead_s_k = lead_s_pred[k]
            lead_d_k = lead_d_pred[k]
            
            # Relative longitudinal distance
            rel_s = lead_s_k - xk[0]
                        
            # Collision avoidance constraint (ellipsoid)
            # Define ellipsoid parameters
            a = 4.0  # Longitudinal semi-axis (front-back)
            b = 2.0  # Lateral semi-axis (side-to-side)
            
            # Calculate distances in Frenet coordinates
            ds = (lead_s_k + 20) - xk[0]  # Longitudinal distance
            dd = lead_d_k - xk[1]  # Lateral distance
            
            # Add collision avoidance constraint
            # We want (ds/a)² + (dd/b)² >= 1 (outside the ellipsoid)
            g.append((ds/a)**2 + (dd/b)**2)
            lbg.append(1.0)  # Lower bound: must be greater than 1 to stay outside
            ubg.append(ca.inf)  # Upper bound: infinity
            
            # Lane boundary constraints
            # Calculate distance to each lane boundary
            for boundary in lane_boundaries:
                margin = ca.fabs(xk[1] - boundary)
                # Add a constraint that margin should be greater than safe_margin
                g.append(margin)
                lbg.append(safe_margin)
                ubg.append(ca.inf)
            
            # State propagation - get the next state
            xk_next = dynamics_func(xk, uk)
            
            # Add dynamics constraints (next state must follow the dynamics model)
            if k < self.horizon - 1:
                # When creating opt_vars:
                # For each state variable, we need to define a new symbolic variable for the next state
                xk_next_sym = ca.SX.sym(f'x_{k+1}', 6)  # Now 6-dimensional
                opt_vars.append(xk_next_sym)
                
                # No bounds on states except for practical limits, adjusted for 6D state
                opt_vars_lb.extend([-ca.inf, -ca.inf, -ca.inf, 0, -ca.inf, -ca.inf])  # v >= 0
                opt_vars_ub.extend([ca.inf, ca.inf, ca.inf, 40, ca.inf, ca.inf])
                
                # Add the dynamics constraint: next state must equal dynamics model
                g.append(xk_next_sym - xk_next)
                lbg.extend([0, 0, 0, 0, 0, 0])  # Equality constraints for 6D state
                ubg.extend([0, 0, 0, 0, 0, 0])
                
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
                'warm_start_init_point': 'yes'  # Use warm starting
            },
            'print_time': False
        }

        # Store constraint bounds when creating the problem
        self.lbg = lbg  # Store lower bounds for constraints
        self.ubg = ubg  # Store upper bounds for constraints
        
        # Create the solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Store information about the problem dimensions
        self.num_states = 6
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
                lbx.extend([self.min_accel, self.min_steer_rate])
                ubx.extend([self.max_accel, self.max_steer_rate])
                
                # If not the last step, add state bounds
                if _ < self.horizon - 1:
                    # Bounds for 6D state
                    lbx.extend([-np.inf, -np.inf, -np.inf, 0, -np.inf, -np.inf])
                    ubx.extend([np.inf, np.inf, np.inf, 40, np.inf, np.inf])
            
            # Solve the optimization problem with stored constraint bounds
            sol = self.solver(
                x0=x_init,
                lbx=lbx,
                ubx=ubx,
                lbg=self.lbg,  # Use stored lower bounds
                ubg=self.ubg,  # Use stored upper bounds
                p=p
            )
            
            # Extract the optimal control sequence
            x_opt = sol['x'].full().flatten()
            
            # Reshape to get control sequence - adjusted for the 6+2 dimensionality
            u_optimal = []
            for i in range(self.horizon):
                idx = i * (self.num_controls + self.num_states) if i < self.horizon - 1 else i * 2
                u_i = x_opt[idx:idx+2]
                u_optimal.append(u_i)
                
            return np.array(u_optimal)
                        
        except Exception as e:
            print(f"CasADi optimization failed: {e}")
            
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

        return predictions
    
    def convert_to_control(self, u_optimal, current_steering):
        """
        Convert MPC control output to CARLA control
        
        Args:
            u_optimal: Optimal control input [acceleration, steering_rate]
            current_steering: Current steering angle
        
        Returns:
            carla.VehicleControl object
        """
        # Extract control inputs
        acceleration = u_optimal[0]
        steering_rate = u_optimal[1]
        
        # Compute target steering angle (not used directly, just for debugging)
        # In practice, the steering rate will be integrated by the vehicle dynamics
        target_steering = current_steering + steering_rate * self.dt
        
        # Convert to CARLA control
        control = carla.VehicleControl()
        
        # Convert acceleration to throttle/brake
        if acceleration >= 0:
            control.throttle = min(acceleration / 3.0, 1.0)  # Assuming max accel of 3 m/s^2
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-acceleration / 5.0, 1.0)  # Assuming max decel of 5 m/s^2
        
        # Apply steering rate by converting to steering command
        # Note: CARLA expects steering in [-1, 1] range
        max_steering_angle = 0.5  # Max physical steering angle in radians
        
        # We're going to apply the steering rate directly as a change to the steering value
        # This simplification works if the simulation step is close to the controller dt
        control.steer = (current_steering + steering_rate * self.dt) / max_steering_angle
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
        next_wps = current_wp.next(100.0)  # 100 meters ahead
    
        if next_wps:  # if there is a waypoint ahead
            destination = next_wps[0].transform.location
        else:
            destination = current_location
        agent.set_destination(destination)
    
        # Spawn ego vehicle
        spawn_loc_copy = SPAWN_LOCATION.copy()  # Make a copy to avoid modifying the original
        spawn_loc_copy[0] += 20  # 20 meters behind preceding vehicle
        ego_vehicle_actor = carla_manager.spawn_vehicle("vehicle.tesla.model3", spawn_loc_copy)
        time.sleep(1)  # allow the vehicle to spawn
        
        # Wrap with Vehicle class
        ego_vehicle = Vehicle(ego_vehicle_actor)
    
        # Create Frenet MPC controller with CasADi for ego vehicle
        mpc_controller = FrenetMPCController(horizon=30, dt=0.1, carla_manager=carla_manager)
    
        # Target speed for ego vehicle (m/s)
        target_speed = 20 / 3.6  # Convert from km/h to m/s
    
        # Main control loop
        try:
            print("Starting simulation with CasADi MPC. Press Ctrl+C to exit...")
            while True:
                # Get next waypoint for Frenet coordinates
                current_location = ego_vehicle_actor.get_location()
                next_waypoint = carla_manager.map.get_waypoint(current_location, project_to_road=True)
                
                # Control preceding vehicle (keep stationary)
                control_cmd = agent.run_step()
                preceding_vehicle_actor.apply_control(control_cmd)
    
                # Control ego vehicle with CasADi Frenet MPC
                ego_control = mpc_controller.run_step(
                    ego_vehicle, preceding_vehicle, target_speed, next_waypoint
                )
                ego_vehicle_actor.apply_control(ego_control)
    
                # Print current status
                ego_frenet, _ = ego_vehicle.get_frenet_states(next_waypoint)
                lead_frenet, _ = preceding_vehicle.get_frenet_states(next_waypoint)
                
                # Calculate real-world distance between vehicles
                ego_loc = ego_vehicle_actor.get_location()
                lead_loc = preceding_vehicle_actor.get_location()
                distance = np.sqrt(
                    (ego_loc.x - lead_loc.x) ** 2 + (ego_loc.y - lead_loc.y) ** 2
                )
                
                # print(f"Distance to lead vehicle: {distance:.2f} m")
                # print(f"Ego lateral deviation: {ego_frenet[1]:.2f} m")
                # print(f"Ego heading error: {np.degrees(ego_frenet[2]):.2f} degrees")
                # print(f"Optimization using CasADi solver")
    
                time.sleep(0.05)
    
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
        carla_manager.restart_world()
        print("Reset complete. You can run the program again without restarting Carla.")

if __name__ == "__main__":
    run_simulation_with_casadi()