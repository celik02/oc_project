import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # just constant velocity no lateral control
from agents.navigation.basic_agent import BasicAgent
import time
from hydra import initialize, compose
import numpy as np
from scipy.linalg import solve_continuous_are
import math
import atexit
import traceback
from vehicle import Vehicle  # custom vehicle class to handle transformations
import csv
import os
from datetime import datetime



dt = 0.1  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

# make the lqr has a longer horizon 
lookahead_offset = 2 
# safety distance:
safe_distance = 8.0

# synchronous_mode will make the simulation predictable
synchronous_mode = True

def log_both(msg, log_file):
    """
    log for debug
    """
    print(msg)
    print(msg, file=log_file)

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
                x=SPAWN_LOCATION[0]+15, y=SPAWN_LOCATION[1], z=SPAWN_LOCATION[2]
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
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors()])

    def spawn_vehicle(self, blueprint_name, spawn_point, role_name='default', GPS=False, IMU=False):
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

    def debug_np_locations(self, np_locations, z_height=0.2):
        # draw all the locations
        for pt in np_locations:
            self.world.debug.draw_string(
                carla.Location(x=pt[0], y=pt[1], z=pt[2] + z_height),
                "0",
                draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0),
                life_time=360.0,
                persistent_lines=True,
            )

def wrap_to_pi(angle):
    """
    Convert any radius to (-pi, pi)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def vehicle_linear_dynamics(x0, u0, L=3.0, Tau=1.0):
    """
    Linearize bicycle model around x0 and u0.
    
    Parameters:
    - x0: np.array([x, y, psi, v, a]) - current state
    - u0: np.array([delta, u_a])     - current control input
    - L: float - wheelbase length of the vehicle
    - Tau: float - acceleration time constant.

    Returns:
    - A: Jacobian w.r.t. state x
    - B: Jacobian w.r.t. input u
    """
    x, y, psi, v, a = x0
    delta, u_a = u0

    A = np.zeros((5, 5))
    B = np.zeros((5, 2))

    # ∂f/∂x
    A[0, 2] = -v * np.sin(psi)          # ∂(xdot)/∂psi
    A[0, 3] = np.cos(psi)              # ∂(xdot)/∂v
    A[1, 2] = v * np.cos(psi)          # ∂(ydot)/∂psi
    A[1, 3] = np.sin(psi)              # ∂(ydot)/∂v
    A[2, 3] = np.tan(delta) / L        # ∂(psidot)/∂v
    A[3, 4] = 1                        # ∂(vdot)/∂a
    A[4, 4] = -1 / Tau                 # ∂(adot)/∂a

    # ∂f/∂u
    B[2, 0] = v / (L * np.cos(delta)**2)  # ∂(psidot)/∂delta
    B[4, 1] = 1 / Tau                     # ∂(adot)/∂u_a

    return A, B

def nonlinear_dynamics(x, u, L=3.0, Tau=1.0):
    """
    Nonlinear bicycle model.
    x = [x, y, psi, v, a]
    u = [delta, u_a]
    """
    x_, y_, psi, v, a = x
    delta, u_a = u

    dxdt = np.zeros(5)
    dxdt[0] = v * np.cos(psi)
    dxdt[1] = v * np.sin(psi)
    dxdt[2] = v * np.tan(delta) / L
    dxdt[3] = a
    dxdt[4] = (u_a - a) / Tau

    return dxdt

def generate_lane_change_segment(last_wp, carla_map, direction="left", forward_length = 20.0, steps = 10):
    """
    Generate a smooth lane change segment from last_wp into adjacent lane

    Parameters:
        last_wp: the waypoint in the current lane
        carla_map: carla.Map object
        direction: "left" or "right"
        forward_length: (float) forward length to finish transition
        lateral_length: (float) lateral length to finsih transition
        steps: #wp generated for thr transition
        

    Returns:
        segment: List of interpolated carla.Location points
        seg_loc: np.array list of interpolated points
    """

    # 1. Current point
    segment = []
    seg_loc = []
    p_cur = last_wp.transform.location   # current position
    lateral_length = last_wp.lane_width
    dForward = forward_length/steps     # compare to last wp, each new wp go forward w/ this value
    dLateral = lateral_length/steps     # compare to last wp, each new wp go lateral w/ this value

   # 2. Forward direction vector
    next_wps = last_wp.next(0.5)
    if not next_wps:
        print("No forward waypoint found.")
        return segment, seg_loc
    
    p_forward = next_wps[0].transform.location
    forward_vec = np.array([p_forward.x - p_cur.x, p_forward.y - p_cur.y])
    forward_vec /= np.linalg.norm(forward_vec)

    dForwardVec = dForward*forward_vec   # the forward vector compared to last wp

    # 3. Lateral direction vector (left or right)
    if direction == "left":
        side_wp = last_wp.get_left_lane()
    else:
        side_wp = last_wp.get_right_lane()
    
    if side_wp is None:
        print(f"No {direction} lane available.")
        return segment, seg_loc
    
    p_side = side_wp.transform.location
    lateral_vec = np.array([p_side.x - p_cur.x, p_side.y - p_cur.y])
    lateral_vec -= np.dot(lateral_vec, forward_vec) * forward_vec  # remove forward component

    if np.linalg.norm(lateral_vec) < 1e-3:
        print("Lateral vector is degenerate.")
        return segment, seg_loc
    
    lateral_vec /= np.linalg.norm(lateral_vec)
    dLateralVec = dLateral * lateral_vec    # the lateral vector compared to last wp
    
    # 4. Generate waypoints
    for i in range(1, steps + 1):
        offset = i * dForwardVec + i * dLateralVec
        new_loc = carla.Location(
            x=p_cur.x + offset[0],
            y=p_cur.y + offset[1],
            z=p_cur.z
        )
        curr_location = np.array([new_loc.x, new_loc.y, new_loc.z])
        new_wp = carla_map.get_waypoint(new_loc, project_to_road=False)
        if new_wp is None:
            # if the new_wp is not at lane, use projection to put it back to the center
            new_wp = carla_map.get_waypoint(new_loc, project_to_road=True)
            print(f"Point {i} is off-road, projected back to center.")
        segment.append(new_wp)
        seg_loc.append(curr_location)

    return segment, seg_loc
               
def generate_overtake_waypoints(carla_manager, vehicle, direction="left",
                                distance_current_lane=10.0,
                                lane_change_step=2.0,
                                overtake_distance=50.0,
                                merge_distance=10.0):
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
        waypoints - List of carla.Waypoint objects forming the overtaking trajectory.
        locationPos - List only recording location points for LQR Controller
        transitionIdx - List of tranistion points start and end. For gain schedule.
    """
    _map = carla_manager.map
    waypoints = []
    locationPos = []    
    transitionIdx = []

    # 1. Start in current lane.
    current_wp = _map.get_waypoint(vehicle.get_location(), project_to_road=True)
    loc = current_wp.transform.location
    curr_location = np.array([loc.x, loc.y, loc.z])

    waypoints.append(current_wp)
    locationPos.append(curr_location)

    # Follow current lane for distance_current_lane
    traveled = 0.0
    last_wp = current_wp
    while traveled < distance_current_lane:
        next_wps = last_wp.next(lane_change_step)
        if not next_wps:
            break
        next_wp = next_wps[0]
        loc = next_wp.transform.location
        curr_location = np.array([loc.x, loc.y, loc.z])
        traveled += next_wp.transform.location.distance(last_wp.transform.location)
        
        waypoints.append(next_wp)
        locationPos.append(curr_location)

        last_wp = next_wp
    
    transitionIdx.append(len(waypoints)-1)

    # 2. Change lane: get the adjacent lane from the last waypoint
    #if direction == "left":
    #    adjacent_wp = last_wp.get_left_lane()
    #    # next_adjacent_wp = adjacent_wp.next(lane_change_step)
    #    # next_adjacent_wp =next_adjacent_wp[0]
    #else:
    #    adjacent_wp = last_wp.get_right_lane()
    #    # next_adjacent_wp = adjacent_wp.next(lane_change_step)
    #    # next_adjacent_wp =next_adjacent_wp[0]
#
    #if adjacent_wp is None:
    #    print("No adjacent lane available in direction", direction)
    #    return waypoints
#
    #waypoints.append(adjacent_wp)
    #last_wp = adjacent_wp
    ## waypoints.append(next_adjacent_wp)
    ## last_wp = next_adjacent_wp

    lane_change_segment, location_segment = generate_lane_change_segment(last_wp, _map, forward_length=7.0,
                                                                        direction=direction, steps=10)

    waypoints.extend(lane_change_segment)
    locationPos.extend(location_segment)

    last_wp = lane_change_segment[-1]
    transitionIdx.append(len(waypoints)-1)

    # 3. Follow adjacent lane for overtaking distance.
    traveled = 0.0
    while traveled < overtake_distance:
        next_wps = last_wp.next(lane_change_step)
        if not next_wps:
            break
        next_wp = next_wps[0]
        loc = next_wp.transform.location
        curr_location = np.array([loc.x, loc.y, loc.z])
        traveled += next_wp.transform.location.distance(last_wp.transform.location)

        waypoints.append(next_wp)
        locationPos.append(curr_location)

        last_wp = next_wp

    transitionIdx.append(len(waypoints)-1)

    # 4. Merge back to the original lane.
    # if direction == "left":
    #     merging_wp = last_wp.get_right_lane()
    #     # next_merging_wp = merging_wp.next(lane_change_step)
    #     # next_merging_wp = next_merging_wp[0]
    # else:
    #     merging_wp = last_wp.get_left_lane()
    #     # next_merging_wp = merging_wp.next(lane_change_step)
    #     # next_merging_wp = next_merging_wp[0]
# 
    # if merging_wp is not None:
    #     waypoints.append(merging_wp)
    #     last_wp = merging_wp
    #     # waypoints.append(next_merging_wp)
    #     # last_wp = next_merging_wp

    if direction == "left":
        lane_change_segment, location_segment = generate_lane_change_segment(last_wp, _map, "right", forward_length=15.0)
    else:
        lane_change_segment, location_segment = generate_lane_change_segment(last_wp, _map, "left", forward_length=15.0)

    waypoints.extend(lane_change_segment)
    locationPos.extend(location_segment)

    last_wp = lane_change_segment[-1]
    transitionIdx.append(len(waypoints)-1)

    # 5. Follow the lane    
    traveled = 0.0
    while traveled < merge_distance:
        next_wps = last_wp.next(lane_change_step)
        if not next_wps:
            break
        next_wp = next_wps[0]
        loc = next_wp.transform.location
        curr_location = np.array([loc.x, loc.y, loc.z])
        traveled += next_wp.transform.location.distance(last_wp.transform.location)

        waypoints.append(next_wp)
        locationPos.append(curr_location)

        last_wp = next_wp

    return waypoints, locationPos, transitionIdx

def get_next_waypoint_from_list(waypoints, vehicle, current_index, threshold=2.0):
    """
    Return the next waypoint from 'waypoints' based on the vehicle's current position.

    Args:
        waypoints: List of generated overtaking waypoints.
        vehicle: The vehicle actor.
        current_index: The current index in the waypoint list.
        threshold (float): Distance threshold to consider the waypoint reached.

    Returns:
        (next_wp, updated_index)
    """
    vehicle_loc = vehicle.get_location()

    # If the vehicle is close enough to the current target,
    # advance to the next waypoint if available.
    if current_index < len(waypoints):
        wp = waypoints[current_index]

        # print("the distance to last idx wp is: ", vehicle_loc.distance(wp.transform.location))

        if vehicle_loc.distance(wp.transform.location) < threshold and current_index < len(waypoints)-1:
            current_index += 1
    else:
        current_index = len(waypoints) - 1  # stay on the last one
    return waypoints[current_index], current_index

def get_current_state(vehicle):
    """
    Get current vehicle state and return a np.array([x, y, psi, v, a]) 
    """
    # 1. Coordinate (x, y)
    location = vehicle.get_location()
    x = location.x
    y = location.y

    # 2. Heading angle ψ (yaw)
    yaw_deg = vehicle.get_transform().rotation.yaw
    yaw_rad = math.radians(yaw_deg)  # radians for math/control use

    # 3. Velocity vector → magnitude for scalar speed
    velocity_vector = vehicle.get_velocity()
    v = math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)

    # 4. Acceleration vector → magnitude for scalar acceleration
    accel_vector = vehicle.get_acceleration()
    a = math.sqrt(accel_vector.x**2 + accel_vector.y**2 + accel_vector.z**2)

    # 5. Pack into a NumPy array
    state_vec = np.array([x, y, yaw_rad, v, a])

    return state_vec

def LQR_Controller(x_cur, location_points, lookahead_idx, last_u, desired_vel = 30.0, stage = "STRAIGHT"):
    """
    Parameters:
        x_cur: np.array([x, y, psi, v, a]) - vehicle current state
        location_points: array of np.array([x, y, z]) - all location points
        lookahead_idx: int - idx of the reference point in location_points
        last_u: p.array([delta, u_a]) - last control command
        desired_vel: float - desired velocity in km/h
        stage: string - for gain schdule. psi low for straight line, psi high for tranistion line

    Returns:
        u_next: np.array([delta, u_a]) - next control command
        x_next: np.array - next state vector
        x0: np.array - reference location points
    """

    # ===== 1. Construct linearization point (x0, u0) =====
    x0 = np.zeros(5)
    u0 = np.zeros(2)
    next_location = location_points[lookahead_idx]

    x_target = next_location[0]
    y_target = next_location[1]

    x0[0] = x_target
    x0[1] = y_target
    if lookahead_idx + 1 < len(location_points):
        p0 = location_points[lookahead_idx]
        p1 = location_points[lookahead_idx + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        x0[2] = math.atan2(dy, dx)
    else:
        x0[2] = x_cur[2]
    x0[3] = desired_vel * 5 / 18  # convert km/h to m/s
    x0[4] = 0  # assume steady state

    u0[0] = 0  # nominal steering (can be refined)
    u0[1] = 0  # nominal acceleration input

    # Linearized system matrices
    A, B = vehicle_linear_dynamics(x0, u0)
    # A, B = vehicle_linear_dynamics(x_cur, last_u)

    # ===== 2. Solve Continuous Algebraic Riccati Equation (CARE) =====
    if stage == "STRAIGHT":
        Q = np.diag([10, 8, 5, 10, 5])
        R = np.diag([20, 1]) 
    else:
        Q = np.diag([10, 8, 7, 10, 7])   # weight on state errors
        R = np.diag([12, 1])             # weight on control effort
    
    x_hat = x_cur - x0       # state deviation
    x_hat[2] = wrap_to_pi(x_hat[2])     # make sure the angle difference is within (-pi ,pi)
    
    heading_err = abs(x_hat[2])     # penalise brakes more during large heading changes
    if heading_err > np.deg2rad(8):
        R[1,1] = 8.0
    else:
        R[1,1] = 1.0

    # Solve CARE: A'P + PA - PBR⁻¹B'P + Q = 0
    P = solve_continuous_are(A, B, Q, R)

    # Compute gain matrix: K = R⁻¹ Bᵀ P
    K = np.linalg.inv(R) @ B.T @ P

    # ===== 3. Feedback control =====
    u_hat = -K @ x_hat       # control deviation from nominal
    u_next = u0 + u_hat      # actual control input
    u_next[0] = np.clip(u_next[0], -0.4, 0.4)      # ±23°
    u_next[1] = np.clip(u_next[1], -2.0, 2.0)      # throttle / brake
    # u_next[0] = 0.7 * last_u[0] + 0.3 * u_next[0]


    # ===== 4. Update state =====
    dxdt = nonlinear_dynamics(x_cur, u_next)
    x_next = x_cur + dt * dxdt

    return u_next, x_next, x0

def convert2Carla(u_next, max_steer=0.4, max_accel=3.0, max_decel=5.0):
    """
    Convert LQR controller output to carla.VehicleControl()

    Parameters:
    - u_next: np.array([delta, u_a]) -- LQR controller output 
    - max_steer: maximum steering angle in radians (used for normalization)
    - max_accel: maximum throttle acceleration (m/s^2)
    - max_decel: maximum brake deceleration (m/s^2)

    Returns:
    - control: carla.VehicleControl() -- control input for Carla
    """
    delta, u_a = u_next
    control = carla.VehicleControl()

    # Clamp steering to [-max_steer, max_steer], then normalize to [-1, 1]
    steer_cmd = np.clip(delta, -max_steer, max_steer) / max_steer
    control.steer = float(np.clip(steer_cmd, -1.0, 1.0))

    # Acceleration to throttle/brake
    if u_a >= 0:
        control.throttle = float(np.clip(u_a / max_accel, 0.0, 1.0))
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = float(np.clip(-u_a / max_decel, 0.0, 1.0))

    return control



if __name__ == "__main__":

    # Debug
    log_file = open("lqr_tracking_log.txt", "w")
    atexit.register(log_file.close)
    log_file.write("==== LQR Tracking Log Start ====\n")

    carla_manager = CarlaManager()
    print("CarlaManager is created")
    try:
        preceding_vehicle_actor = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
        preceding_vehicle_actor.set_autopilot(False)
        time.sleep(1)  # allow the vehicle to spawn
        carla_manager.world.tick()

        # Wrap with Vehicle class
        preceding_vehicle = Vehicle(preceding_vehicle_actor)

        # start autopilot controller for preciding vehicle
        agent = BasicAgent(preceding_vehicle_actor, target_speed=10)
        # set destination for the preceding vehicle
        current_location = preceding_vehicle_actor.get_location()
        current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)
        next_wps = current_wp.next(100.0)  
        if next_wps:  # if there is a waypoint ahead
            destination = next_wps[0].transform.location
        else:
            destination = current_location
        agent.set_destination(destination)  # choose a destination appropriately
        # spawn ego vehicle
        SPAWN_LOCATION[0] += 20
        ego_vehicle_actor = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
        time.sleep(1)  # allow the vehicle to spawn
        carla_manager.world.tick()

        # generate overtaking waypoints
        waypoints, location_points, transitionIdx = generate_overtake_waypoints(carla_manager, ego_vehicle_actor,
                                                direction="left",
                                                distance_current_lane=10.0,
                                                lane_change_step=1.0,
                                                overtake_distance=50.0,
                                                merge_distance=50.0)

        # For debugging, you can visualize the waypoints:
        # carla_manager.debug_waypoints(waypoints)
        carla_manager.debug_np_locations(location_points)

        current_wp_index = 0
        u_next = np.zeros(2)
        stage = "STRAIGHT"
        desired_vel = 25.0

        # Initialize data collection before the main loop
        import os
        from datetime import datetime
        os.makedirs('results', exist_ok=True)
        simulation_data = []
        sim_time = 0.0

        # Wrap with Vehicle class
        ego_vehicle = Vehicle(ego_vehicle_actor)

        # Save spawn transform and initialize transformation matrices
        ego_vehicle.transform_to_spawn = ego_vehicle_actor.get_transform()
        world_to_ego, ego_to_world = ego_vehicle.get_transform_matrices()

        try:
            # main loop:
            while True:
                # Control for the preceding vehicle
                control_cmd = agent.run_step()
                preceding_vehicle_actor.apply_control(control_cmd)
                
                # LQR for the ego vehicle
                next_overtake_wp, current_wp_index = get_next_waypoint_from_list(waypoints, ego_vehicle.actor, current_wp_index, threshold=2.0)
                lookahead_idx = min(current_wp_index + lookahead_offset, len(location_points)-1)
                next_overtake_location = location_points[lookahead_idx]
                
                # Get current state
                x_cur = get_current_state(ego_vehicle.actor)

                # Get control w.r.t. next_wp and current state
                if (lookahead_idx == transitionIdx[0]) or (lookahead_idx == transitionIdx[2]):  # start transition
                    stage = "TRANSITION"
                elif (lookahead_idx == transitionIdx[1]) or (lookahead_idx == transitionIdx[3]):  # start straight
                    stage = "STRAIGHT"

                dist = abs(ego_vehicle.actor.get_location().x - preceding_vehicle.actor.get_location().x)
                lateral_dis = abs(ego_vehicle.actor.get_location().y - preceding_vehicle.actor.get_location().y)
                if dist < safe_distance and lateral_dis < 1.0 and lookahead_idx < transitionIdx[0]:
                    desired_vel = 15.0  
                else:
                    desired_vel = 25.0

                u_next, x_next, x0 = LQR_Controller(x_cur, location_points, lookahead_idx, u_next, desired_vel, stage)

                control = convert2Carla(u_next)
                ego_vehicle.actor.apply_control(control)

                # Save the state and control input as a csv file for plotting later
                sim_time += dt
                
                # Get ego vehicle state in ego coordinates
                ego_state = ego_vehicle.get_vehicle_state()
                
                # Get preceding vehicle position in ego coordinates
                preceding_location = preceding_vehicle_actor.get_location()
                preceding_pos_in_ego = ego_vehicle.world_to_ego_coordinates(preceding_location)
                
                # Get target waypoint in ego coordinates
                wp_location = carla.Location(x=next_overtake_location[0], 
                                            y=next_overtake_location[1], 
                                            z=next_overtake_location[2])
                wp_pos_in_ego = ego_vehicle.world_to_ego_coordinates(wp_location)
                
                # Create data record
                data_point = {
                    'time': sim_time,
                    'ego_x': ego_state[0],
                    'ego_y': ego_state[1],
                    'ego_heading': ego_state[2],
                    'ego_speed': ego_state[3],
                    'ego_accel': ego_state[4],
                    'preceding_x': preceding_pos_in_ego[0],
                    'preceding_y': preceding_pos_in_ego[1],
                    'control_throttle': control.throttle,
                    'control_brake': control.brake,
                    'control_steer': control.steer,
                    'target_waypoint_x': wp_pos_in_ego[0],
                    'target_waypoint_y': wp_pos_in_ego[1],
                    'lqr_delta_cmd': u_next[0],
                    'lqr_accel_cmd': u_next[1]
                }
                
                # Append to data collection
                simulation_data.append(data_point)
                
                # Check if target position reached (100m in ego x-direction)
                if ego_state[0] >= 100.0:
                    print(f"Target position reached at {ego_state[0]:.2f}m in ego frame. Ending simulation.")
                    
                    # Save data to CSV
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f'results/overtaking_simulation_{timestamp}_LQR.csv'
                    
                    with open(csv_filename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=data_point.keys())
                        writer.writeheader()
                        for data_row in simulation_data:
                            writer.writerow(data_row)
                    
                    print(f"Data saved to {csv_filename}")
                    break
                    
                carla_manager.world.tick()
                time.sleep(0.05)


                log_both("=" * 50, log_file)
                log_both(f"x_cur: x = {x_cur[0]:.2f}, y = {x_cur[1]:.2f}, ψ = {np.degrees(x_cur[2]):.1f}°, v = {x_cur[3]:.2f} m/s", log_file)
                log_both(f"target (linearization center): x = {x0[0]:.2f}, y = {x0[1]:.2f}, ψ = {np.degrees(x0[2]):.1f}°", log_file)
                log_both(f"x_next: x = {x_next[0]:.2f}, y = {x_next[1]:.2f}, ψ = {np.degrees(x_next[2]):.1f}°", log_file)
                log_both(f"distance to the leading vehicle: {dist}, lateral distance: {lateral_dis}, current desired vel: {desired_vel* 5 / 18}", log_file)
                log_both(f"index: {lookahead_idx}, current stage: {stage}", log_file)
                dpsi = wrap_to_pi(x_cur[2] - x0[2])
                log_both(f"error: dx = {x_cur[0] - next_overtake_location[0]:.2f}, dy = {x_cur[1] - next_overtake_location[1]:.2f}, dpsi = {np.degrees(dpsi):.1f}", log_file)
                log_both("=" * 50, log_file)

        except KeyboardInterrupt:
            print("\nSimulation terminated by user")

    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()

    finally:
        # Save collected data to CSV if simulation was interrupted
        if 'simulation_data' in locals() and len(simulation_data) > 0:

            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'results/overtaking_simulation_{timestamp}_LQR.csv'
            
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=simulation_data[0].keys())
                writer.writeheader()
                for data_row in simulation_data:
                    writer.writerow(data_row)
            
            print(f"Simulation data saved to {csv_filename}")
        
        # Cleanup
        if 'preceding_vehicle_actor' in locals() and preceding_vehicle_actor is not None:
            preceding_vehicle_actor.destroy()
        if 'ego_vehicle_actor' in locals() and ego_vehicle_actor is not None:
            ego_vehicle_actor.destroy()
        carla_manager.__del__()
        log_both("==== LQR Tracking Log End ====", log_file)
        log_file.close()
        print("Simulation completed and log file closed.")