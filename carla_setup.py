import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # just constant velocity no lateral control
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.controller import VehiclePIDController
import time
from hydra import initialize, compose
import numpy as np
import math
import atexit

dt = 0.1  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

# synchronous_mode will make the simulation predictable
synchronous_mode = True


def log_both(msg, log_file):
    """
    log for debug
    """
    print(msg)
    print(msg, file=log_file)


class SimplePIDController:
    def __init__(self, vehicle, K_lat=1.0, Kp=1.0, Ki=0.0, Kd=0.0):
        self.vehicle = vehicle
        self.K_lat = K_lat  # lateral gain (steering)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._e_integral = 0
        self._prev_error = 0

    def get_yaw_from_vehicle(self):
        return math.radians(self.vehicle.get_transform().rotation.yaw)

    def get_speed(self):
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def run_step(self, target_speed, target_location, target_yaw=None, dt=0.1):
        # === Longitudinal control ===
        target_speed = target_speed * 5 / 18    # convert km/h to to m/s
        current_speed = self.get_speed()
        speed_error = target_speed - current_speed

        self._e_integral += speed_error * dt
        d_error = (speed_error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = speed_error

        throttle = self.Kp * speed_error + self.Ki * self._e_integral + self.Kd * d_error
        throttle = max(0.0, min(throttle, 1.0))  # Clamp between 0-1
        brake = 0.0 if speed_error >= 0 else min(-throttle, 1.0)

        # === Lateral control (simple pure pursuit-ish P controller) ===
        vehicle_loc = self.vehicle.get_location()
        dx = target_location.x - vehicle_loc.x
        dy = target_location.y - vehicle_loc.y
        heading = self.get_yaw_from_vehicle()

        # heading error
        target_yaw_est = target_yaw
        if target_yaw_est is None:
            target_yaw_est = math.atan2(dy, dx)

        yaw_error = target_yaw_est - heading
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi  # normalize to [-π, π]

        steer = self.K_lat * yaw_error
        steer = max(-1.0, min(steer, 1.0))  # Clamp

        # === Output control ===
        control = carla.VehicleControl()
        control.throttle = throttle if speed_error >= 0 else 0.0
        control.brake = brake
        control.steer = steer

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


def generate_lane_change_segment(last_wp, carla_map, direction="left", forward_length=20.0, steps=10):
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

    lane_change_segment, location_segment = generate_lane_change_segment(last_wp, _map, direction)

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
        lane_change_segment, location_segment = generate_lane_change_segment(last_wp, _map, "right")
    else:
        lane_change_segment, location_segment = generate_lane_change_segment(last_wp, _map, "left")

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


# def generate_overtake_waypoints(carla_manager, vehicle, direction="left",
#                                 distance_current_lane=10.0,
#                                 lane_change_step=2.0,
#                                 overtake_distance=50.0,
#                                 merge_distance=10.0):
#     """
#     Generate a list of waypoints for an overtaking maneuver.
#
#     Args:
#         carla_manager: Instance of CarlaManager containing the map.
#         vehicle: The vehicle actor (rear vehicle).
#         direction (str): "left" or "right" lane change direction.
#         distance_current_lane (float): Distance to follow in current lane before lane change.
#         lane_change_step (float): Interval (in meters) to sample waypoints.
#         overtake_distance (float): Distance to follow in the adjacent lane.
#         merge_distance (float): Distance to follow in merging lane after overtaking.
#
#     Returns:
#         List of carla.Waypoint objects forming the overtaking trajectory.
#     """
#     _map = carla_manager.map
#     waypoints = []
#
#     # 1. Start in current lane.
#     current_wp = _map.get_waypoint(vehicle.get_location(), project_to_road=True)
#     waypoints.append(current_wp)
#
#     # Follow current lane for distance_current_lane
#     traveled = 0.0
#     last_wp = current_wp
#     while traveled < distance_current_lane:
#         next_wps = last_wp.next(lane_change_step)
#         if not next_wps:
#             break
#         next_wp = next_wps[0]
#         traveled += next_wp.transform.location.distance(last_wp.transform.location)
#         waypoints.append(next_wp)
#         last_wp = next_wp
#
#     # 2. Change lane: get the adjacent lane from the last waypoint
#         # 45 degree line segment spacing
#     if direction == "left":
#         adjacent_wp = last_wp.get_left_lane()
#         # next_adjacent_wp = adjacent_wp.next(lane_change_step)
#         # next_adjacent_wp =next_adjacent_wp[0]
#     else:
#         adjacent_wp = last_wp.get_right_lane()
#         # next_adjacent_wp = adjacent_wp.next(lane_change_step)
#         # next_adjacent_wp =next_adjacent_wp[0]
#
#     if adjacent_wp is None:
#         print("No adjacent lane available in direction", direction)
#         return waypoints
#
#     waypoints.append(adjacent_wp)
#     last_wp = adjacent_wp
#     # waypoints.append(next_adjacent_wp)
#     # last_wp = next_adjacent_wp
#
#     # 3. Follow adjacent lane for overtaking distance.
#     traveled = 0.0
#     while traveled < overtake_distance:
#         next_wps = last_wp.next(lane_change_step)
#         if not next_wps:
#             break
#         next_wp = next_wps[0]
#         traveled += next_wp.transform.location.distance(last_wp.transform.location)
#         waypoints.append(next_wp)
#         last_wp = next_wp
#
#     # 4. Merge back to the original lane.
#         # 45 degree line segment spaceing
#     if direction == "left":
#         merging_wp = last_wp.get_right_lane()
#         # next_merging_wp = merging_wp.next(lane_change_step)
#         # next_merging_wp = next_merging_wp[0]
#     else:
#         merging_wp = last_wp.get_left_lane()
#         # next_merging_wp = merging_wp.next(lane_change_step)
#         # next_merging_wp = next_merging_wp[0]
#
#     if merging_wp is not None:
#         waypoints.append(merging_wp)
#         last_wp = merging_wp
#         # waypoints.append(next_merging_wp)
#         # last_wp = next_merging_wp
#
#         traveled = 0.0
#         while traveled < merge_distance:
#             next_wps = last_wp.next(lane_change_step)
#             if not next_wps:
#                 break
#             next_wp = next_wps[0]
#             traveled += next_wp.transform.location.distance(last_wp.transform.location)
#             waypoints.append(next_wp)
#             last_wp = next_wp
#
#     return waypoints


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
        print(vehicle_loc.distance(wp.transform.location))
        if vehicle_loc.distance(wp.transform.location) < threshold and current_index < len(waypoints)-1:
            current_index += 1
    else:
        current_index = len(waypoints) - 1  # stay on the last one
    return waypoints[current_index], current_index


if __name__ == "__main__":

    # Debug
    log_file = open("pid_tracking_log.txt", "w")
    atexit.register(log_file.close)
    log_file.write("==== PID Tracking Log Start ====\n")

    carla_manager = CarlaManager()
    print("CarlaManager is created")
    preceding_vehicle = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
    preceding_vehicle.set_autopilot(False)
    time.sleep(1)  # allow the vehicle to spawn
    carla_manager.world.tick()

    # start autopilot controller for preciding vehicle
    agent = BasicAgent(preceding_vehicle, target_speed=10)

    # set destination for the preceding vehicle
    current_location = preceding_vehicle.get_location()
    current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)
    next_wps = current_wp.next(130.0)  # 100 meters ahead

    if next_wps:  # if there is a waypoint ahead
        destination = next_wps[0].transform.location
    else:
        destination = current_location

    agent.set_destination(destination)  # choose a destination appropriately

    # spawn ego vehicle
    SPAWN_LOCATION[0] += 10
    ego_vehicle = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
    time.sleep(1)  # allow the vehicle to spawn
    carla_manager.world.tick()  # tick the world to sync
    # TODO fix the waypoints for smooth lane change
    # generate overtaking waypoints
    waypoints, location_points, transitionIdx = generate_overtake_waypoints(carla_manager, ego_vehicle,
                                            direction="left",
                                            distance_current_lane=10.0,
                                            lane_change_step=1.0,
                                            overtake_distance=100.0,
                                            merge_distance=10.0)
    # For debugging, you can visualize the waypoints:
    # carla_manager.debug_waypoints(waypoints)
    carla_manager.debug_np_locations(location_points)


    # pid_controller = VehiclePIDController(ego_vehicle, {'K_P': 6.0, 'K_I':0.5, 'K_D':0.2}, {'K_P': 0.01, 'K_I': 0.0, 'K_D': 0.5})
    pid_controller = SimplePIDController(ego_vehicle, K_lat=0.5, Kp=3.0, Ki=0.5, Kd=0.2)

    current_wp_index = 0
    # main loop
    try:
        while True:
            control_cmd = agent.run_step()
            preceding_vehicle.apply_control(control_cmd)

            # PID controller for ego vehicle -- not working yet
            next_overtake_wp, current_wp_index = get_next_waypoint_from_list(waypoints, ego_vehicle, current_wp_index, threshold=2.0)

            if current_wp_index > 0:
                prev_point = location_points[current_wp_index - 1]
            else:
                prev_point = location_points[current_wp_index]
            next_point = location_points[current_wp_index]

            target_location = carla.Location(x=next_point[0], y=next_point[1], z=next_point[2])

            # estimate yaw
            dx = next_point[0] - prev_point[0]
            dy = next_point[1] - prev_point[1]
            target_yaw = math.atan2(dy, dx)

            # ego_control = pid_controller.run_step(20, next_overtake_wp)  # target speed 30 km/h
            ego_control = pid_controller.run_step(20, target_location, target_yaw, dt)

            ego_vehicle.apply_control(ego_control)
            # print('Specator location:', carla_manager.spectator.get_transform().location, 'rotation:', carla_manager.spectator.get_transform().rotation)
            carla_manager.world.tick()
            time.sleep(0.05)

            ## debug
            v1 = preceding_vehicle.get_velocity()
            v2 = ego_vehicle.get_velocity()
            speed1 = (v1.x**2 + v1.y**2 + v1.z**2)**0.5  # km/h
            speed2 = (v2.x**2 + v2.y**2 + v2.z**2)**0.5

            log_both("=" * 50, log_file)
            log_both(f"[EGO CONTROL] throttle: {ego_control.throttle:.2f}, steer: {ego_control.steer:.2f}, brake: {ego_control.brake:.2f}", log_file)
            log_both(f"Preceding speed: {speed1:.2f} km/h | Ego speed: {speed2:.2f} km/h", log_file)

            log_both("=" * 50, log_file)

    except KeyboardInterrupt:
        carla_manager.__del__()
        print('Closing Carla')

