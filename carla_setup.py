import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # just constant velocity no lateral control
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.controller import VehiclePIDController
import time
from hydra import initialize, compose

dt = 0.1  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

# synchronous_mode will make the simulation predictable
synchronous_mode = True


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
        if vehicle_loc.distance(wp.transform.location) < threshold and current_index < len(waypoints)-1:
            current_index += 1
    else:
        current_index = len(waypoints) - 1  # stay on the last one
    return waypoints[current_index], current_index


if __name__ == "__main__":
    carla_manager = CarlaManager()
    print("CarlaManager is created")
    preceding_vehicle = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
    preceding_vehicle.set_autopilot(False)
    time.sleep(1)  # allow the vehicle to spawn

    # start autopilot controller for preciding vehicle
    agent = BasicAgent(preceding_vehicle, target_speed=10)

    # set destination for the preceding vehicle
    current_location = preceding_vehicle.get_location()
    current_wp = carla_manager.map.get_waypoint(current_location, project_to_road=True)
    next_wps = current_wp.next(100.0)  # 100 meters ahead

    if next_wps:  # if there is a waypoint ahead
        destination = next_wps[0].transform.location
    else:
        destination = current_location
    agent.set_destination(destination)  # choose a destination appropriately

    # spawn ego vehicle
    SPAWN_LOCATION[0] += 10
    ego_vehicle = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
    time.sleep(1)  # allow the vehicle to spawn

    # TODO fix the waypoints for smooth lane change
    # generate overtaking waypoints
    waypoints = generate_overtake_waypoints(carla_manager, ego_vehicle,
                                            direction="left",
                                            distance_current_lane=10.0,
                                            lane_change_step=1.0,
                                            overtake_distance=100.0,
                                            merge_distance=10.0)
    # For debugging, you can visualize the waypoints:
    carla_manager.debug_waypoints(waypoints)
    pid_controller = VehiclePIDController(ego_vehicle, {'K_P': 0.9}, {'K_P': 0.01, 'K_I': 0.0, 'K_D': 0.5})

    current_wp_index = 0
    # main loop
    while True:
        control_cmd = agent.run_step()
        preceding_vehicle.apply_control(control_cmd)

        # PID controller for ego vehicle -- not working yet
        next_overtake_wp, current_wp_index = get_next_waypoint_from_list(waypoints, ego_vehicle, current_wp_index, threshold=2.0)
        ego_control = pid_controller.run_step(20, next_overtake_wp)  # target speed 30 km/h
        ego_vehicle.apply_control(ego_control)
        # print('Specator location:', carla_manager.spectator.get_transform().location, 'rotation:', carla_manager.spectator.get_transform().rotation)
        carla_manager.world.tick()
        # time.sleep(0.05)
