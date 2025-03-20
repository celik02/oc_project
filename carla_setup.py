import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # just constant velocity no lateral control
from agents.navigation.basic_agent import BasicAgent
import time


dt = 0.1  # seconds
SPAWN_LOCATION = [
    111.229362,
    13.511219,
    14.171306,
]  # for spectator camera

# synchronous_mode will make the simulation predictable
synchronous_mode = False


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
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors()])

    def spawn_vehicle(self, blueprint_name, spawn_point):
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


if __name__ == "__main__":
    carla_manager = CarlaManager()
    print("CarlaManager is created")
    preceding_vehicle = carla_manager.spawn_vehicle("vehicle.tesla.model3", SPAWN_LOCATION)
    preceding_vehicle.set_autopilot(False)
    time.sleep(1)  # allow the vehicle to spawn

    # start autopilot controller for preciding vehicle
    agent = BasicAgent(preceding_vehicle)

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

    # generate overtaking waypoints
    waypoints = generate_overtake_waypoints(carla_manager, ego_vehicle,
                                            direction="left",
                                            distance_current_lane=10.0,
                                            lane_change_step=2.0,
                                            overtake_distance=50.0,
                                            merge_distance=10.0)
    # For debugging, you can visualize the waypoints:
    carla_manager.debug_waypoints(waypoints)

    # main loop
    while True:
        control_cmd = agent.run_step()
        preceding_vehicle.apply_control(control_cmd)
        # print('Specator location:', carla_manager.spectator.get_transform().location, 'rotation:', carla_manager.spectator.get_transform().rotation)
        time.sleep(0.05)
