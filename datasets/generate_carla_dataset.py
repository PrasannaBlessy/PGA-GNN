"""
PGA-GNN
CARLA Synthetic Adverse-Condition Dataset Generation

CARLA version:
    0.9.15

Purpose:
    Generate synthetic driving images under different visual
    conditions for PGA-GNN lane-detection experiments.

The script generates:
    - RGB camera images
    - Semantic-segmentation images
    - Frame metadata
    - Weather/condition information
    - Vehicle and lane information

Target:
    3,000 synthetic frames

Conditions:
    - Clear daytime
    - Cloudy
    - Rain
    - Fog
    - Night
    - Rain + fog
"""

import carla

import argparse
import json
import os
import queue
import random
import time


# ============================================================
# Configuration
# ============================================================

DEFAULT_FRAMES = 3000

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
CAMERA_FOV = 90

RANDOM_SEED = 42


# ============================================================
# Weather profiles
# ============================================================

WEATHER_PROFILES = {

    "clear": carla.WeatherParameters(
        cloudiness=5.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=30.0,
        sun_altitude_angle=60.0,
        fog_density=0.0,
        fog_distance=100.0
    ),

    "cloudy": carla.WeatherParameters(
        cloudiness=70.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        sun_azimuth_angle=30.0,
        sun_altitude_angle=45.0,
        fog_density=5.0,
        fog_distance=80.0
    ),

    "rain": carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=70.0,
        precipitation_deposits=60.0,
        wind_intensity=20.0,
        sun_azimuth_angle=20.0,
        sun_altitude_angle=35.0,
        fog_density=5.0,
        fog_distance=70.0
    ),

    "fog": carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=20.0,
        sun_altitude_angle=30.0,
        fog_density=70.0,
        fog_distance=25.0
    ),

    "night": carla.WeatherParameters(
        cloudiness=30.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=180.0,
        sun_altitude_angle=-20.0,
        fog_density=5.0,
        fog_distance=70.0
    ),

    "rain_fog": carla.WeatherParameters(
        cloudiness=90.0,
        precipitation=80.0,
        precipitation_deposits=70.0,
        wind_intensity=15.0,
        sun_azimuth_angle=10.0,
        sun_altitude_angle=20.0,
        fog_density=55.0,
        fog_distance=30.0
    )
}


# ============================================================
# Utility functions
# ============================================================

def create_directories(output_dir):

    rgb_dir = os.path.join(
        output_dir,
        "rgb"
    )

    semantic_dir = os.path.join(
        output_dir,
        "semantic"
    )

    os.makedirs(
        rgb_dir,
        exist_ok=True
    )

    os.makedirs(
        semantic_dir,
        exist_ok=True
    )

    return rgb_dir, semantic_dir


def choose_weather(frame_id):

    conditions = list(
        WEATHER_PROFILES.keys()
    )

    # Approximately balanced distribution
    condition = conditions[
        frame_id % len(conditions)
    ]

    return condition


def get_lane_information(vehicle, world):

    """
    Obtain lane/road information from the current
    ego-vehicle position.

    CARLA waypoints provide:
        - road ID
        - lane ID
        - lane width
        - intersection information
    """

    carla_map = world.get_map()

    location = vehicle.get_location()

    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if waypoint is None:

        return {
            "road_id": None,
            "lane_id": None,
            "lane_width": None,
            "is_intersection": None
        }

    return {
        "road_id": waypoint.road_id,
        "lane_id": waypoint.lane_id,
        "lane_width": waypoint.lane_width,
        "is_intersection": waypoint.is_intersection
    }


def save_metadata(
    metadata,
    output_dir
):

    metadata_file = os.path.join(
        output_dir,
        "metadata.jsonl"
    )

    with open(
        metadata_file,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(metadata)
            + "\n"
        )


# ============================================================
# Main generation function
# ============================================================

def generate_dataset(
    host,
    port,
    output_dir,
    num_frames
):

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Connect to CARLA
    # --------------------------------------------------------

    print(
        f"Connecting to CARLA at "
        f"{host}:{port} ..."
    )

    client = carla.Client(
        host,
        port
    )

    client.set_timeout(
        20.0
    )

    world = client.get_world()

    print(
        "Connected to CARLA"
    )

    print(
        "CARLA server version:",
        client.get_server_version()
    )

    # --------------------------------------------------------
    # Save original settings
    # --------------------------------------------------------

    original_settings = (
        world.get_settings()
    )

    # --------------------------------------------------------
    # Enable synchronous simulation
    # --------------------------------------------------------

    settings = world.get_settings()

    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05

    world.apply_settings(
        settings
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    rgb_dir, semantic_dir = (
        create_directories(
            output_dir
        )
    )

    # --------------------------------------------------------
    # Actor containers
    # --------------------------------------------------------

    actors = []

    rgb_queue = queue.Queue()
    semantic_queue = queue.Queue()

    try:

        # ====================================================
        # Vehicle
        # ====================================================

        blueprint_library = (
            world.get_blueprint_library()
        )

        vehicle_blueprints = (
            blueprint_library.filter(
                "vehicle.*"
            )
        )

        # Prefer common passenger vehicles
        preferred = [
            bp for bp in vehicle_blueprints
            if "tesla" in bp.id.lower()
            or "audi" in bp.id.lower()
            or "ford" in bp.id.lower()
            or "lincoln" in bp.id.lower()
        ]

        if preferred:

            vehicle_bp = random.choice(
                preferred
            )

        else:

            vehicle_bp = random.choice(
                vehicle_blueprints
            )

        # ----------------------------------------------------
        # Spawn point
        # ----------------------------------------------------

        spawn_points = (
            world.get_map()
            .get_spawn_points()
        )

        if not spawn_points:

            raise RuntimeError(
                "No vehicle spawn points "
                "available in the current map."
            )

        random.shuffle(
            spawn_points
        )

        vehicle = None

        for spawn_point in spawn_points:

            vehicle = world.try_spawn_actor(
                vehicle_bp,
                spawn_point
            )

            if vehicle is not None:
                break

        if vehicle is None:

            raise RuntimeError(
                "Unable to spawn ego vehicle."
            )

        actors.append(
            vehicle
        )

        print(
            "Ego vehicle spawned:",
            vehicle.type_id
        )

        # Enable autopilot
        vehicle.set_autopilot(
            True
        )

        # ====================================================
        # RGB Camera
        # ====================================================

        camera_bp = (
            blueprint_library.find(
                "sensor.camera.rgb"
            )
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(IMAGE_WIDTH)
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(IMAGE_HEIGHT)
        )

        camera_bp.set_attribute(
            "fov",
            str(CAMERA_FOV)
        )

        camera_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        camera_transform = carla.Transform(
            carla.Location(
                x=1.5,
                z=1.6
            )
        )

        rgb_camera = (
            world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=vehicle
            )
        )

        actors.append(
            rgb_camera
        )

        rgb_camera.listen(
            rgb_queue.put
        )

        # ====================================================
        # Semantic segmentation camera
        # ====================================================

        semantic_bp = (
            blueprint_library.find(
                "sensor.camera.semantic_segmentation"
            )
        )

        semantic_bp.set_attribute(
            "image_size_x",
            str(IMAGE_WIDTH)
        )

        semantic_bp.set_attribute(
            "image_size_y",
            str(IMAGE_HEIGHT)
        )

        semantic_bp.set_attribute(
            "fov",
            str(CAMERA_FOV)
        )

        semantic_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        semantic_camera = (
            world.spawn_actor(
                semantic_bp,
                camera_transform,
                attach_to=vehicle
            )
        )

        actors.append(
            semantic_camera
        )

        semantic_camera.listen(
            semantic_queue.put
        )

        print(
            "Camera sensors initialized."
        )

        # ====================================================
        # Main frame-generation loop
        # ====================================================

        generated = 0

        while generated < num_frames:

            # ------------------------------------------------
            # Select condition
            # ------------------------------------------------

            condition = choose_weather(
                generated
            )

            weather = (
                WEATHER_PROFILES[
                    condition
                ]
            )

            world.set_weather(
                weather
            )

            # ------------------------------------------------
            # Advance simulation
            # ------------------------------------------------

            world.tick()

            try:

                rgb_image = (
                    rgb_queue.get(
                        timeout=5.0
                    )
                )

                semantic_image = (
                    semantic_queue.get(
                        timeout=5.0
                    )
                )

            except queue.Empty:

                print(
                    "WARNING: camera timeout."
                )

                continue

            # ------------------------------------------------
            # File names
            # ------------------------------------------------

            frame_name = (
                f"{generated:06d}"
            )

            rgb_path = os.path.join(
                rgb_dir,
                frame_name
            )

            semantic_path = os.path.join(
                semantic_dir,
                frame_name
            )

            # ------------------------------------------------
            # Save RGB
            # ------------------------------------------------

            rgb_image.save_to_disk(
                rgb_path
            )

            # ------------------------------------------------
            # Save semantic image
            # ------------------------------------------------

            semantic_image.save_to_disk(
                semantic_path,
                carla.ColorConverter.CityScapesPalette
            )

            # ------------------------------------------------
            # Lane information
            # ------------------------------------------------

            lane_info = (
                get_lane_information(
                    vehicle,
                    world
                )
            )

            # ------------------------------------------------
            # Vehicle information
            # ------------------------------------------------

            transform = (
                vehicle.get_transform()
            )

            velocity = (
                vehicle.get_velocity()
            )

            speed = (
                3.6
                * (
                    velocity.x ** 2
                    + velocity.y ** 2
                    + velocity.z ** 2
                ) ** 0.5
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {

                "frame_id":
                    generated,

                "rgb_image":
                    f"rgb/{frame_name}.png",

                "semantic_image":
                    f"semantic/{frame_name}.png",

                "condition":
                    condition,

                "map":
                    world.get_map().name,

                "vehicle":
                    vehicle.type_id,

                "location": {
                    "x":
                        transform.location.x,
                    "y":
                        transform.location.y,
                    "z":
                        transform.location.z
                },

                "rotation": {
                    "pitch":
                        transform.rotation.pitch,
                    "yaw":
                        transform.rotation.yaw,
                    "roll":
                        transform.rotation.roll
                },

                "speed_kmh":
                    speed,

                "road_id":
                    lane_info["road_id"],

                "lane_id":
                    lane_info["lane_id"],

                "lane_width":
                    lane_info["lane_width"],

                "is_intersection":
                    lane_info[
                        "is_intersection"
                    ]
            }

            save_metadata(
                metadata,
                output_dir
            )

            generated += 1

            if generated % 100 == 0:

                print(
                    f"Generated "
                    f"{generated}/"
                    f"{num_frames} frames"
                )

    finally:

        print(
            "Cleaning up CARLA actors..."
        )

        for actor in actors:

            if actor is not None:

                try:
                    actor.destroy()

                except Exception:
                    pass

        # Restore original world settings
        try:

            world.apply_settings(
                original_settings
            )

        except Exception:
            pass

        print(
            "CARLA cleanup completed."
        )

    print()
    print(
        "========================================"
    )
    print(
        "CARLA DATASET GENERATION COMPLETED"
    )
    print(
        "========================================"
    )
    print(
        f"Frames generated: {num_frames}"
    )
    print(
        f"Output directory: {output_dir}"
    )
    print(
        "========================================"
    )


# ============================================================
# Command-line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic adverse-condition "
            "driving images using CARLA."
        )
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="CARLA server host."
    )

    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port."
    )

    parser.add_argument(
        "--output",
        default=(
            "pga_gnn_data/carla"
        ),
        help="Output directory."
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=(
            "Number of synthetic frames "
            "to generate."
        )
    )

    args = parser.parse_args()

    generate_dataset(
        host=args.host,
        port=args.port,
        output_dir=args.output,
        num_frames=args.frames
    )


if __name__ == "__main__":
    main()
