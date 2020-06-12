# coding=utf-8
"""After starting a carla server off-sceen, use this to connect as spectator."""

from __future__ import print_function

import argparse
import cv2
import datetime
import json
import math
import sys
import os
import glob
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla
import pygame
import time

from utils import setup_static
from utils import setup_walker_bps
from utils import setup_vehicle_bps
from utils import anchor_cameras_annotation
from utils import get_bp
from utils import control_data_to_traj
from utils import compute_intrinsic
from utils import cross
from utils import parse_carla_depth
from utils import compute_extrinsic_from_transform
from utils import get_degree_of_two_vectors
from utils import save_rgb_image
from utils import get_3d_bbox
from utils import make_text_surface
from utils import get_direction_and_speed
from utils import run_sim_for_one_frame
from utils import cleanup_actors
from utils import interpolate_controls
from utils import reset_x_agent_key
from utils import reset_bps
from utils import realism_weather

parser = argparse.ArgumentParser()

parser.add_argument("moment_file_list",
                    help="file list to the moment json file, assuming each "
                         " filename is scene.*.json format")
parser.add_argument("annotation_file", help="path to save json annotation "
                                            "file.")

parser.add_argument("log_file", help="json file to save the failures and "
                                     "other logs.")

# interface setup
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--res", default="1280x720")

parser.add_argument("--is_actev", action="store_true")



# moment setup
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--annotation_fps", type=float, default=2.5)
parser.add_argument("--obs_length", type=int, default=12,
                    help="observation timestep, 12/2.5=4.8 seconds")
parser.add_argument("--pred_length", type=int, default=26,
                    help="observation timestep, 26/2.5=10.4 seconds")

# annotation setup
parser.add_argument("--start_idx", type=int, default=0,
                    help="change this then the job split will be inconsistent.")
parser.add_argument("--job", type=int, default=1, help="total job")
parser.add_argument("--curJob", type=int, default=1,
                    help="split the moment for this annotator.")


class Camera(object):
  """Camera object to have a surface."""
  def __init__(self, camera_actor, camera_type="rgb",
               image_type=carla.ColorConverter.Raw):
    self.camera_actor = camera_actor
    self.image_type = image_type

    self.last_image_frame_num = None  # the frame num of the image
    self.last_image_seconds = None  # the seconds since beginning of eposide?
    self.rgb_image = None  # last RGB image
    self.pygame_surface = None  # last RGB image made pygame surface

    self.camera_type = camera_type

    # initialize
    camera_actor.listen(self.parse_image)

  # callback for sensor.listen()
  # this is called whenever a data comes from CARLA server
  def parse_image(self, image):
    """Process one camera captured image."""
    # parse the image data into a pygame surface for display or screenshot
    # raw image is BGRA
    # if image_type is segmentation, here will convert to the pre-defined color
    image.convert(self.image_type)

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # BGR -> RGB
    self.rgb_image = array
    self.pygame_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    self.last_image_seconds = image.timestamp
    self.last_image_frame_num = image.frame


"""
moving:
  WASD or Arrows
  add shift to increase speed


Starting the game: SPACE

Toggle the 3D bbox to the current player: c

"""
def keyboard_control(args_, pygame_clock, world_, client_, runtime_bucket,
                     moment_data, client):
  """Process all the keyboard event since last tick."""
  # the rgb camera and seg camera are in the args
  # since we may need to change the fov for them
  # by rebuild the camera actor

  # return True to exit
  ms_since_last_tick = pygame_clock.get_time()

  # get all event from event queue
  # empty out the event queue
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      return True
    elif event.type == pygame.KEYUP:
      if event.key == pygame.K_SPACE:
        if not runtime_bucket["is_annotating"]:
          runtime_bucket["is_annotating"] = True
          reset_moment(runtime_bucket, client)
      elif event.key == pygame.K_c:
        if not runtime_bucket["show_bbox"]:
          runtime_bucket["show_bbox"] = True
        else:
          runtime_bucket["show_bbox"] = False

  # last obs speed may not updated yet
  if x_agent_pid in runtime_bucket["moment_vars"]["pid2actor"] and (
      runtime_bucket["last_obs_speed"] is not None) and (
          runtime_bucket["prev_x_agent_rotation"] is not None):

    # get a big dict of what key is pressed now, so to avoid hitting forward
    # multiple times to go forward for a distance
    step = ms_since_last_tick  # this is from experimenting
    keys = pygame.key.get_pressed()
    control = carla.WalkerControl()
    control.speed = 0.0

    # need to use one object to remember the rotation of the actor,
    # rather than call the actor and get rotation at each frame
    # since the rotation of the actor called at each frame is one frame
    # behind the actual rotation
    # then you will get stuttering
    rotation = runtime_bucket["prev_x_agent_rotation"]

    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
      control.speed = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
      control.speed = .01
      rotation.yaw -= 0.08 * step
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
      control.speed = .01
      rotation.yaw += 0.08 * step
    if keys[pygame.K_UP] or keys[pygame.K_w]:
      if pygame.key.get_mods() & pygame.KMOD_SHIFT:
        control.speed = runtime_bucket["last_obs_speed"] * 1.2
      else:
        control.speed = runtime_bucket["last_obs_speed"]

    rotation.yaw = round(rotation.yaw, 1)
    control.direction = rotation.get_forward_vector()
    runtime_bucket["this_x_agent_control"] = control
  else:
    runtime_bucket["this_x_agent_control"] = None

  return False


def select_new_actor(runtime_bucket, get_idx):
  # all the current actor list
  # here we assume the person_id and vehicle id does not conflict
  # since they are track_id
  all_actors = sorted(runtime_bucket["person_data"].keys()) + \
      sorted(runtime_bucket["vehicle_data"].keys())
  if all_actors:
    # there is nothing selected currently
    if (runtime_bucket["selected"] is None) or \
        (runtime_bucket["selected"] not in all_actors):
      runtime_bucket["selected"] = all_actors[0]
      runtime_bucket["is_vehicle"] = len(
          runtime_bucket["person_data"].keys()) == 0
    else:
      cur_selected_idx = all_actors.index(runtime_bucket["selected"])
      cur_selected_idx += get_idx
      if cur_selected_idx >= len(all_actors):
        cur_selected_idx = 0
      elif cur_selected_idx < 0:
        cur_selected_idx = len(all_actors) - 1
      runtime_bucket["selected"] = all_actors[cur_selected_idx]
      runtime_bucket["is_vehicle"] = \
          cur_selected_idx >= len(runtime_bucket["person_data"].keys())

def xyz_to_carla(xyz):
  return carla.Location(x=xyz[0], y=xyz[1], z=xyz[2])


def carla_to_xyz(carla_location):
  return [carla_location.x, carla_location.y, carla_location.z]

def spawn_actor(bps, xyz, world):
  bp = get_bp(bps)
  actor = world.try_spawn_actor(bp, carla.Transform(
      location=carla.Location(x=xyz[0],
                              y=xyz[1],
                              z=xyz[2])))
  return actor


def spawn_static(traj_data, static_dict, bps, frame_idx, world):
  for pid in traj_data:
    actor = spawn_actor(bps, traj_data[pid][frame_idx]["xyz"], world)
    if actor is None:
      print("Person/vehicle %s failed to spawn at frame %s" %
            (pid, frame_idx))
    else:
      static_dict[pid] = actor


def plot_actor_3d_bbox(world, actor, color, fps, thickness=0.1):
  color = carla.Color(r=color[0], g=color[1], b=color[2])
  # get the current transform (location + rotation)
  transform = actor.get_transform()
  # bounding box is relative to the actor
  bounding_box = actor.bounding_box
  bounding_box.location += transform.location  # from relative to world
  world.debug.draw_box(bounding_box, transform.rotation, thickness=thickness,
                       color=color, life_time=1.0/fps)


def get_trajs(moment_data, moment_idx):
  # person_id -> list of points
  person_data, _ = control_data_to_traj(
      moment_data[moment_idx]["ped_controls"])
  vehicle_data, _ = control_data_to_traj(
      moment_data[moment_idx]["vehicle_controls"])
  return person_data, vehicle_data


def init_moment(world, client, moment_data, moment_idx, global_actor_list):
  settings = world.get_settings()
  settings.synchronous_mode = True
  settings.fixed_delta_seconds = 1.0 / \
      moment_data[moment_idx]["static_scene"]["fps"]
  world.apply_settings(settings)
  setup_static(world, client, moment_data[moment_idx]["static_scene"],
               global_actor_list)

  # set weather to rainy for more realism look

  #world.set_weather(carla.WeatherParameters.SoftRainNoon)
  #world.set_weather(carla.WeatherParameters.WetNoon)
  world.set_weather(realism_weather)

  person_data, vehicle_data = get_trajs(moment_data, moment_idx)

  return person_data, vehicle_data


def get_dict_idx_key(dict_, idx):
  keys = sorted([int(float(k)) for k in dict_.keys()])  # json stupid
  return keys[idx]


def reset_moment(runtime_bucket, client):
  runtime_bucket["playing_moment_fidx"] = 0
  # clean up and reset all the variables
  cleanup_actors(
      list(runtime_bucket["moment_vars"]["cur_peds"].values()) + \
      [x.sensor for x in runtime_bucket["moment_vars"][
          "cur_ped_collisions"].values()] + \
      list(runtime_bucket["moment_vars"]["cur_vehicles"].values()),
      client)

  runtime_bucket["moment_vars"] = {
      "cur_peds": {},
      "cur_ped_collisions": {},
      "cur_vehicles": {},
      "cur_vehicle_initial_forward_vector": {},
      "cur_vehicle_prev_yaw": {},
      "actorid2info": {},
      "pid2actor": {},
      "local_actor_list": [],
  }
  reset_bps(runtime_bucket["walker_bps"])
  reset_bps(runtime_bucket["vehicle_bps"])

  if runtime_bucket["player_camera"] is not None:
    # destroy the camera
    runtime_bucket["player_camera"].camera_actor.destroy()
  runtime_bucket["player_camera"] = None
  runtime_bucket["last_obs_speed"] = None
  runtime_bucket["prev_x_agent_rotation"] = None
  runtime_bucket["dist_to_dest"] = 99999.0


def next_traj(runtime_bucket, moment_data, args, at_least_1skip=True):

  skip = 1  # the next one
  if not at_least_1skip:
    skip = 0
  runtime_bucket["cur_task"] += 1
  while (runtime_bucket["cur_task"] % args.job) != (args.curJob - 1):
    runtime_bucket["cur_task"] += 1
    skip += 1

  for i in range(skip):
    if runtime_bucket["cur_moment_idx"] >= len(moment_data):
      break
    this_moment_data = moment_data[runtime_bucket["cur_moment_idx"]]
    # check destination traverse
    x_agent_pid = get_dict_idx_key(this_moment_data["x_agents"],
                                   runtime_bucket["cur_x_agent_idx"])

    next_moment_idx = runtime_bucket["cur_moment_idx"]
    next_cur_x_agent_idx = runtime_bucket["cur_x_agent_idx"]
    next_cur_x_agent_dest_idx = runtime_bucket["cur_x_agent_dest_idx"] + 1

    num_dest = len(this_moment_data["x_agents"][x_agent_pid])
    if next_cur_x_agent_dest_idx >= num_dest:
      next_cur_x_agent_idx += 1
      next_cur_x_agent_dest_idx = 0

    num_x_agents = len(this_moment_data["x_agents"])
    if next_cur_x_agent_idx >= num_x_agents:
      next_moment_idx += 1
      next_cur_x_agent_idx = 0

    runtime_bucket["cur_moment_idx"] = next_moment_idx
    runtime_bucket["cur_x_agent_idx"] = next_cur_x_agent_idx
    runtime_bucket["cur_x_agent_dest_idx"] = next_cur_x_agent_dest_idx


def check_collision_with_actor(collisions):
  has_collision = False
  for _, _, _, _, _, tag in collisions:
    if not tag.startswith("static"):
      has_collision = True
      break
  return has_collision

if __name__ == "__main__":
  args = parser.parse_args()

  # compute the frame_ids for each moment
  args.frame_skip = int(args.video_fps / args.annotation_fps)  # 12/10
  # should be 0 -> 456
  args.moment_frame_ids = range(
      0, (args.obs_length + args.pred_length) * args.frame_skip,
      args.frame_skip)

  args.width, args.height = [int(x) for x in args.res.split("x")]

  # assemble all moment data into one list first
  files = [line.strip()
           for line in open(args.moment_file_list, "r").readlines()]
  filenames = [os.path.splitext(os.path.basename(one))[0] for one in files]

  moment_data = []
  for moment_file in files:
    with open(moment_file, "r") as f:
      data = json.load(f)
    moment_data += data

  # reset all x_agents dict's key to int, since stupid json
  reset_x_agent_key(moment_data)

  # compute the total traj time if take into account the x_agent destinations
  total_traj = 0
  for one in moment_data:
    assert one["x_agents"].keys(), (one["filename"],
                                    one["original_start_frame_id"])
    for x_agent_pid in one["x_agents"]:
      total_traj += len(one["x_agents"][x_agent_pid])

  print("%s moment file, total moment %s, total traj %s" % (
      len(files), len(moment_data), total_traj))

  if not moment_data:
    print("Data is empty.")
    sys.exit()

  # save annotation data
  saved_annotation = {}  # (moment_idx, x_agent_pid, destination_idx) -> traj

  scene = moment_data[args.start_idx]["scenename"]
  camera_location_preset = anchor_cameras_annotation[scene][0]
  args.fov = anchor_cameras_annotation[scene][1]

  map_name = "Town03_ethucy"
  if args.is_actev:
    map_name = "Town05_actev"

  # record the failure rate for the moment that are annotated
  failure_counts = {}  # traj_key -> count
  try:
    # record all the actor so we could clean them
    global_actor_list = []

    # connect to carla server
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.load_world(map_name)

    walker_bps = setup_walker_bps(world)
    vehicle_bps = setup_vehicle_bps(world)

    blueprint_library = world.get_blueprint_library()

    # the world will have a spectator already in it with actor id == 1
    spectator = world.get_spectator()

    pygame.init()

    # pygame screen
    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    # configure the rgb camera # this is the scene camera
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "%s" % args.width)
    camera_bp.set_attribute("image_size_y", "%s" % args.height)
    #camera_bp.set_attribute("fov", "%s" % args.fov)
    # always has a big fov so user can see the destinations
    camera_bp.set_attribute("fov", "%s" % 90)
    # Set the time in seconds between sensor captures
    camera_bp.set_attribute("sensor_tick", "%s" % 0.0)
    camera_bp.set_attribute("enable_postprocess_effects", "true")
    # no motion blur
    camera_bp.set_attribute("motion_blur_intensity", "%s" % 0.0)
    # 2.2 is default, 1.5 is the carla default spectator gamma (darker)
    camera_bp.set_attribute("gamma", "%s" % 1.6)

    # player camera
    camera_bp_player = blueprint_library.find("sensor.camera.rgb")
    camera_bp_player.set_attribute("image_size_x", "%s" % args.width)
    camera_bp_player.set_attribute("image_size_y", "%s" % args.height)
    camera_bp_player.set_attribute("fov", "%s" % 110)
    # Set the time in seconds between sensor captures
    camera_bp_player.set_attribute("sensor_tick", "%s" % 0.0)
    camera_bp_player.set_attribute("enable_postprocess_effects", "true")
    # no motion blur
    camera_bp_player.set_attribute("motion_blur_intensity", "%s" % 0.0)
    # 2.2 is default, 1.5 is the carla default spectator gamma (darker)
    camera_bp_player.set_attribute("gamma", "%s" % 1.6)
    # we will spawn and attach this at runtime
    args.camera_bp_player = camera_bp_player


    # the rgb camera actor and the seg camera actor
    spawn_commands = [
        carla.command.SpawnActor(camera_bp, carla.Transform(), spectator),
    ]
    response = client.apply_batch_sync(spawn_commands)
    camera_actor_ids = [x.actor_id for x in response]
    scene_camera, = world.get_actors(camera_actor_ids)

    scene_camera = Camera(scene_camera, camera_type="rgb")

    args.spectator = spectator
    args.scene_camera = scene_camera

    spectator.set_transform(camera_location_preset)

    client_clock = pygame.time.Clock()

    # some variables
    runtime_bucket = {
        "cur_moment_idx": args.start_idx,
        "cur_x_agent_idx": 0,
        "cur_x_agent_dest_idx": 0,
        "cur_task": 0,
        "this_annotation": [],  # frame_id , control, location_xyz

        "playing_moment_fidx": 0,
        "vehicle_controls": None,
        "moment_vars": {
            "cur_peds": {},
            "cur_ped_collisions": {},
            "cur_vehicles": {},
            "cur_vehicle_initial_forward_vector": {},
            "cur_vehicle_prev_yaw": {},
            "actorid2info": {},
            "pid2actor": {},
            "local_actor_list": [],
        },
        "is_annotating": False,
        "this_x_agent_control": None,
        "dist_to_dest": 99999.0,
        "last_obs_speed": None,
        "prev_x_agent_rotation": None,
        "player_camera": None,

        "show_bbox": True,  # whether show 3d bbox of the player during anno

        "walker_bps": walker_bps,
        "vehicle_bps": vehicle_bps,
    }
    dist_to_reach = 2.0  # within this to consider reach destination

    next_traj(runtime_bucket, moment_data, args, at_least_1skip=False)

    person_data, vehicle_data = init_moment(
        world, client, moment_data, runtime_bucket["cur_moment_idx"],
        global_actor_list)

    # this tick applies all the static stuff
    static_frame_id = world.tick()

    # count each moment annotation attempts
    this_fail_count = 0

    # one frame loop
    # moment_data and runtime_bucket will be changed based on keyboard input
    while True:
      # keep the simulation run under this fps
      client_clock.tick_busy_loop(args.video_fps)
      # some shortening vars
      ped_controls = moment_data[runtime_bucket["cur_moment_idx"]][
          "ped_controls"]

      # interpolate the 2.5 fps vehicle_controls to 30 to make it look better
      if runtime_bucket["vehicle_controls"] is None:
        runtime_bucket["vehicle_controls"] = interpolate_controls(
            moment_data[runtime_bucket["cur_moment_idx"]]["vehicle_controls"],
            args.video_fps)

      this_moment_data = moment_data[runtime_bucket["cur_moment_idx"]]
      # int;
      x_agent_pid = get_dict_idx_key(this_moment_data["x_agents"],
                                     runtime_bucket["cur_x_agent_idx"])
      # stupid json
      dest_xyz = this_moment_data["x_agents"][x_agent_pid][
          runtime_bucket["cur_x_agent_dest_idx"]]

      # keyboard and mouse control
      if keyboard_control(
          args, client_clock, world, client, runtime_bucket, moment_data,
          client):
        break

      cur_moment_idx = runtime_bucket["cur_moment_idx"]


      if runtime_bucket["is_annotating"]:
        # check whether x_agent has reached destination
        if x_agent_pid in runtime_bucket["moment_vars"]["pid2actor"]:
          x_agent_actor = runtime_bucket["moment_vars"]["pid2actor"][
              x_agent_pid]
          dest_location = carla.Location(x=dest_xyz[0], y=dest_xyz[1],
                                         z=dest_xyz[2])
          dist = dest_location.distance(x_agent_actor.get_location())
          runtime_bucket["dist_to_dest"] = dist
          if dist <= dist_to_reach:
            # save this success control
            this_scene = moment_data[
                runtime_bucket["cur_moment_idx"]]["scenename"]
            traj_key = "%s_%d_%d_%d" % (
                this_scene, runtime_bucket["cur_moment_idx"], x_agent_pid,
                runtime_bucket["cur_x_agent_dest_idx"])
            assert traj_key not in saved_annotation
            saved_annotation[traj_key] = runtime_bucket["this_annotation"][:]
            runtime_bucket["this_annotation"] = []

            # record the fail counts
            failure_counts[traj_key] = this_fail_count
            this_fail_count = 0

            # reset everything and move on
            runtime_bucket["player_camera"].camera_actor.destroy()
            runtime_bucket["player_camera"] = None

            reset_moment(runtime_bucket, client)
            runtime_bucket["is_annotating"] = False

            # set the right next moment idx, x_agent_idx, etc.
            prev_moment_idx = runtime_bucket["cur_moment_idx"]

            next_traj(runtime_bucket, moment_data, args)

            if prev_moment_idx != runtime_bucket["cur_moment_idx"]:
              # need to redo the interpolation for next vehicle control
              # if the moment changed
              runtime_bucket["vehicle_controls"] = None

            if runtime_bucket["cur_moment_idx"] >= len(moment_data):
              print("Everything done.")
              break
            continue  # directly jump out and start the next tick

        # annotation failure
        has_collision = False
        if x_agent_pid in runtime_bucket["moment_vars"]["cur_ped_collisions"]:
          if check_collision_with_actor(
              runtime_bucket["moment_vars"]["cur_ped_collisions"][
                  x_agent_pid].history):
            has_collision = True
            print("%s: X agent collision: %s" % (
                runtime_bucket["cur_task"],
                runtime_bucket["moment_vars"]["cur_ped_collisions"][
                    x_agent_pid].history))
        if (runtime_bucket["playing_moment_fidx"] > \
              args.moment_frame_ids[-1]) or has_collision:
          print("%s: Annotation failed. Reset and try again" % (
              runtime_bucket["cur_task"]))
          runtime_bucket["this_annotation"] = []
          reset_moment(runtime_bucket, client)
          this_fail_count += 1

      if runtime_bucket["playing_moment_fidx"] <= \
          args.moment_frame_ids[args.obs_length-1]:
        # grab the control data of this frame if any
        batch_cmds, sim_stats = run_sim_for_one_frame(
            "%s" % runtime_bucket["playing_moment_fidx"],
            ped_controls, runtime_bucket["vehicle_controls"],
            runtime_bucket["moment_vars"]["cur_peds"],
            runtime_bucket["moment_vars"]["cur_ped_collisions"],
            runtime_bucket["moment_vars"]["cur_vehicles"],
            runtime_bucket["moment_vars"][
                "cur_vehicle_initial_forward_vector"],
            runtime_bucket["moment_vars"][
                "cur_vehicle_prev_yaw"],
            runtime_bucket["walker_bps"],
            runtime_bucket["vehicle_bps"],
            world,
            runtime_bucket["moment_vars"]["local_actor_list"],
            runtime_bucket["moment_vars"]["actorid2info"],
            show_traj=False, verbose=False,
            max_yaw_change=90, exit_if_spawn_fail=False,
            pid2actor=runtime_bucket["moment_vars"]["pid2actor"])

        if runtime_bucket["is_annotating"]:
          # try to spawn the camera on the x_agent
          if runtime_bucket["player_camera"] is None:
            if x_agent_pid in runtime_bucket["moment_vars"]["pid2actor"]:
              x_agent_actor = runtime_bucket["moment_vars"]["pid2actor"][
                  x_agent_pid]
              # over the shoulder
              player_camera = world.spawn_actor(
                  args.camera_bp_player,
                  carla.Transform(
                      carla.Location(x=-5.5, y=0, z=1.5),
                      carla.Rotation(pitch=10.0)),
                  attach_to=x_agent_actor,
                  attachment_type=carla.AttachmentType.SpringArm)

              runtime_bucket["player_camera"] = Camera(
                  player_camera, camera_type="rgb")

          # get the last obs speed for later
          if runtime_bucket["last_obs_speed"] is None:
            person_data, _ = control_data_to_traj(
                moment_data[runtime_bucket["cur_moment_idx"]]["ped_controls"])
            runtime_bucket["last_obs_speed"] = \
                person_data[x_agent_pid][args.obs_length-1]["speed"]

        if batch_cmds:
          response = client.apply_batch_sync(batch_cmds)

        if not runtime_bucket["is_annotating"] or runtime_bucket["show_bbox"]:
          # plot the current x agent box
          plot_actor_3d_bbox(
              world, runtime_bucket["moment_vars"]["pid2actor"][x_agent_pid],
              (0, 0, 255), args.video_fps, thickness=0.05)

        # keep the spetactor camera in the right place
        this_scene = moment_data[
            runtime_bucket["cur_moment_idx"]]["scenename"]

        camera_location_preset = anchor_cameras_annotation[this_scene][0]
        args.spectator.set_transform(camera_location_preset)

        runtime_bucket["playing_moment_fidx"] += 1
      else:
        # during prediction time

        if not runtime_bucket["is_annotating"]:
          # simluate from "playing_moment_fidx" 0 to
          # the last observation frame
          # this is to stop and reset to show the observation
          reset_moment(runtime_bucket, client)
        else:
          # during annotation and the annotator needs to control
          if runtime_bucket["prev_x_agent_rotation"] is None:
            x_agent_actor = runtime_bucket["moment_vars"]["pid2actor"][
                x_agent_pid]
            # set the initial rotation of the x_agent
            runtime_bucket["prev_x_agent_rotation"] = \
                x_agent_actor.get_transform().rotation
          # simulate everything else without the x_agent
          # grab the control data of this frame if any
          batch_cmds, sim_stats = run_sim_for_one_frame(
              "%s" % runtime_bucket["playing_moment_fidx"],
              ped_controls, runtime_bucket["vehicle_controls"],
              runtime_bucket["moment_vars"]["cur_peds"],
              runtime_bucket["moment_vars"]["cur_ped_collisions"],
              runtime_bucket["moment_vars"]["cur_vehicles"],
              runtime_bucket["moment_vars"][
                  "cur_vehicle_initial_forward_vector"],
              runtime_bucket["moment_vars"][
                  "cur_vehicle_prev_yaw"],
              runtime_bucket["walker_bps"],
              runtime_bucket["vehicle_bps"],
              world,
              runtime_bucket["moment_vars"]["local_actor_list"],
              runtime_bucket["moment_vars"]["actorid2info"],
              show_traj=False, verbose=True,
              max_yaw_change=90, exit_if_spawn_fail=False,
              pid2actor=runtime_bucket["moment_vars"]["pid2actor"],
              excepts=[x_agent_pid])
          # x_agent will apply the keyboard control
          if runtime_bucket["this_x_agent_control"] is not None:

            #x_agent_actor = runtime_bucket["moment_vars"]["pid2actor"][
            #    x_agent_pid]
            #x_agent_actor.apply_control(
            #    runtime_bucket["this_x_agent_control"])
            batch_cmds.append(carla.command.ApplyWalkerControl(
                runtime_bucket["moment_vars"]["pid2actor"][x_agent_pid],
                runtime_bucket["this_x_agent_control"]))

            # save this control
            runtime_bucket["this_annotation"].append([
                runtime_bucket["playing_moment_fidx"],
                carla_to_xyz(
                    runtime_bucket["this_x_agent_control"].direction),
                runtime_bucket["this_x_agent_control"].speed,
                carla_to_xyz(
                    runtime_bucket["moment_vars"]["pid2actor"][
                        x_agent_pid].get_location())])
            # reset
            runtime_bucket["this_x_agent_control"] = None

          if batch_cmds:
            response = client.apply_batch_sync(batch_cmds)
          runtime_bucket["playing_moment_fidx"] += 1

          if runtime_bucket["show_bbox"]:
            # plot the current x agent box
            plot_actor_3d_bbox(
                world, runtime_bucket["moment_vars"]["pid2actor"][x_agent_pid],
                (0, 255, 255), args.video_fps, thickness=0.015)

      # always plot the destination
      world.debug.draw_point(
          xyz_to_carla(dest_xyz), color=carla.Color(r=255, g=0, b=0),
          size=0.7, life_time=1.0/args.video_fps)

      # update camera image at the pygame screen
      # could be the scene camera or the player camera
      if runtime_bucket["is_annotating"]:
        if runtime_bucket["player_camera"] is not None:
          if runtime_bucket["player_camera"].pygame_surface is not None:
            display.blit(runtime_bucket["player_camera"].pygame_surface, (0, 0))
      else:
        if args.scene_camera.pygame_surface is not None:
          display.blit(args.scene_camera.pygame_surface, (0, 0))

      #  ------ show the current moment stats in text on the screen
      show_stats = {
          "moment": "%s/%s" % (cur_moment_idx + 1,
                               len(moment_data)),
          "task": "%s/%s" % (runtime_bucket["cur_task"], total_traj),
          "time": "%.2f" % (
              runtime_bucket["playing_moment_fidx"] / args.video_fps)
      }

      show_stats = ", ".join(
          #["%s: %s" % (k, v) for k, v in show_stats.iteritems()])
          ["%s: %s" % (k, v) for k, v in show_stats.items()])
      text_surface, text_offset = make_text_surface(show_stats, 0)
      display.blit(text_surface, (0, 0))

      show_stats = {}
      if not runtime_bucket["is_annotating"]:
        show_stats["Instruction"] = \
            "Move person in blue box to red dot location after 4.8 seconds " \
            " (within 15.2 seconds)."
        show_stats["Control"] = \
            "Press space to start."
      else:
        show_stats["Dist to destination"] = "%.1f" % \
            runtime_bucket["dist_to_dest"]
        show_stats["Control"] = \
            "Press and hold UP/W to walk. Hold SHIFT to walk faster. " \
            " LEFT/RIGHT or w/s for direction."
            #"c to show/hide the player bbox."
      show_stats = ", ".join(
          #["%s: %s" % (k, v) for k, v in show_stats.iteritems()])
          ["%s: %s" % (k, v) for k, v in show_stats.items()])
      text_surface, _ = make_text_surface(show_stats, text_offset)
      display.blit(text_surface, (0, text_offset))

      pygame.display.flip()
      server_frame_id = world.tick()

  finally:
    # save all annotation
    with open(args.annotation_file, "w") as f:
      json.dump(saved_annotation, f)

    # save the log
    log = {
        "failure_counts": failure_counts,
    }
    with open(args.log_file, "w") as f:
      json.dump(log, f)
    failure_sum = sum([failure_counts[k] for k in failure_counts])
    print("total failure %s for annotating %s tasks" % (
        failure_sum, len(failure_counts)))
    # finished, clean actors
    # destroy the camera actor separately
    if args.scene_camera.camera_actor is not None:
      global_actor_list.append(args.scene_camera.camera_actor)
    if runtime_bucket["player_camera"] is not None:
      global_actor_list.append(runtime_bucket["player_camera"].camera_actor)

    global_actor_list += runtime_bucket["moment_vars"]["local_actor_list"]
    cleanup_actors(global_actor_list, client)

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    pygame.quit()
