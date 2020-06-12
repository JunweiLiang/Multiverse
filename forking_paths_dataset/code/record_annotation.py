# coding=utf-8
"""After starting a carla server off-sceen, use this to spawn camera to record stuff."""

from __future__ import print_function

import argparse
import cv2
import datetime
import json
import math
import sys
import os
import glob
from tqdm import tqdm
import numpy as np

import sys
if sys.version_info > (3, 0):
  import subprocess as commands
else:
  import commands

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla
import time

from utils import setup_walker_bps
from utils import setup_vehicle_bps
from utils import anchor_cameras
from utils import get_3d_bbox
from utils import get_2d_bbox
from utils import run_sim_for_one_frame
from utils import cleanup_actors
from utils import reset_bps
from utils import recording_cameras
from utils import static_scenes
from utils import compute_intrinsic
from utils import realism_weather

parser = argparse.ArgumentParser()

parser.add_argument("moment_json", help="control data of the moment")
parser.add_argument("outpath",
                    help="path to save the videos of rgb, segmentation,"
                         " and bounding box json, and carla xyz.")

# interface setup
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--res", default="1920x1080")

parser.add_argument("--only", default=None, help="only record scene=only")

parser.add_argument("--is_actev", action="store_true")
parser.add_argument("--start_offset", type=int, default=10)

parser.add_argument("--is_anchor_moment", action="store_true",
                    help="record the auto-gen anchor moment")

parser.add_argument("--add_3view_to_anchor", action="store_true",
                    help="add views except anchor view to the anchor videos")
parser.add_argument("--add_dashboard_view_to_anchor", action="store_true")
parser.add_argument("--no_ori_view", action="store_true")
parser.add_argument("--cam_num_offset", type=int, default=0)

parser.add_argument("--use_alter_weather", action="store_true")

# moment setup
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--annotation_fps", type=float, default=2.5)
parser.add_argument("--obs_length", type=int, default=12,
                    help="observation timestep, 12/2.5=4.8 seconds")
parser.add_argument("--pred_length", type=int, default=26,
                    help="observation timestep, 26/2.5=10.4 seconds")


class Camera(object):
  """Camera object to have a surface."""
  def __init__(self, camera_actor, save_path, camera_type="rgb",
               image_type=carla.ColorConverter.Raw,
               base_frame_id=0,
               start_from_frame=0,
               width=None, height=None, fov=None):
    self.camera_actor = camera_actor
    self.image_type = image_type
    self.base_frame_id = base_frame_id
    self.start_from_frame = start_from_frame
    self.save_path = save_path
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    self.camera_type = camera_type

    # initialize
    camera_actor.listen(self.parse_image)
    if self.camera_type == "rgb":
      self.camera_actor.intrinsic = compute_intrinsic(width, height, fov)

  # callback for sensor.listen()
  # this is called whenever a data comes from CARLA server
  def parse_image(self, image):
    """Process one camera captured image."""
    # raw image is BGRA
    # if image_type is segmentation, here will convert to the pre-defined color
    frame_id = image.frame - self.base_frame_id
    if frame_id >= self.start_from_frame:
      frame_id = frame_id - self.start_from_frame
      image.convert(self.image_type)
      image.save_to_disk(
          os.path.join(self.save_path, "%08d.jpg" % frame_id))

def setup_camera(args, transform, fov, rgb_save_path, seg_save_path,
                 base_frame_id):
  # configure the rgb camera
  camera_rgb_bp = blueprint_library.find("sensor.camera.rgb")
  camera_rgb_bp.set_attribute("image_size_x", "%s" % args.width)
  camera_rgb_bp.set_attribute("image_size_y", "%s" % args.height)
  camera_rgb_bp.set_attribute("fov", "%s" % fov)
  # Set the time in seconds between sensor captures
  camera_rgb_bp.set_attribute("sensor_tick", "%s" % 0.0)
  camera_rgb_bp.set_attribute("enable_postprocess_effects", "true")
  # no motion blur
  camera_rgb_bp.set_attribute("motion_blur_intensity", "%s" % 0.0)
  # 2.2 is default, 1.5 is the carla default spectator gamma (darker)
  camera_rgb_bp.set_attribute("gamma", "%s" % 1.6)

  # seg camera
  camera_seg_bp = blueprint_library.find(
      "sensor.camera.semantic_segmentation")
  camera_seg_bp.set_attribute("image_size_x", "%s" % args.width)
  camera_seg_bp.set_attribute("image_size_y", "%s" % args.height)
  camera_seg_bp.set_attribute("fov", "%s" % fov)
  camera_seg_bp.set_attribute("sensor_tick", "%s" % 0.0)

  camera_rgb_actor = world.spawn_actor(camera_rgb_bp, transform)
  camera_seg_actor = world.spawn_actor(camera_seg_bp, transform)

  rgb_camera = Camera(camera_rgb_actor, rgb_save_path,
                      width=args.width, height=args.height,
                      base_frame_id=base_frame_id,
                      start_from_frame=args.start_offset,
                      fov=fov, camera_type="rgb")

  seg_camera = Camera(camera_seg_actor, seg_save_path,
                      width=args.width, height=args.height,
                      fov=fov, camera_type="seg",
                      base_frame_id=base_frame_id,
                      start_from_frame=args.start_offset,
                      image_type=carla.ColorConverter.CityScapesPalette)
  return rgb_camera, seg_camera

if __name__ == "__main__":
  args = parser.parse_args()

  # compute the frame_ids for each moment
  args.frame_skip = int(args.video_fps / args.annotation_fps)  # 12/10
  args.moment_frame_ids = range(
      0, (args.obs_length + args.pred_length) * args.frame_skip,
      args.frame_skip)

  args.width, args.height = [int(x) for x in args.res.split("x")]

  # all the folder generated
  args.rgb_path = os.path.join(args.outpath, "rgb_videos")
  # save scene segmentation as video
  args.seg_path = os.path.join(args.outpath, "seg_videos")
  args.bbox_path = os.path.join(args.outpath, "bbox")
  # temporay folder to save the video frames for each camera
  args.temp_path = os.path.join(args.outpath, "temp%s" % args.port)
  for path in [args.rgb_path, args.seg_path, args.bbox_path, args.temp_path]:
    if not os.path.exists(path):
      os.makedirs(path)

  if args.is_anchor_moment:
    # except the first camera view
    additional_cameras = {k: recording_cameras[k][1:4]
                          for k in recording_cameras}
    # dashboard view
    dashboard_camera = {k: recording_cameras[k][4]
                        for k in recording_cameras
                        if len(recording_cameras[k]) >= 5}

    # reset the recording cameras
    recording_cameras = {}
    for k in anchor_cameras:
      recording_cameras[k] = [anchor_cameras[k]]
      if args.no_ori_view:
        recording_cameras[k] = []
      if args.add_3view_to_anchor:
        recording_cameras[k] += additional_cameras[k]
      if args.add_dashboard_view_to_anchor:
        if k in dashboard_camera:
          recording_cameras[k].append(dashboard_camera[k])

  with open(args.moment_json, "r") as f:
    moment_data = json.load(f)
  if not moment_data:
    print("Data is empty.")
    sys.exit()

  map_name = "Town03_ethucy"
  if args.is_actev:
    map_name = "Town05_actev"

  # some initialization just in case
  cur_peds, cur_vehicles, local_actor_list, camera_actors = {}, {}, [], []
  try:
    # record all the actor so we could clean them
    global_actor_list = []

    # connect to carla server
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.load_world(map_name)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.video_fps
    world.apply_settings(settings)

    walker_bps = setup_walker_bps(world)
    vehicle_bps = setup_vehicle_bps(world)

    blueprint_library = world.get_blueprint_library()

    for this_moment in tqdm(moment_data):

      scene = this_moment["scenename"]
      if args.only is not None:
        if scene != args.only:
          continue
      if args.is_anchor_moment:
        moment_id = "%s_F_%s_obs%d_pred%d" % (
            this_moment["filename"], this_moment["original_start_frame_id"],
            args.obs_length, args.pred_length)
        x_agent_pid = None
      else:
        moment_id = this_moment["moment_id"]
        _, _, x_agent_pid, _, _ = moment_id.split("_")
        x_agent_pid = int(x_agent_pid)
      # camera_transform, camera_fov for each
      camera_params = recording_cameras[scene]

      # 1. set weather
      weather = carla.WeatherParameters(
          cloudyness=static_scenes[scene]["weather"]["cloudyness"],
          precipitation=static_scenes[scene]["weather"]["precipitation"],
          sun_altitude_angle=static_scenes[scene]["weather"][
              "sun_altitude_angle"],
          precipitation_deposits=static_scenes[scene]["weather"][
              "precipitation_deposits"],
          wind_intensity=static_scenes[scene]["weather"]["wind_intensity"],
          sun_azimuth_angle=static_scenes[scene]["weather"][
              "sun_azimuth_angle"])
      if args.use_alter_weather:
        weather = realism_weather
      world.set_weather(weather)

      base_frame_id = world.tick()

      # 2. setup cameras
      camera_actors = []
      this_cameras = []  # each camera has a rgb and a seg Camera obj
      for i, (camera_transform, camera_fov) in enumerate(camera_params):
        save_rgb_frame_path = os.path.join(args.temp_path, "cam%d_rgb" % (i+1))
        save_seg_frame_path = os.path.join(args.temp_path, "cam%d_seg" % (i+1))
        rgb_camera, seg_camera = setup_camera(
            args, camera_transform, camera_fov,
            rgb_save_path=save_rgb_frame_path,
            seg_save_path=save_seg_frame_path,
            base_frame_id=base_frame_id)
        this_cameras.append((rgb_camera, seg_camera))
        camera_actors += [rgb_camera.camera_actor, seg_camera.camera_actor]

      # 3. run the simulation
      cur_peds, cur_vehicles, v_vector, v_yaw = {}, {}, {}, {}
      local_actor_list = []
      actorid2info = {}
      bboxes = {}  # camera_idx -> []
      all_frame_ids = sorted([int(fidx)
                              for fidx in this_moment["ped_controls"]])
      if args.is_anchor_moment:
        all_frame_ids = args.moment_frame_ids
      # start at later frame
      for sim_frame_idx in range(0, all_frame_ids[-1]):

        # grab the control data of this frame if any
        batch_cmds, sim_stats = run_sim_for_one_frame(
            "%s" % sim_frame_idx,  # json's fault
            this_moment["ped_controls"], this_moment["vehicle_controls"],
            cur_peds,
            {},
            cur_vehicles,
            v_vector,
            v_yaw,
            walker_bps, vehicle_bps,
            world,
            local_actor_list,
            actorid2info,
            show_traj=False, verbose=False,
            no_collision_detector=True,
            max_yaw_change=90, exit_if_spawn_fail=False)
        if batch_cmds:
          response = client.apply_batch_sync(batch_cmds)

        if sim_frame_idx < args.start_offset:
          # ignore the first few frames
          server_frame_id = world.tick()
          continue
        # compute the bounding boxes for each camera
        for i, (rgb_camera, _) in enumerate(this_cameras):

          # each camera each frame should see the x_agent
          has_x_agent = False

          # all actor in the scene
          for actor in list(cur_peds.values()) + list(cur_vehicles.values()):
            class_name, track_id = actorid2info[actor.id]

            # [8, 3], last dim is depth,
            bbox_3d = get_3d_bbox(actor, rgb_camera.camera_actor)

            bbox = get_2d_bbox(bbox_3d, args.width, args.height)

            if bbox is not None:
              this_bbox_data = {
                  "bbox": bbox,
                  "class_name": class_name,
                  "track_id": track_id,
                  "is_x_agent": int(track_id == x_agent_pid),
                  "frame_id": sim_frame_idx - args.start_offset,
              }
              if track_id == x_agent_pid:
                has_x_agent = True
              if i not in bboxes:
                bboxes[i] = []
              bboxes[i].append(this_bbox_data)
          # all good
          #if not has_x_agent:
          #  tqdm.write(
          #      "Warning, moment %s, camera %s frame %s has no x_agent." % (
          #          moment_id, i, sim_frame_idx))

        server_frame_id = world.tick()

      # done, cleaning up
      reset_bps(walker_bps)
      reset_bps(vehicle_bps)
      cleanup_actors(list(cur_peds.values()) + list(cur_vehicles.values()) + \
          local_actor_list + camera_actors, client)

      # make the rgb, seg frames into videos
      for i, (rgb_camera, seg_camera) in enumerate(this_cameras):
        rgb_frame_path = os.path.join(rgb_camera.save_path, "%08d.jpg")
        seg_frame_path = os.path.join(seg_camera.save_path, "%08d.jpg")

        rgb_video = os.path.join(args.rgb_path, "%s_cam%d.mp4" % (
            moment_id, i+1+args.cam_num_offset))
        seg_video = os.path.join(args.seg_path, "%s_cam%d.mp4" % (
            moment_id, i+1+args.cam_num_offset))

        if os.path.exists(rgb_video):
          tqdm.write("Warning, skipping existing %s..." % rgb_video)
          continue
        output = commands.getoutput("ffmpeg -y -framerate %.1f -i '%s' '%s'" % (
            args.video_fps, rgb_frame_path, rgb_video))
        #print(output)
        output = commands.getoutput("ffmpeg -y -framerate %.1f -i '%s' '%s'" % (
            args.video_fps, seg_frame_path, seg_video))

        # remove the frame path
        output = commands.getoutput("rm -rf '%s'" % rgb_camera.save_path)
        output = commands.getoutput("rm -rf '%s'" % seg_camera.save_path)

        # save the bboxs for each camera
        bbox_file = os.path.join(args.bbox_path, "%s_cam%d.json" % (
            moment_id, i+1+args.cam_num_offset))
        with open(bbox_file, "w") as f:
          json.dump(bboxes[i], f)

  finally:
    cleanup_actors(list(cur_peds.values()) + list(cur_vehicles.values()) + \
          local_actor_list + camera_actors, client)
    client.apply_batch(
        [carla.command.DestroyActor(x) for x in global_actor_list])
    output = commands.getoutput("rm -rf '%s'" % args.temp_path)

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
