# coding=utf-8
"""Given the carla trajectory files, test the trajectory in the simulator to
find moments without spawn fail and collision in the duration of the moment.
"""

import argparse
import glob
import math
import pygame
import json
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla

import numpy as np
from tqdm import tqdm

from utils import setup_walker_bps
from utils import setup_vehicle_bps
from utils import setup_static
from utils import static_scenes
from utils import get_scene
from utils import get_controls
from utils import vehicle_z
from utils import run_sim_for_one_frame
from utils import cleanup_actors

parser = argparse.ArgumentParser()
parser.add_argument("traj_path")
parser.add_argument("--vehicle_traj_path")

parser.add_argument("--is_actev", action="store_true")
parser.add_argument("--only_scene", default=None)

parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=2000, type=int)

# moments param
parser.add_argument("--moment_length", default=15.2, type=float,
                    help="length of the moment in seconds")
parser.add_argument("--test_skip", default=1, type=int,
                    help="shift how many control point to continue. The "
                         "control points are 2.5 fps so skip 10 means skipping"
                         " 4 seconds.")

# outputs
parser.add_argument("moment_path", help="save the candidates into json files")
parser.add_argument("--log_file", default=None,
                    help="save the collision, spawn fail info")


if __name__ == "__main__":
  args = parser.parse_args()

  # record the data
  fails = []  # (filename, start_frame, fail_at_frame, frame_reason)
  # scene -> list of{filename, ped_controls, vehicle_controls}
  success_moments = {}

  traj_files = glob.glob(os.path.join(args.traj_path, "*.txt"))

  traj_files.sort()

  if not os.path.exists(args.moment_path):
    os.makedirs(args.moment_path)

  try:
    global_actor_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()

    walker_bps = setup_walker_bps(world)
    vehicle_bps = setup_vehicle_bps(world)

    # set the world to sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 30.0
    world.apply_settings(settings)

    client_clock = pygame.time.Clock()

    # the main loop, go over each trajectory files
    for traj_file in tqdm(traj_files):
      filename = os.path.splitext(os.path.basename(traj_file))[0]
      this_moment_count, this_success_count = 0, 0

      if args.is_actev:
        scene = get_scene(filename)
        if args.only_scene is not None:
          if scene != args.only_scene:
            continue
      else:
        scene = filename
      static_scene = static_scenes[scene]

      # changing the map everytime to reset
      world = client.load_world(static_scene["map"])
      # set world fps
      settings = world.get_settings()
      settings.synchronous_mode = True
      settings.fixed_delta_seconds = 1.0 / static_scene["fps"]
      world.apply_settings(settings)

      # set up the static env
      setup_static(world, client, static_scene, global_actor_list)
      # this tick applies all the static stuff
      static_frame_id = world.tick()

      # load the trajectories
      ped_controls, _ = get_controls(
          traj_file, -1, -1, static_scene["fps"], no_offset=True)

      vehicle_controls = {}
      if args.vehicle_traj_path is not None:
        vehicle_traj_file = os.path.join(
            args.vehicle_traj_path, "%s.txt" % filename)
        # no vehicle interpolation to save space
        vehicle_controls, _ = get_controls(
            vehicle_traj_file, -1, -1, static_scene["fps"],
            interpolate=False, z_to=vehicle_z[scene], no_offset=True)

      all_frame_ids = list(ped_controls.keys())
      all_frame_ids.sort()  # always start from zero

      moment_frame_length = args.moment_length * static_scene["fps"]

      # traverse through all moment
      for i in range(0, len(all_frame_ids), args.test_skip):
        this_moment_count += 1
        # this moment start frame
        # until the length of the moment
        start_frame_id = all_frame_ids[i]
        end_idx = -1
        for j in range(i+1, len(all_frame_ids)):
          end_frame_id = all_frame_ids[j]
          if end_frame_id >= start_frame_id + moment_frame_length:
            end_idx = j
            break
        # some stats for this moment
        # how many frames to simulate in this session
        total_moment_frame_num = int(all_frame_ids[end_idx] - start_frame_id)

        # reset all the running memory of some variables
        cur_peds = {}  # person_id -> actor
        cur_ped_collisions = {}  # person_id -> CollisionSensor
        cur_vehicles = {}
        cur_vehicle_initial_forward_vector = {}
        cur_vehicle_prev_yaw = {}
        actorid2info = {}
        max_yaw_change = 90  # no sudden yaw change
        local_actor_list = []
        walker_bps[1] = 0
        vehicle_bps[1] = 0

        # this is the server side frame Id we start within each session
        # this tick applies all the static stuff
        server_frame_id = world.tick()

        success_moment = True
        for moment_frame_count in range(total_moment_frame_num):
          # check for collision first
          has_collision = False
          for person_id in cur_ped_collisions:
            if cur_ped_collisions[person_id].history:
              has_collision = True
              break
          if has_collision:
            fails.append((filename,
                          start_frame_id,
                          moment_frame_count + start_frame_id,
                          "Ped collision detected."))
            success_moment = False
            break

          # grab the control data of this frame if any
          batch_cmds, sim_stats = run_sim_for_one_frame(
              moment_frame_count + start_frame_id,  # start from start_frame_id
              ped_controls, vehicle_controls,
              cur_peds, cur_ped_collisions,
              cur_vehicles, cur_vehicle_initial_forward_vector,
              cur_vehicle_prev_yaw,
              walker_bps, vehicle_bps,
              world, local_actor_list, actorid2info,
              show_traj=False, verbose=False,
              max_yaw_change=max_yaw_change, exit_if_spawn_fail=True)
          # spawning fails
          if batch_cmds is None:
            fails.append((filename,
                          start_frame_id,
                          moment_frame_count + start_frame_id,
                          "Ped spawn fails."))
            success_moment = False
            break

          if batch_cmds:
            response = client.apply_batch_sync(batch_cmds)

          # let the sim run as fast as possible
          #client_clock.tick_busy_loop(static_scene["fps"])
          server_frame_id = world.tick()

        # no spawn fail, no collision during this moment
        # save the controls
        if success_moment:
          # the frame_id should be offset to start from 0
          save_ped_controls = {}
          save_veh_controls = {}
          for frame_id in range(total_moment_frame_num):
            ori_frame_id = frame_id + start_frame_id
            #if ped_controls.has_key(ori_frame_id):
            if ori_frame_id in ped_controls:
              save_ped_controls[frame_id] = ped_controls[ori_frame_id]
            #if vehicle_controls.has_key(ori_frame_id):
            if ori_frame_id in vehicle_controls:
              save_veh_controls[frame_id] = vehicle_controls[ori_frame_id]

          # Duh.
          if not save_ped_controls and not save_veh_controls:
            fails.append((filename,
                          start_frame_id,
                          moment_frame_count + start_frame_id,
                          "Both ped and veh control empty."))
            success_moment = False
          else:
            success_data = {
                "filename": filename,
                "scenename": scene,
                "static_scene": static_scene,
                "original_start_frame_id": start_frame_id,
                "vehicle_spawn_failed": sim_stats["vehicle_spawn_failed"],
                "ped_controls": save_ped_controls,
                "vehicle_controls": save_veh_controls,
                # this is reserved for getting the annotation data
                # person_id -> a list of destinations
                "x_agents": {}}

            #if not success_moments.has_key(scene):
            if scene not in success_moments:
              success_moments[scene] = []
            success_moments[scene].append(success_data)

            this_success_count += 1

        # delete all the actors in the scene
        cleanup_actors(
            list(cur_peds.values()) + \
            [x.sensor for x in cur_ped_collisions.values()] + \
            list(cur_vehicles.values()),
            client)

        tqdm.write("%s, at frame %s, current moment %s, success %s (%.2f), "
                   "total success %s" %
                   (filename, start_frame_id, this_moment_count,
                    this_success_count,
                    float(this_success_count)/this_moment_count,
                    sum([len(success_moments[s]) for s in success_moments])))

  finally:
    # save the data here
    for scene in success_moments:
      target_file = os.path.join(args.moment_path, "%s.json" % scene)
      with open(target_file, "w") as f:
        json.dump(success_moments[scene], f)
    if args.log_file is not None:
      with open(args.log_file, "w") as f:
        json.dump(fails, f)

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    if local_actor_list:
      cleanup_actors(local_actor_list, client)
    cleanup_actors(global_actor_list, client)

    pygame.quit()
