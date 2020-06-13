# coding=utf-8
"""Given the carla trajectory file, reconstruct person walking."""

import argparse
import glob
import math
import pygame
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla

import numpy as np

from utils import setup_walker_bps
from utils import setup_vehicle_bps
from utils import setup_static
from utils import static_scenes
from utils import get_scene
from utils import get_controls
from utils import run_sim_for_one_frame

parser = argparse.ArgumentParser()
parser.add_argument("traj_file")
parser.add_argument("start_frame_idx", type=int,
                    help="inclusive")
parser.add_argument("end_frame_idx", type=int,
                    help="inclusive")
parser.add_argument("--vehicle_traj", default=None)

parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=2000, type=int)

parser.add_argument("--show_traj", action="store_true")

parser.add_argument("--vehicle_z", type=float, default=0.0,
                    help="set all vehicle z to this value")

if __name__ == "__main__":
  args = parser.parse_args()
  filename = os.path.splitext(os.path.basename(args.traj_file))[0]
  if filename.startswith("VIRAT"):  # ActEV dataset
    scene = get_scene(filename)
    assert scene in static_scenes
    static_scene = static_scenes[scene]
  else:
    assert filename in static_scenes
    static_scene = static_scenes[filename]

  fps = static_scene["fps"]
  # process the traj first.
  # gather all trajectory control within the frame num
  # frame_id -> list of [person_id, xyz, direction vector, speed]
  ped_controls, total_moment_frame_num = get_controls(
      args.traj_file, args.start_frame_idx, args.end_frame_idx, fps)
  print("Control data prepared.")

  vehicle_controls = {}
  if args.vehicle_traj is not None:
    vehicle_controls, _ = get_controls(
        args.vehicle_traj, args.start_frame_idx, args.end_frame_idx, fps,
        interpolate=True, z_to=args.vehicle_z)

  actorid2info = {}  # carla actorId to the personId or vehicle Id

  try:
    global_actor_list = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()

    walker_bps = setup_walker_bps(world)
    vehicle_bps = setup_vehicle_bps(world)

    # 1. set up the static env
    setup_static(world, client, static_scene, global_actor_list)

    settings = world.get_settings()
    settings.fixed_delta_seconds = 1.0 / fps
    settings.synchronous_mode = True
    world.apply_settings(settings)

    # this is the server side frame Id we start with
    baseline_frame_id = world.tick()

    client_clock = pygame.time.Clock()

    moment_frame_count = 0
    current_peds = {}  # person_id -> actor
    current_ped_collisions = {}  # person_id -> CollisionSensor
    current_vehicles = {}
    vehicle_initial_forward_vector = {}
    vehicle_prev_yaw = {}
    max_yaw_change = 90  # no sudden yaw change
    for moment_frame_count in range(total_moment_frame_num):
      # grab the control data of this frame if any
      batch_cmds, _ = run_sim_for_one_frame(
          moment_frame_count, ped_controls,
          vehicle_controls,
          current_peds, current_ped_collisions,
          current_vehicles,
          vehicle_initial_forward_vector,
          vehicle_prev_yaw,
          walker_bps, vehicle_bps,
          world,
          global_actor_list, actorid2info,
          show_traj=args.show_traj, verbose=True,
          max_yaw_change=max_yaw_change)

      if batch_cmds:
        response = client.apply_batch_sync(batch_cmds)

      # block if faster than fps
      client_clock.tick_busy_loop(fps)
      server_frame_id = world.tick()

  finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    for actor in global_actor_list:
      if actor.type_id.startswith("sensor"):
        actor.stop()
    # finished, clean actors
    client.apply_batch(
        [carla.command.DestroyActor(x) for x in global_actor_list])

    pygame.quit()
