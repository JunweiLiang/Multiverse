# coding=utf-8
# plot world traj on the carla ground
import argparse
import glob
import math
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla

import numpy as np
from visualize_real_data import load_traj
from visualize_real_data import get_traj
from visualize_real_data import rotate
from visualize_real_data import get_scene

parser = argparse.ArgumentParser()

# params for getting the trajectory
parser.add_argument("traj_world_file")
parser.add_argument("start_frame_idx", type=int)

# carla mapping
parser.add_argument("origin_x", type=float)
parser.add_argument("origin_y", type=float)
parser.add_argument("origin_z", type=float)
parser.add_argument("carla_rotation", type=float,
                    help="rotate degrees before translate to origin.")

# actev will also get the vehicle traj
parser.add_argument("--is_actev", action="store_true")
parser.add_argument("--vehicle_world_traj_file", default=None)
parser.add_argument("--save_vehicle_carla_traj_file", default=None,
                    help="if set this, will save ALL the 3D carla coor instead")

parser.add_argument("--world_rotate", type=float, default=0.0,
                    help="rotation in degrees")
parser.add_argument("--scale", type=float, default=1.0,
                    help="scaling the meters")
# carla param
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=2000, type=int)

parser.add_argument("--obs_length", type=int, default=8)
parser.add_argument("--pred_length", type=int, default=12)

parser.add_argument("--line_time", type=float, default=30,
                    help="how long does the traj stays, -1 is perminent")

parser.add_argument("--save_carla_traj_file", default=None,
                    help="if set this, will save ALL the 3D carla coor instead")


def plot_trajs_carla(world, trajs, carla_color, z, line_time=30.0,
                     show_person_id=False):

  for person_id, traj in trajs:
    points = zip(traj[:-1], traj[1:])
    for p1, p2 in points:
      p1 = carla.Location(x=p1[0], y=p1[1], z=z)
      p2 = carla.Location(x=p2[0], y=p2[1], z=z)

      world.debug.draw_arrow(
          p1, p2,
          thickness=0.1,
          arrow_size=0.1, color=carla_color, life_time=line_time)
    if show_person_id:
      world.debug.draw_string(
          carla.Location(x=traj[0][0], y=traj[0][1], z=z), "# %s" % person_id,
          draw_shadow=False, color=carla.Color(r=255, g=0, b=0),
          life_time=line_time, persistent_lines=False)


# computed using compute_actev_world_norm.py
# min -> max
actev_norm = {
    "0400": {
        "x": [-113.339996, 15.906000], "y": [-51.101002, 82.049004]
    },
    "0401": {
        "x": [-76.031998, 28.722000], "y": [-3.993000, 90.141998]
    },
    "0000": {
        "x": [-7.510000, 48.320000], "y": [-7.984000, 14.305000]
    },
    "0002": {
        "x": [-38.488998, 67.762001], "y": [-29.208000, 128.421005]
    },
    "0500": {
        "x": [-25.212000, -0.499000], "y": [-25.396999, 35.426998]
    },
}


if __name__ == "__main__":
  args = parser.parse_args()

  traj_world_data = load_traj(args.traj_world_file)
  # 1. preprocess world trajectory
  if args.world_rotate != 0:
    traj_world_data = rotate(
        traj_world_data, (0, 0), math.radians(args.world_rotate))

  # translate to 0, 0, but keeping the meters unit
  if args.is_actev:
    assert args.vehicle_world_traj_file is not None
    vehicle_traj_world_data = load_traj(args.vehicle_world_traj_file)
    if args.world_rotate != 0:
      vehicle_traj_world_data = rotate(
          vehicle_traj_world_data, (0, 0), math.radians(args.world_rotate))
    videoname = os.path.splitext(os.path.basename(args.traj_world_file))[0]
    scene = get_scene(videoname)
    min_x, max_x = actev_norm[scene]["x"]
    min_y, max_y = actev_norm[scene]["y"]
    # prepare the vehicle trajectory # scaling up or down
    # since the meter unit in the carla might be off
    vehicle_traj_stage1 = vehicle_traj_world_data.copy()
    vehicle_traj_stage1[:, 2] = (vehicle_traj_stage1[:, 2] - min_x) * args.scale
    vehicle_traj_stage1[:, 3] = (vehicle_traj_stage1[:, 3] - min_y) * args.scale

    # rotate and translate into designated carla space
    vehicle_traj_stage2 = rotate(
        vehicle_traj_stage1, (0, 0), math.radians(args.carla_rotation))
    vehicle_traj_stage2[:, 2] = vehicle_traj_stage2[:, 2] + args.origin_x
    vehicle_traj_stage2[:, 3] = vehicle_traj_stage2[:, 3] + args.origin_y

    vehicle_ids = np.unique(
        vehicle_traj_world_data[
            vehicle_traj_world_data[:, 0] == args.start_frame_idx, 1])
    vehicle_ids = vehicle_ids.tolist()
  else:
    min_x = np.amin(np.array(traj_world_data)[:, 2])
    max_x = np.amax(np.array(traj_world_data)[:, 2])
    min_y = np.amin(np.array(traj_world_data)[:, 3])
    max_y = np.amax(np.array(traj_world_data)[:, 3])

  # scaling up or down
  # since the meter unit in the carla might be off
  traj_world_stage1 = traj_world_data.copy()
  traj_world_stage1[:, 2] = (traj_world_stage1[:, 2] - min_x) * args.scale
  traj_world_stage1[:, 3] = (traj_world_stage1[:, 3] - min_y) * args.scale

  # rotate and translate into designated carla space
  traj_world_stage2 = rotate(
      traj_world_stage1, (0, 0), math.radians(args.carla_rotation))
  traj_world_stage2[:, 2] = traj_world_stage2[:, 2] + args.origin_x
  traj_world_stage2[:, 3] = traj_world_stage2[:, 3] + args.origin_y

  if args.save_carla_traj_file is not None:
    with open(args.save_carla_traj_file, "w") as f:
      for frame_id, person_id, x, y in traj_world_stage2:
        f.writelines("%d\t%d\t%.6f\t%.6f\t%.6f\n" % (
            frame_id, person_id, x, y, args.origin_z))
      if args.is_actev:
        assert args.save_vehicle_carla_traj_file is not None
        with open(args.save_vehicle_carla_traj_file, "w") as f:
          for frame_id, vehicle_id, x, y in vehicle_traj_stage2:
            f.writelines("%d\t%d\t%.6f\t%.6f\t%.6f\n" % (
                frame_id, vehicle_id, x, y, args.origin_z))
    sys.exit()

  # get the specific trajectories
  frame_ids = np.unique(traj_world_data[:, 0]).tolist()
  frame_ids.sort()
  f_idx = frame_ids.index(args.start_frame_idx)
  # we will draw pred_frame first then obs frame traj so there is 2 color?
  obs_frame_ids = frame_ids[f_idx:f_idx + args.obs_length]
  full_frame_ids = frame_ids[f_idx:f_idx + args.obs_length + args.pred_length]

  # all the person in frame_ids
  person_ids = np.unique(
      traj_world_data[np.isin(traj_world_data[:, 0], full_frame_ids), 1])
  person_ids = person_ids.tolist()

  # (person_id, list of xy)
  obs_person_trajs = [get_traj(
      traj_world_stage2, obs_frame_ids, person_id) for person_id in person_ids]
  full_person_trajs = [get_traj(
      traj_world_stage2, full_frame_ids, person_id) for person_id in person_ids]

  if args.is_actev:
    full_vehicle_trajs = [get_traj(
        vehicle_traj_stage2, full_frame_ids, vehicle_id)
                          for vehicle_id in vehicle_ids]

  # plot the trajectories and the person Id
  try:
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()

    green = carla.Color(r=0, g=255, b=0)
    yellow = carla.Color(r=255, g=255, b=0)
    plot_trajs_carla(world, full_person_trajs, green, args.origin_z,
                     show_person_id=True, line_time=args.line_time)
    plot_trajs_carla(world, obs_person_trajs, yellow, args.origin_z,
                     line_time=args.line_time)

    if args.is_actev:
      blue = carla.Color(r=0, g=0, b=255)
      plot_trajs_carla(world, full_vehicle_trajs, blue, args.origin_z,
                       line_time=args.line_time, show_person_id=True)

  finally:
    pass
