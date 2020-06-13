# coding=utf-8
"""Batch convert the world traj in actev to carla traj."""

import argparse
import os
from glob import glob
from tqdm import tqdm
import sys
if sys.version_info > (3, 0):
  import subprocess as commands
else:
  import commands


parser = argparse.ArgumentParser()
parser.add_argument("traj_world_path")
parser.add_argument("--traj_vehicle_world_path", default=None)
parser.add_argument("save_carla_traj_path")
parser.add_argument("--save_carla_vehicle_path", default=None)


calibrations = {
    "0000": {
        "world_rotate": 320,
        "carla_rotate": 130,
        "scale": 1.0,
        "origin": [3.5, -48.0, 0.3]
    },
    "0400": {
        "world_rotate": 100,
        "carla_rotate": 153,
        "scale": 1.0,
        "origin": [-10.0, 58.0, 0.5]
    },
    "0401": {
        "world_rotate": 120,
        "carla_rotate": 135,
        "scale": 1.0,
        "origin": [-48.0, 24.0, 0.5]
    },
    "0500": {
        "world_rotate": 90,
        "carla_rotate": 179,
        "scale": 1.0,
        "origin": [-65.5, -75.5, 0.1]
    },
}

# Zara
calibration = {
    "world_rotate": 270,
    "carla_rotate": -3.04,
    "scale": 1.2,
    "origin": [-44.0511921243, -79.6225002047, 0.],
}


def get_scene(videoname_):
  """ActEV scene extractor from videoname."""
  s = videoname_.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]


if __name__ == "__main__":
  args = parser.parse_args()

  # all files
  ped_traj_files = glob(os.path.join(args.traj_world_path, "*.txt"))
  if args.traj_vehicle_world_path is not None:
    assert args.save_carla_vehicle_path is not None
    if not os.path.exists(args.save_carla_vehicle_path):
      os.makedirs(args.save_carla_vehicle_path)
  if not os.path.exists(args.save_carla_traj_path):
    os.makedirs(args.save_carla_traj_path)

  script_path = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "plot_traj_carla.py")
  assert os.path.exists(script_path), script_path

  for ped_traj_file in tqdm(ped_traj_files):
    filename = os.path.splitext(os.path.basename(ped_traj_file))[0]
    target_ped_file = os.path.join(
        args.save_carla_traj_path, "%s.txt" % filename)

    if args.traj_vehicle_world_path is None:
      output = commands.getoutput("python3 %s %s 0 %f %f %f %f --world_rotate"
                                  " %f --scale %f --save_carla_traj_file %s" % (
                                      script_path, ped_traj_file,
                                      calibration["origin"][0],
                                      calibration["origin"][1],
                                      calibration["origin"][2],
                                      calibration["carla_rotate"],
                                      calibration["world_rotate"],
                                      calibration["scale"],
                                      target_ped_file))
    else:
      scene = get_scene(filename)
      if scene == "0002":
        continue
      vehicle_traj_file = os.path.join(args.traj_vehicle_world_path,
                                       "%s.txt" % filename)
      target_vehicle_file = os.path.join(args.save_carla_vehicle_path,
                                         "%s.txt" % filename)
      cmd = "python3 %s %s 0 %f %f %f %f --world_rotate" \
            " %f --scale %f --save_carla_traj_file %s" \
            " --vehicle_world_traj_file %s" \
            " --save_vehicle_carla_traj_file %s" % (
                script_path, ped_traj_file,
                calibrations[scene]["origin"][0],
                calibrations[scene]["origin"][1],
                calibrations[scene]["origin"][2],
                calibrations[scene]["carla_rotate"],
                calibrations[scene]["world_rotate"],
                calibrations[scene]["scale"],
                target_ped_file,
                vehicle_traj_file,
                target_vehicle_file)
      output = commands.getoutput("python3 %s %s 0 %f %f %f %f --world_rotate"
                                  " %f --scale %f --save_carla_traj_file %s"
                                  " --vehicle_world_traj_file %s --is_actev"
                                  " --save_vehicle_carla_traj_file %s" % (
                                      script_path, ped_traj_file,
                                      calibrations[scene]["origin"][0],
                                      calibrations[scene]["origin"][1],
                                      calibrations[scene]["origin"][2],
                                      calibrations[scene]["carla_rotate"],
                                      calibrations[scene]["world_rotate"],
                                      calibrations[scene]["scale"],
                                      target_ped_file,
                                      vehicle_traj_file,
                                      target_vehicle_file))
