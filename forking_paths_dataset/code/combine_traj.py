# coding=utf-8
"""given the train/val/test trajectory files, combine into each video."""
# if given homography matrices,

import argparse
import json
import operator
import os

from glob import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("split_path")
parser.add_argument("target_path")
parser.add_argument("frame_file",
                    help="a json file with all used frames.")
parser.add_argument("--reverse_xy", action="store_true", help="ETH/UCY")
parser.add_argument("--is_actev", action="store_true")
parser.add_argument("--h_path", default=None, help="path to homography matrics")
parser.add_argument("--target_w_path", default=None)


def parse_line(traj_line, reverse_xy):
  if reverse_xy:
    frame_id, person_id, y, x = traj_line.strip().split("\t")
  else:
    frame_id, person_id, x, y = traj_line.strip().split("\t")
  frame_id = float(frame_id)
  person_id = float(person_id)
  x, y = float(x), float(y)
  return [frame_id, person_id, x, y]


def get_scene(videoname_):
  """ActEV scene extractor from videoname."""
  s = videoname_.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]


def get_world_coordinates(img_xy, h_):
  """Transform image xy to world ground plane xy."""
  img_x, img_y = img_xy
  world_x, world_y, world_z = np.tensordot(
      h_, np.array([img_x, img_y, 1]), axes=1)
  return [world_x / world_z, world_y / world_z]


def save_files(target_path, traj_data):
  for videoname in traj_data:
    new_file = os.path.join(target_path, "%s.txt" % videoname)
    traj_data[videoname].sort(key=operator.itemgetter(0))
    with open(new_file, "w") as f:
      for frame_id_, person_id_, x, y in traj_data[videoname]:
        f.writelines("%.1f\t%.1f\t%.3f\t%.3f\n" % (frame_id_, person_id_, x, y))

if __name__ == "__main__":
  args = parser.parse_args()

  if args.is_actev:
    assert args.h_path is not None
    assert args.target_w_path is not None
    if not os.path.exists(args.target_w_path):
      os.makedirs(args.target_w_path)
    h_dict = {}
    h_files = glob(os.path.join(args.h_path, "*.txt"))
    for h_file in h_files:
      scene = os.path.splitext(os.path.basename(h_file))[0]
      h_matrix = []
      with open(h_file, "r") as f:
        for line in f:
          h_matrix.append(line.strip().split(","))
      h_matrix = np.array(h_matrix, dtype="float")

      h_dict[scene] = h_matrix

  # assuming no overlap of data
  all_trajs = {}  # videoname -> list
  all_trajs_h = {}  # transformed ones

  all_frames = {}

  for split in ["train", "val", "test"]:
    allfiles = glob(os.path.join(args.split_path, split, "*.txt"))

    for traj_file in allfiles:
      filename = os.path.splitext(os.path.basename(traj_file))[0]

      trajs = [parse_line(line.strip(), args.reverse_xy)
               for line in open(traj_file).readlines()]

      if filename not in all_trajs:
        all_trajs[filename] = []
      all_trajs[filename] += trajs

      if filename not in all_frames:
        all_frames[filename] = {}
      all_frames[filename].update({frame_id:1 for frame_id, _, _, _ in trajs})

      if args.is_actev:
        # get the world coordinates as well
        scene = get_scene(filename)
        H_matrix = h_dict[scene]
        trajs_world = []
        for frame_id, person_id, x, y in trajs:
          if scene == "0002":
            # all trajectory is under 1920x1080, but original 0002 is 1280x720
            x = x * (1280 / 1920.0)
            y = y * (720 / 1080.0)
          w_x, w_y = get_world_coordinates((x, y), H_matrix)
          # ************ notice this negative, it is special fix for actev
          # otherwise the visualization will be mirrored
          w_x = - w_x
          trajs_world.append([frame_id, person_id, w_x, w_y])
        if filename not in all_trajs_h:
          all_trajs_h[filename] = []
        all_trajs_h[filename] += trajs_world

  for filename in all_frames:
    all_frames[filename] = sorted(all_frames[filename].keys())
  with open(args.frame_file, "w") as f:
    json.dump(all_frames, f)

  if not os.path.exists(args.target_path):
    os.makedirs(args.target_path)
  save_files(args.target_path, all_trajs)
  if args.is_actev:
    save_files(args.target_w_path, all_trajs_h)
