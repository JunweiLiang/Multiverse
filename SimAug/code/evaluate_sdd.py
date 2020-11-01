# coding=utf-8
"""Given the eval output pickle file, evaluate."""
from __future__ import print_function

import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("changelst", help="the resize records")
parser.add_argument("outp")
parser.add_argument("--eval_grid", type=int, default=0)

def parse_seq_id(seq_id):
  if str(type(seq_id)) == "<class 'numpy.str_'>":
    stuff = seq_id.split("_")
  else:
    stuff = seq_id.decode().split("_")
  return "_".join(stuff[:2]), stuff[-2], stuff[-1]

if __name__ == "__main__":
  args = parser.parse_args()

  eval_resolution = (1920.0, 1080.0)
  # load the change lst
  resized = {}  # video_id -> scale
  for line in open(args.changelst).readlines():
    video_id, ori_reso, rotated = line.strip().split(",")
    w, h = ori_reso.split("x")
    if rotated == "True":
      w, h = h, w
    w, h = float(w), float(h)
    resized[video_id] = (w / eval_resolution[0] + h / eval_resolution[1]) / 2.0

  with open(args.outp, "rb") as f:
    data = pickle.load(f)

  # [N, T_pred, 2]
  pred_gt = np.array(data["pred_gt_list"])
  #print(pred_gt.shape)
  pred_traj = np.array(data["grid%s_pred_traj" % args.eval_grid])

  diffs = []
  scale_changes = []

  for n in range(len(pred_gt)):
    seq_id = data["seq_ids"][n]
    video_id, frame_idx, track_id = parse_seq_id(seq_id)
    diff = pred_gt[n] - pred_traj[n]  # [T_pred, 2]
    diff = diff ** 2
    diff = np.sqrt(np.sum(diff, axis=1))  # [T_pred]

    # changing the scale
    # so if original resolution is higher than 1920x1080,
    # the error should be linearly scaled up
    diff *= resized[video_id]
    scale_changes.append(resized[video_id])

    diffs.append(diff)

  ade = [t for o in diffs for t in o]
  # final displacement
  fde = [o[-1] for o in diffs]

  ade = np.mean(ade)
  fde = np.mean(fde)
  print("grid %s, ade/fde %s,%s, scale_changes %.5f" % (
      args.eval_grid, ade, fde, np.mean(scale_changes)))

