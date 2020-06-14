# coding=utf-8
"""Given the multifuture trajectory output, compute ADE/FDE"""

import argparse
import os
import pickle
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("gt_path")
parser.add_argument("prediction_file")


def get_min(errors):
  # num_pred, [pred_len] / [1]
  sums = [sum(e) for e in errors]
  min_sum = min(sums)
  min_idx = sums.index(min_sum)
  return errors[min_idx], min_idx

if __name__ == "__main__":
  args = parser.parse_args()

  with open(args.prediction_file, "rb") as f:
    prediction = pickle.load(f)

  # error per ground truth trajectory
  ade_errors = {
      "45-degree": [],
      "top-down": [],
      "all": []
  }
  fde_errors = {
      "45-degree": [],
      "top-down": [],
      "all": []
  }

  for traj_id in tqdm(prediction):
    camera = traj_id.split("_")[-1]
    gt_file = os.path.join(args.gt_path, "%s.p" % traj_id)
    with open(gt_file, "rb") as f:
      gt = pickle.load(f)

    # for each ground truth possibilities, get the minimum ADE/FDE prediction's
    # as this error
    for future_id in gt:
      gt_traj = gt[future_id]["x_agent_traj"]  # (frameIdx, pid, x, y)
      gt_traj = np.array([one[2:] for one in gt_traj])
      pred_len = len(gt_traj)

      # compute ADE and FDE for all prediction
      this_ade_errors = []  # [num_pred] [pred_len]
      this_fde_errors = []  # [num_pred] [1]
      for pred_out in prediction[traj_id]:
        assert len(pred_out) >= pred_len
        diff = gt_traj - pred_out[:pred_len]  # [pred_len, 2]
        diff = diff**2
        diff = np.sqrt(np.sum(diff, axis=1))  # [pred_len]
        this_ade_errors.append(diff.tolist())
        this_fde_errors.append([diff[-1]])

      # [pred_len]
      min_ade_errors, min_ade_traj_idx = get_min(this_ade_errors)

      # [1]
      min_fde_errors, min_fde_traj_idx = get_min(this_fde_errors)

      if camera == "cam4":
        ade_errors["top-down"] += min_ade_errors
        fde_errors["top-down"] += min_fde_errors
      else:
        ade_errors["45-degree"] += min_ade_errors
        fde_errors["45-degree"] += min_fde_errors
      ade_errors["all"] += min_ade_errors
      fde_errors["all"] += min_fde_errors

  print("ADE/FDE:")
  keys = ["45-degree", "top-down", "all"]
  print(" ".join(keys + keys))
  print(" ".join(["%s" % np.mean(ade_errors[k])
                  for k in keys] + ["%s" % np.mean(fde_errors[k])
                  for k in keys]))



