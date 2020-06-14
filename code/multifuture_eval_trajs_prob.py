# coding=utf-8
"""Given the multifuture trajectory output, compute NLL"""

import argparse
import os
import pickle
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("gt_path")
parser.add_argument("prediction_file")
parser.add_argument("--scene_h", type=int, default=18)
parser.add_argument("--scene_w", type=int, default=32)
parser.add_argument("--video_h", type=int, default=1080)
parser.add_argument("--video_w", type=int, default=1920)

def softmax(x, axis=None):
  """Stable softmax."""
  x = x - x.max(axis=axis, keepdims=True)
  y = np.exp(x)
  return y / y.sum(axis=axis, keepdims=True)

def get_hw_prob(beams, probs, t):
  # beams [beam_size, T, h*w]
  # probs [beam_size]
  beam_size, _, hw = beams.shape
  new_prob = np.zeros((beam_size, hw), dtype="float32")
  for b in range(beam_size):
    new_prob[b, :] = beams[b, t, :] * probs[b]
  # sum to 1
  new_prob = np.sum(new_prob, axis=0) # [h*w]
  return new_prob

def compute_nll(pred_probs, gt_indexes):
  # pred_probs is [h*w], sum to 1
  # gt_indexs is K index from 0 to h*w -1
  nll = 0.0
  for gt_index in gt_indexes:
    nll += -np.log(pred_probs[gt_index] + np.finfo(float).eps)
  nll /= len(gt_indexes)
  return nll

def xys_to_indexes(xys, args):
  # xys [K, 2]
  x_indexes = np.ceil(xys[:, 0] / args.w_gap)

  y_indexes = np.ceil(xys[:, 1] / args.h_gap)
  x_indexes = np.asarray(x_indexes, dtype="int")
  y_indexes = np.asarray(y_indexes, dtype="int")

  # ceil(0.0) = 0.0, we need
  x_indexes[x_indexes == 0] = 1
  y_indexes[y_indexes == 0] = 1
  x_indexes = x_indexes - 1
  y_indexes = y_indexes - 1

  one_hot = np.zeros((len(xys), args.scene_h, args.scene_w), dtype="uint8")
  one_hot[range(len(xys)), y_indexes, x_indexes] = 1
  one_hot_flat = one_hot.reshape((len(xys), -1))  # [len(xys), h*w]
  classes = np.argmax(one_hot_flat, axis=1)  # [len(xys)]
  return classes.tolist()



if __name__ == "__main__":
  args = parser.parse_args()

  args.w_gap = args.video_w*1.0/args.scene_w
  args.h_gap = args.video_h*1.0/args.scene_h

  with open(args.prediction_file, "rb") as f:
    predictions = pickle.load(f)  # traj_id -> [1, beam_size, T, h*W]
  # T ~ [14, 25]
  time_list = [0, 1, 2, 3, 4]  # 5 frame, 2second and 10 frame 4 second length prediction
  # NLL for each sample
  nlls = {}
  for timestep in time_list:
    nlls["T=%d" % (timestep+1)] = []

  for traj_id in tqdm(predictions):
    camera = traj_id.split("_")[-1]  # cam4 ...
    gt_file = os.path.join(args.gt_path, "%s.p" % traj_id)
    with open(gt_file, "rb") as f:
      gt = pickle.load(f)  # annotation_key -> x_agent_traj

    # [1, beam_size, T, H*W] and beam_size of prob
    beams, logprobs = predictions[traj_id]
    # normalize the prob first
    logprobs = softmax(np.squeeze(logprobs))
    beams = softmax(np.squeeze(beams), axis=-1)
    assert beams.shape[-1] == args.scene_h * args.scene_w
    # time_list number of h*w
    grid_probs = [get_hw_prob(beams, logprobs, t) for t in time_list]

    for i, timestep in enumerate(time_list):
      gt_xys = []
      for future_id in gt:
        if len(gt[future_id]["x_agent_traj"]) <= timestep:
          continue
        x, y = gt[future_id]["x_agent_traj"][timestep][2:]
        gt_xys.append([x, y])
      if not gt_xys:
        continue
      # a list of indices between 1 and h*w
      gt_indexes = xys_to_indexes(np.asarray(gt_xys), args)
      nll = compute_nll(grid_probs[i], gt_indexes)
      nlls["T=%d" % (timestep+1)].append(nll)

  print([len(nlls[k]) for k in nlls])
  print("NLL:")
  keys = sorted(nlls.keys())
  print(" ".join(keys))
  print(" ".join(["%s" % np.mean(nlls[k])
                  for k in keys]))



