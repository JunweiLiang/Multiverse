# coding=utf-8
"""Visualize one frame, multifuture ground truth and prediction."""

import argparse
import cv2
import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("gt_path")
parser.add_argument("prediction_file")
parser.add_argument("multivideo_path")
parser.add_argument("vis_path")  # traj_id
parser.add_argument("--show_obs", action="store_true")
parser.add_argument("--plot_points", action="store_true")
parser.add_argument("--use_heatmap", action="store_true")
parser.add_argument("--show_less_gt", action="store_true")
parser.add_argument("--drop_frame", type=int, default=1)
parser.add_argument("--job", type=int, default=1, help="total job")
parser.add_argument("--curJob", type=int, default=1,
                    help="split the moment for this annotator.")


def get_valid_idx(trajs, args):
  new_traj_list = []
  for traj in trajs:
    traj = np.array(traj)
    traj_indexed = np.zeros_like(traj)
    for i, (x,y) in enumerate(traj):
      x = round(x) - 1
      y = round(y) - 1
      if x < 0:
        x = 0
      if y < 0:
        y=0
      if x >= args.imgw:
        x = args.imgw - 1
      if y >= args.imgh:
        y = args.imgh - 1
      traj_indexed[i] = x, y
    new_traj_list.append(traj_indexed)
  return new_traj_list

# traj is a list of xy tuple
def plot_traj(img, traj, color):
  """Plot a trajectory on image."""
  traj = np.array(traj, dtype="float32")
  points = zip(traj[:-1], traj[1:])

  for p1, p2 in points:
    img = cv2.line(img, tuple(p1), tuple(p2), color=color, thickness=2)

  return img

if __name__ == "__main__":
  args = parser.parse_args()
  args.imgh, args.imgw = 1080, 1920

  with open(args.prediction_file, "rb") as f:
    prediction = pickle.load(f)

  if not os.path.exists(args.vis_path):
    os.makedirs(args.vis_path)

  count = 0
  for traj_id in tqdm(prediction):
    count += 1
    if (count % args.job) != (args.curJob - 1):
      continue
    gt_file = os.path.join(args.gt_path, "%s.p" % traj_id)
    with open(gt_file, "rb") as f:
      gt = pickle.load(f)

    video_file = os.path.join(args.multivideo_path, "%s.mp4" % traj_id)

    target_path = os.path.join(args.vis_path, "%s" % traj_id)
    if not os.path.exists(target_path):
      os.makedirs(target_path)

    vcap = cv2.VideoCapture(video_file)
    if not vcap.isOpened():
      raise Exception("Cannot open %s" % video_file)
    frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    cur_frame = 0
    printed_frame_count = 0
    while cur_frame < frame_count:
      _, frame_data = vcap.read()
      if cur_frame % args.drop_frame != 0:
        cur_frame += 1
        continue

      # get length first
      max_gt_pred_length = 0
      for future_id in gt:
        points = gt[future_id]["x_agent_traj"]  # (frameIdx, pid, x, y)
        points = [one[2:] for one in points]
        max_gt_pred_length = max([len(points), max_gt_pred_length])

      # 1. plot all the outputs

      if args.use_heatmap:
        new_layer = np.zeros((args.imgh, args.imgw), dtype="float")
        num_between_line = 40
        # convert all the point into valid index
        trajs_indexed = get_valid_idx(prediction[traj_id], args)
        for traj_indexed in trajs_indexed:
          for (x1, y1), (x2, y2) in zip(traj_indexed[:-1], traj_indexed[1:]):
            # all x,y between
            xs = np.linspace(x1, x2, num=num_between_line, endpoint=True)
            ys = np.linspace(y1, y2, num=num_between_line, endpoint=True)
            points = zip(xs, ys)
            for x, y in points:
              x = int(x)
              y = int(y)
              new_layer[y, x] = 1.0

        # gaussian interpolate
        from scipy.ndimage import gaussian_filter

        f_new_layer = gaussian_filter(new_layer, sigma=10)

        f_new_layer = np.uint8(f_new_layer*255)

        ret, mask = cv2.threshold(f_new_layer, 1, 255, cv2.THRESH_BINARY)

        #print new_layer

        heatmap_img = cv2.applyColorMap(f_new_layer, cv2.COLORMAP_AUTUMN)

        heatmap_img_masked = cv2.bitwise_and(heatmap_img,heatmap_img, mask=mask)

        frame_data = cv2.addWeighted(frame_data, 1.0, heatmap_img_masked, 1.0, 0)

      # 2. plot all the ground truth first
      for future_id in gt:
        points = gt[future_id]["x_agent_traj"]  # (frameIdx, pid, x, y)
        gt_len = len(points)
        if args.show_less_gt:
          gt_len = max_gt_pred_length/2
        points = [one[2:] for one in points[:gt_len]]
        frame_data = plot_traj(frame_data, points, (0, 255, 0))

        if args.show_obs:
          frame_data = plot_traj(frame_data,
                                 [one[2:] for one in gt[future_id]["obs_traj"]],
                                 (0, 255, 255))

      # plot the predicted trajectory
      for pred_out in prediction[traj_id]:
        if args.plot_points:
          for x, y in pred_out[:max_gt_pred_length]:
            frame_data = cv2.circle(frame_data, (int(x), int(y)), radius=5,
                                    color=(255, 0, 0), thickness=1)
        if not args.use_heatmap:
          frame_data = plot_traj(frame_data, pred_out[:max_gt_pred_length],
                                 (0, 0, 255))

      target_file = os.path.join(target_path, "%08d.jpg" % printed_frame_count)
      cv2.imwrite(target_file, frame_data)
      printed_frame_count += 1

      cur_frame += 1
