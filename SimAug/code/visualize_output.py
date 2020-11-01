# coding=utf-8
# given a list of pickle output with ground truth, and the frame path, visualize
# the prediction and ground truth

import argparse
import pickle
import cv2
import os
import numpy as np
import random

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

from visualize import plot_traj
from pred_utils import get_scene

from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("outlist",
                    help="each line is path to pickle file, and the rgb")
parser.add_argument("framepath")
parser.add_argument("outpath")
parser.add_argument("--vis_num", type=int, default=500)
parser.add_argument("--use_heatmap", action="store_true")
parser.add_argument("--ordered", action="store_true")
parser.add_argument("--only_scene", default=None)


def parse_seq_id(key):
  if str(type(key)) != "<class 'numpy.str_'>":
    key = key.decode()

  stuff = key[::-1].split("_", 2)[::-1]
  stuff = [o[::-1] for o in stuff]
  return stuff

if __name__ == "__main__":
  args = parser.parse_args()

  output_files = [line.strip().split(",")
                  for line in open(args.outlist, "r").readlines()]
  run2colormap = {}
  colormaps = [
      cv2.COLORMAP_WINTER,
      cv2.COLORMAP_AUTUMN,
  ]

  obs_color = (0, 255, 255) # BGR
  gt_pred_color = (0, 255, 0)
  # load all the data
  traj_data = {}  # filename(runname) -> seq_id -> prediction_traj
  run2color = {}
  runlist = []
  gt_obs = {}  # seq_id
  gt_pred = {}
  print("loading outputs...")
  for output_file, color in tqdm(output_files):
    runname = os.path.splitext(os.path.basename(output_file))[0]
    bgr = tuple([int(o) for o in color.split("_")])
    run2color[runname] = bgr
    run2colormap[runname] = colormaps.pop()
    traj_data[runname] = {}
    runlist.append(runname)

    with open(output_file, "rb") as f:
      data = pickle.load(f)

    for j in range(len(data["seq_ids"])):
      seq_id = data["seq_ids"][j]
      videoname, frame_idx, track_id = parse_seq_id(seq_id)

      traj_data[runname][seq_id] = data["grid0_pred_traj"][j]  # [T, 2]

      if seq_id not in gt_obs:
        gt_obs[seq_id] = data["obs_list"][j]
        gt_pred[seq_id] = data["pred_gt_list"][j]

  print("loaded.")
  if not os.path.exists(args.outpath):
    os.makedirs(args.outpath)
  # random ordered
  all_seq_ids = list(gt_obs.keys())
  if args.ordered:
    all_seq_ids.sort()
  else:
    random.shuffle(all_seq_ids)
  if args.only_scene is not None:
    new_all_seq_ids = []
    for seq_id in all_seq_ids:
      videoname, frame_idx, track_id = parse_seq_id(seq_id)
      scene = get_scene(videoname)
      if scene == args.only_scene:
        new_all_seq_ids.append(seq_id)
    all_seq_ids = new_all_seq_ids

  for seq_id in tqdm(all_seq_ids[:args.vis_num]):
    videoname, frame_idx, track_id = parse_seq_id(seq_id)

    frame_file = os.path.join(
        args.framepath, videoname,
        "%s_F_%08d.jpg" % (videoname, int(frame_idx)))

    frame_img = cv2.imread(frame_file, cv2.IMREAD_COLOR)

    imgh, imgw, _= frame_img.shape

    # draw each prediction with their color
    for runname in runlist[::-1]:  # so the first one on top
      pred_traj = traj_data[runname][seq_id]
      pred_traj = np.concatenate(
          [gt_obs[seq_id][-1, :].reshape(1, 2), pred_traj], axis=0)
      if not args.use_heatmap:
        frame_img = plot_traj(frame_img, pred_traj, run2color[runname],
                              thickness=5)
      else:

        new_layer = np.zeros((imgh, imgw), dtype="float")
        colormap = run2colormap[runname]
        num_between_line = 40
        this_traj = pred_traj
        # convert all the point into valid index
        traj_indexed = np.zeros_like(this_traj)
        for i, (x, y) in enumerate(this_traj): # [13,2], make all xy along this line be 1.0
          x = round(x) - 1
          y = round(y) - 1
          if x < 0:
            x = 0
          if y < 0:
            y=0
          if x >= imgw:
            x = imgw - 1
          if y >= imgh:
            y= imgh - 1
          traj_indexed[i] = x, y

        for i, ((x1, y1), (x2, y2)) in enumerate(
              zip(traj_indexed[:-1], traj_indexed[1:])):
          # all x,y between
          xs = np.linspace(x1, x2, num=num_between_line, endpoint=True)
          ys = np.linspace(y1, y2, num=num_between_line, endpoint=True)
          points = list(zip(xs, ys))
          for x, y in points[:-1]:
            x = int(x)
            y = int(y)
            new_layer[y, x] = 1.0 + 1.6**i

        f_new_layer = gaussian_filter(new_layer, sigma=7)
        #f_new_layer = gaussian_filter(f_new_layer, sigma=5)
        f_new_layer /= f_new_layer.max()
        f_new_layer = np.uint8(f_new_layer * 255)

        ret, mask = cv2.threshold(f_new_layer, 1, 255, cv2.THRESH_BINARY)

        #print new_layer

        heatmap_img = cv2.applyColorMap(f_new_layer, colormap)

        heatmap_img_masked = cv2.bitwise_and(heatmap_img, heatmap_img,
                                             mask=mask)

        frame_img = cv2.addWeighted(frame_img, 1.0, heatmap_img_masked, 0.9, 0)

    # [seq_len, 2]
    full_gt_traj = np.concatenate([gt_obs[seq_id],
                                   gt_pred[seq_id]], axis=0)
    frame_img = plot_traj(frame_img, full_gt_traj, gt_pred_color, thickness=2)
    # then overlay with observable yellow traj
    frame_img = plot_traj(frame_img, gt_obs[seq_id], obs_color, thickness=2)

    target_file = os.path.join(args.outpath, "%s.jpg" % seq_id.decode("utf-8"))
    cv2.imwrite(target_file, frame_img)





