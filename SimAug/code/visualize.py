# coding=utf-8
"""visualize the prediction output."""

from __future__ import print_function

import argparse
import cv2
import pickle
import os
import tqdm
import random
import numpy as np
from pred_utils import get_scene

import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("outp", help="output pickle file")
parser.add_argument("vis_path")
parser.add_argument("video_frame_path", help="path to videoname/frame.jpg")
parser.add_argument("--vis_start", type=int, default=0)
parser.add_argument("--vis_end", type=int, default=-1)
parser.add_argument("--use_beam_search", action="store_true")
parser.add_argument("--show_scene_scale", type=int, default=0)
parser.add_argument("--beam_size", type=int, default=5)

parser.add_argument("--only_video", default=None)
parser.add_argument("--only_after_frameid", default=None, type=int)
parser.add_argument("--only_trackid", default=None, type=int)
parser.add_argument("--no_first_step", action="store_true")
parser.add_argument("--no_pred_traj", action="store_true")
parser.add_argument("--no_gt_pred", action="store_true")


def plot_traj(img, traj, color, thickness=4):
  traj = np.array(traj, dtype="float32")
  points = zip(traj[:-1], traj[1:])

  for p1, p2 in points:
    #img = cv2.arrowedLine(img, tuple(p1), tuple(p2), color=color, thickness=1,
    #                      tipLength=0.2)
    img = cv2.line(img, tuple(p1), tuple(p2), color=color, thickness=thickness)

  return img


def draw_grid(img, scene_grid):
  imgh, imgw, _ = img.shape
  scene_h, scene_w = scene_grid
  per_grid_w, per_grid_h = imgw / float(scene_w), imgh / float(scene_h)
  thickness = 1
  color = (255, 0, 0)

  for h_ in range(scene_h):
    y = int(per_grid_h * h_)
    img = cv2.arrowedLine(img, (0, y), (imgw, y),
                          color=color, thickness=thickness, tipLength=0)
  for w_ in range(scene_w):
    x = int(per_grid_w * w_)
    img = cv2.arrowedLine(img, (x, 0), (x, imgh),
                          color=color, thickness=thickness, tipLength=0)

  return img

def draw_grid_class_pred_at_t(img, colormap, args, data):
  """Draw the grid classification using heatmap."""

  H, W = args.scene_grids[args.show_scene_scale]

  # [H*W]
  pred_prob_at_t = data

  # softmax is clearer to see

  #pred_prob_at_t = (pred_prob_at_t - pred_prob_at_t.min()) / \
  #                 (pred_prob_at_t.max() - pred_prob_at_t.min())
  pred_prob_at_t = softmax(pred_prob_at_t)

  add_prob = np.zeros((H, W), dtype="float")
  # for fig 1, add some bubbles
  add_prob[6, 11] = 0.012
  add_prob[6, 12] = 0.021
  add_prob[6, 13] = 0.005
  add_prob[5, 15] = 0.031
  add_prob[5, 15] = 0.015
  add_prob = add_prob.reshape([-1])


  # k is from [0, H*W)
  new_layer = np.zeros((imgh, imgw), dtype="float")
  for k in range(len(args.grid_centers[args.show_scene_scale])):
    center_x, center_y = args.grid_centers[args.show_scene_scale][k]
    center_x, center_y = int(center_x), int(center_y)
    prob = pred_prob_at_t[k]

    # for fig1
    add = add_prob[k]

    new_layer[center_y, center_x] = prob + add

  f_new_layer = gaussian_filter(new_layer, sigma=10)
  # linearly scale the matrix again since gaussian makes the numbers small
  f_new_layer = (f_new_layer - f_new_layer.min()) / \
                (f_new_layer.max() - f_new_layer.min())
  f_new_layer = np.uint8(f_new_layer * 255)
  ret, mask = cv2.threshold(f_new_layer, 1, 255, cv2.THRESH_BINARY)
  heatmap_img = cv2.applyColorMap(f_new_layer, colormap)
  heatmap_img_masked = cv2.bitwise_and(
      heatmap_img, heatmap_img, mask=mask)
  return cv2.addWeighted(img, 1.0, heatmap_img_masked, 0.7, 0)

def draw_grid_class_pred_through_t(img, colormap, args, data, beam_label):
  """Draw the grid classification using heatmap."""

  H, W = args.scene_grids[args.show_scene_scale]

  # k is from [0, H*W)
  new_layer = np.zeros((imgh, imgw), dtype="float")
  for t in range(len(data)):
    grid_class = data[t]  # classid from 0 to H*W-1
    center_x, center_y = args.grid_centers[args.show_scene_scale][grid_class]
    center_x, center_y = int(center_x), int(center_y)
    if t==0:
      beam_label_xy = center_x, center_y
    prob = (t+1)/2.0
    new_layer[center_y, center_x] = prob

  img = cv2.putText(img, beam_label, beam_label_xy,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(255, 0, 0))

  f_new_layer = gaussian_filter(new_layer, sigma=10)
  # linearly scale the matrix again since gaussian makes the numbers small
  f_new_layer = (f_new_layer - f_new_layer.min()) / \
                (f_new_layer.max() - f_new_layer.min())
  f_new_layer = np.uint8(f_new_layer * 255)
  ret, mask = cv2.threshold(f_new_layer, 1, 255, cv2.THRESH_BINARY)
  heatmap_img = cv2.applyColorMap(f_new_layer, colormap)
  heatmap_img_masked = cv2.bitwise_and(
      heatmap_img, heatmap_img, mask=mask)
  return cv2.addWeighted(img, 1.0, heatmap_img_masked, 0.7, 0)

if __name__ == "__main__":
  args = parser.parse_args()

  with open(args.outp, "r") as f:
    data = pickle.load(f)

  num_data = len(data["seq_ids"])

  args.obs_len = 8
  args.frame_gap = 12
  args.vw = 1920
  args.vh = 1080
  # has to be 2,4 to match the scene CNN strides
  args.scene_grid_strides = (2, 4)
  args.scene_h, args.scene_w = 36, 64
  args.scene_grids = []
  args.grid_centers = []
  for i, stride in enumerate(args.scene_grid_strides):
    h, w = args.scene_h, args.scene_w
    this_h, this_w = round(h*1.0/stride), round(w*1.0/stride)
    this_h, this_w = int(this_h), int(this_w)
    args.scene_grids.append((this_h, this_w))

    # [H*W, 2]
    args.grid_centers.append(data["grid_center_%d" % i].reshape([-1, 2]))

  # 1. put all data into per frame result
  new_data = {}

  if args.vis_end == -1:
    args.vis_end = num_data

  for i in range(args.vis_start, args.vis_end):
    seq_id = data["seq_ids"][i]
    person_id, frame_id, videoname = seq_id[::-1].split("_", 2)
    videoname, frame_id, person_id = videoname[::-1], \
                                     frame_id[::-1], person_id[::-1]
    scene = get_scene(videoname)
    if (scene == "0002") or (scene == "0400"):
      continue
    person_id = int(person_id)
    frame_id = int(frame_id)

    if args.only_video is not None:
      if videoname != args.only_video:
        continue
    if args.only_after_frameid is not None:
      if frame_id < args.only_after_frameid:
        continue

    if videoname not in new_data:
      new_data[videoname] = {}
    if frame_id not in new_data[videoname]:
      new_data[videoname][frame_id] = {}

    obs_traj, pred_gt_traj = data["obs_list"][i], data["pred_gt_list"][i]
    this_data = {
        # [T, 2]
        "obs_traj": obs_traj,
        "pred_gt_traj": pred_gt_traj,
    }
    #for j in range(len(args.scene_grids)):
    for j in range(args.show_scene_scale+1):
      # [T, 2]
      this_data.update(
          {("grid%s_pred_traj" % j): data["grid%s_pred_traj" % j][i]})
      # [T, H*W]
      this_data.update({("grid%s_class" % j): data["grid%s_class" % j][i]})
      # [T]
      this_data.update(
          {("grid%s_gt_class" % j): data["grid%s_gt_class" % j][i]})
    if args.use_beam_search:
      this_data.update({
          "beam_grid_ids": data["beam_grid_ids"][i],  # [beam_size, T]
          "beam_logprobs": data["beam_logprobs"][i]})
    new_data[videoname][frame_id][person_id] = this_data

  print("total %s videos." % len(new_data))

  for videoname in tqdm.tqdm(new_data):
    frames = sorted(new_data[videoname].keys())

    target_path = os.path.join(args.vis_path, videoname)
    if not os.path.exists(target_path):
      os.makedirs(target_path)

    random.seed(1)
    for frame_id in tqdm.tqdm(frames):

      # 1. get the one frame id for this seq
      last_obs_frame_id = frame_id + (args.obs_len - 1) * args.frame_gap
      frame_file = os.path.join(
          args.video_frame_path, videoname,
          "%s_F_%08d.jpg" % (videoname, last_obs_frame_id))

      frame_img = cv2.imread(frame_file, cv2.IMREAD_COLOR)

      # the 0002 scene we need to resize to 1920x1080
      imgh, imgw, _ = frame_img.shape
      if imgw != args.vw:
        frame_img = cv2.resize(frame_img, (args.vw, args.vh))
      imgh, imgw, _ = frame_img.shape

      # 1. draw the grid first
      frame_img = draw_grid(frame_img, args.scene_grids[args.show_scene_scale])

      # only show one person at a frame, otherwise the heatmap overlap
      person_ids = new_data[videoname][frame_id].keys()
      random.shuffle(person_ids)
      if args.only_trackid is not None:
        person_ids = [args.only_trackid]
        if not new_data[videoname][frame_id].has_key(args.only_trackid):
          continue
      for person_id in person_ids[:1]:
        this_data = new_data[videoname][frame_id][person_id]

        # 1. plot the full gt trajectory
        # [seq_len, 2]
        full_gt_traj = np.concatenate([this_data["obs_traj"],
                                       this_data["pred_gt_traj"]], axis=0)
        frame_img = plot_traj(frame_img, full_gt_traj, (0, 255, 0))
        # then overlay with observable yellow traj
        frame_img = plot_traj(frame_img, this_data["obs_traj"], (0, 255, 255))

        # 2. plot prediction trajectory
        # [pred_len, 2]
        pred_traj = this_data["grid%s_pred_traj" % args.show_scene_scale]
        pred_traj = np.concatenate(
            [this_data["obs_traj"][-1, :].reshape(1, 2), pred_traj], axis=0)
        if not args.no_pred_traj:
          frame_img = plot_traj(frame_img, pred_traj, (255, 255, 0))

        # 3. draw gt class
        pred_gt_class_points = [
            args.grid_centers[args.show_scene_scale][p, :]
            for p in this_data["grid%s_gt_class" % args.show_scene_scale]]

        for point in pred_gt_class_points:
          x, y = point
          if args.no_gt_pred:
            continue
          frame_img = cv2.circle(frame_img, (int(x), int(y)), radius=30,
                                 color=(255, 0, 0))

        # 4. draw pred class logits
        if args.use_beam_search:
          beam2color = {
              0: cv2.COLORMAP_AUTUMN,
              int(args.beam_size/2.0): cv2.COLORMAP_SPRING,
              args.beam_size-1: cv2.COLORMAP_WINTER,
          }
          # beam 0 -> beam_size, first is the best
          for beam in [0, int(args.beam_size/2.0), args.beam_size-1]:
            # cv2 colormap from 0 -> 11
            frame_img = draw_grid_class_pred_through_t(
                frame_img, beam2color[beam], args,
                this_data["beam_grid_ids"][beam],
                #"#%d, %.2f" % (beam, this_data["beam_logprobs"][beam]))
                "#%d" % (beam))
        else:
          if not args.no_first_step:
            frame_img = draw_grid_class_pred_at_t(
                frame_img, cv2.COLORMAP_WINTER, args,
                this_data["grid%s_class" % args.show_scene_scale][0])
          frame_img = draw_grid_class_pred_at_t(
                frame_img, cv2.COLORMAP_AUTUMN, args,
                this_data["grid%s_class" % args.show_scene_scale][-3])
          frame_img = draw_grid_class_pred_at_t(
                frame_img, cv2.COLORMAP_AUTUMN, args,
                this_data["grid%s_class" % args.show_scene_scale][-2])
          frame_img = draw_grid_class_pred_at_t(
              frame_img, cv2.COLORMAP_AUTUMN, args,
              this_data["grid%s_class" % args.show_scene_scale][-1])

      target_file = os.path.join(target_path,
                                 "%s_F_%08d.jpg" % (videoname, frame_id))
      with open(target_file, "w") as f:
        cv2.imwrite(target_file, frame_img)



