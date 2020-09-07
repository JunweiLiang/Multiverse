# coding=utf-8
"""Given the prepared trajectories and original dataset, get frames and seg."""

import argparse
import cv2
import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("traj_path")
parser.add_argument("rgb_video_path")
parser.add_argument("seg_video_path")
parser.add_argument("out_frame_path")
parser.add_argument("out_seg_feat_path", help="save as npy files")
parser.add_argument("bad_video_lst",
                    help="save the video lst that rgb and scene seg dont"
                         " match.")
parser.add_argument("--scene_h", default=36, type=int)
parser.add_argument("--scene_w", default=64, type=int)
parser.add_argument("--is_multifuture", action="store_true",
                    help="The traj file name is not the video name")
parser.add_argument("--is_debug", action="store_true",
                    help="save the carla class back to rgb")

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def get_frame_idxs(traj_file, start=0):
  frame_idxs = {}
  for line in open(traj_file).readlines():
    frame_idx, pid, x, y = line.strip().split("\t")
    frame_idx = int(frame_idx) + start
    frame_idxs[frame_idx] = 1
  return sorted(frame_idxs)

# https://carla.readthedocs.io/en/latest/cameras_and_sensors/#sensorcamerasemantic_segmentation
carla_to_ade20k = {
    0: 0,
    1: 2,
    2: 33,
    3: 0,
    4: 13,  # person
    5: 94,
    6: 7,
    7: 7,
    8: 12,
    9: 10,
    10: 21,
    11: 1,
    12: 137,
}

# rgb
carla_rgb_to_classid = {
    (0, 0, 0): 0,
    (70, 70, 70): 1,
    (190, 153, 153): 2,
    (250, 170, 160): 3,
    (220, 20, 60): 4,
    (153, 153, 153): 5,
    (157, 234, 50): 6,
    (128, 64, 128): 7,
    (244, 35, 232): 8,
    (107, 142, 35): 9,
    (0, 0, 142): 10,
    (102, 102, 156): 11,
    (220, 220, 0): 12,
}

#carla_classid_to_rgb = {v:k for k, v in carla_rgb_to_classid.iteritems()}
carla_classid_to_rgb = {v:k for k, v in carla_rgb_to_classid.items()}

def rgb_to_carla(color, color_mapping):
  # color is uint8
  img_h, img_w, _ = color.shape
  scene_seg = np.zeros((img_h, img_w), dtype="uint8")
  for i in range(img_h):
    for j in range(img_w):
      r, g, b = color[i, j]
      key = (r, g, b)
      #if not color_mapping.has_key(key):
      if key not in color_mapping:
        scene_seg[i, j] = 0  # other
      else:
        scene_seg[i, j] = color_mapping[key]
  return scene_seg


if __name__ == "__main__":
  args = parser.parse_args()

  traj_files = glob(os.path.join(args.traj_path, "*", "*.txt"))

  multifuture_frame_start = {
      "virat": 40,  # range(40, 125, 12)
      "ethucy": 32,  # range(32, 103, 10)
  }

  # we permute the carla_rgb to scene class mapping
  # due to mp4 compression
  # ugly but it works
  color_mapping = {}
  for r, g, b in carla_rgb_to_classid:
    for i in range(-4, 5):
      for j in range(-4, 5):
        for k in range(-4, 5):
          if ((r+i >= 0) and (r+i <= 255)) and ((g+j >= 0) and (g+j <= 255)) \
              and ((b+k >= 0) and (b+k <= 255)):
            color_mapping[(r+i, g+j, b+k)] = carla_rgb_to_classid[(r, g, b)]

  bad_video_lst = []
  for traj_file in tqdm(traj_files):
    split = traj_file.split("/")[-2]
    videoname = os.path.splitext(os.path.basename(traj_file))[0]

    frame_path = os.path.join(args.out_frame_path, videoname)
    seg_path = os.path.join(args.out_seg_feat_path, videoname)

    mkdir(frame_path)
    mkdir(seg_path)

    # 1. get video frames
    if args.is_multifuture:
      scene, moment_idx, x_agent_pid, camera = videoname.split("_")
      if scene.startswith("0"):
        start = multifuture_frame_start["virat"]
      else:
        start = multifuture_frame_start["ethucy"]
      video_file = glob(os.path.join(
          args.rgb_video_path,
          "%s_%s_%s_*_%s.mp4" % (scene, moment_idx, x_agent_pid, camera)))[0]
    else:
      start = 0
      video_file = os.path.join(args.rgb_video_path, "%s.mp4" % videoname)

    all_frame_idxs = get_frame_idxs(traj_file, start=start)

    try:
      vcap = cv2.VideoCapture(video_file)
      if not vcap.isOpened():
        raise Exception("cannot open %s" % video_file)
    except Exception as e:
      raise e
    frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)

    cur_rgb_frame = 0
    got_rgb_frame = 0
    while cur_rgb_frame < frame_count:
      suc, frame = vcap.read()
      assert suc, (videoname, cur_rgb_frame, frame_count)

      if cur_rgb_frame not in all_frame_idxs:
        cur_rgb_frame += 1
        continue

      frame = frame.astype("float32")
      cv2.imwrite(os.path.join(
          frame_path, "%s_F_%08d.jpg" % (videoname, cur_rgb_frame - start)),
                  frame)
      got_rgb_frame += 1
      cur_rgb_frame += 1

    # 2. get the scene seg feature into the ade20k format and save as npy
    if args.is_multifuture:
      scene, moment_idx, x_agent_pid, camera = videoname.split("_")
      video_file = glob(os.path.join(
          args.seg_video_path,
          "%s_%s_%s_*_%s.mp4" % (scene, moment_idx, x_agent_pid, camera)))[0]
    else:
      video_file = os.path.join(args.seg_video_path, "%s.mp4" % videoname)
    try:
      vcap = cv2.VideoCapture(video_file)
      if not vcap.isOpened():
        raise Exception("cannot open %s" % video_file)
    except Exception as e:
      raise e
    frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)

    cur_seg_frame = 0
    got_seg_frame = 0
    while cur_seg_frame < frame_count:
      suc, frame = vcap.read()
      assert suc, videoname

      if cur_seg_frame not in all_frame_idxs:
        cur_seg_frame += 1
        continue

      # resize first
      frame = cv2.resize(frame, (args.scene_w, args.scene_h),
                         interpolation=cv2.INTER_NEAREST)  # this keep the rgb
      # rgb to the scene class
      # (36, 64, 3) -> (36, 64)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      carla_scene_seg = rgb_to_carla(frame, color_mapping)

      if args.is_debug:
        # for debugging, save the class back to rgb
        carla_scene_seg_rgb = np.zeros((args.scene_h, args.scene_w, 3),
                                       dtype=np.uint8)
        for i in range(args.scene_h):
          for j in range(args.scene_w):
            # rgb to bgr
            r, g, b = carla_classid_to_rgb[carla_scene_seg[i, j]]
            carla_scene_seg_rgb[i, j, :] = [b, g, r]

        cv2.imwrite(os.path.join(
            seg_path, "%s_F_%08d.jpg" % (videoname, cur_seg_frame - start)),
                    carla_scene_seg_rgb)

      # convert carla class to ade20k class
      for i in range(args.scene_h):
        for j in range(args.scene_w):
          carla_scene_seg[i, j] = carla_to_ade20k[carla_scene_seg[i, j]]
      np.save(os.path.join(
          seg_path, "%s_F_%08d.npy" % (videoname, cur_seg_frame - start)),
              carla_scene_seg)

      got_seg_frame += 1
      cur_seg_frame += 1

    if (got_seg_frame != got_rgb_frame) or \
        (got_rgb_frame != len(all_frame_idxs)):
      print("warning, %s video has %s rgb frame, %s seg frames, %s in traj" % (
          videoname, got_rgb_frame, got_seg_frame, len(all_frame_idxs)))
      bad_video_lst.append((split, videoname))

    if args.is_debug:
      sys.exit()
  # save the bad video lst to delete
  print("total video %s, %s bad (%.4f)" % (
      len(traj_files), len(bad_video_lst),
      len(bad_video_lst)/float(len(traj_files))))
  with open(args.bad_video_lst, "w") as f:
    for split, videoname in bad_video_lst:
      f.writelines("%s/%s\n" % (split, videoname))
