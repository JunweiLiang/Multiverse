# coding=utf-8
"""Multi-future visualization of the full dataset."""

import argparse
import cv2
import json
import os
import operator
import pickle
from glob import glob
from tqdm import tqdm

import sys
if sys.version_info > (3, 0):
  import subprocess as commands
else:
  import commands

parser = argparse.ArgumentParser()
parser.add_argument("video_path")
parser.add_argument("bbox_path")
parser.add_argument("gt_path", help="multifuture traj gt path.")
parser.add_argument("out_video_path")


def check_box(box):
  x, y, w, h = [int(round(o)) for o in box]
  x1, y1, x2, y2 = x, y, x+w, y+h
  if (x1 < 0) or (y1 < 0) or (x2 < 0) or (y2 < 0):
    return None
  if (x1 >= 1920) or (x2 >= 1920):
    return None
  if (y1 >= 1080) or (y2 >= 1080):
    return None
  return [x, y, w, h]


def get_obs_videonames(filelst):
  obs_videonames = {}
  for videoname in filelst:
    scene, moment_idx, x_agent_pid, dest_idx, annotator_id, camera = \
          videoname.split("_")
    obs_key = "_".join([scene, moment_idx, x_agent_pid, camera])
    #if not obs_videonames.has_key(obs_key):
    if obs_key not in obs_videonames:
      obs_videonames[obs_key] = []
    obs_videonames[obs_key].append(videoname)
  return obs_videonames

if __name__ == "__main__":
  args = parser.parse_args()

  pred_frame_start = {
      "virat": 125,
      "ethucy": 103,
  }
  video_fpss = {
      "virat": 30.0,
      "ethucy": 25.0,
  }

  all_videos = glob(os.path.join(args.video_path, "*.mp4"))
  all_videonames = [os.path.splitext(os.path.basename(v))[0]
                    for v in all_videos]
  # 1. get all the traj_id and the corresponding future videos
  traj_ids_to_videonames = get_obs_videonames(all_videonames)

  exists_traj_ids_to_videonames = []
  for traj_id in traj_ids_to_videonames:
    if os.path.exists(os.path.join(args.gt_path, "%s.p" % traj_id)):
      exists_traj_ids_to_videonames.append(traj_id)
    else:
      print("warning, ignoring %s.." % traj_id)

  traj_ids_to_videonames = {traj_id:traj_ids_to_videonames[traj_id]
                            for traj_id in traj_ids_to_videonames
                            if traj_id in exists_traj_ids_to_videonames}

  for traj_id in tqdm(traj_ids_to_videonames.keys()):
    target_path = os.path.join(args.out_video_path, "%s" % traj_id)
    if not os.path.exists(target_path):
      os.makedirs(target_path)

    # 1. get the longest pred video
    with open(os.path.join(args.gt_path, "%s.p" % traj_id), "rb") as f:
      gt = pickle.load(f)
    pred_lengths = [(len(gt[vn]["x_agent_traj"]), vn) for vn in gt]
    pred_lengths.sort(key=operator.itemgetter(0), reverse=True)

    base_videoname = pred_lengths[0][1]
    base_video = os.path.join(args.video_path, "%s.mp4" % base_videoname)

    if base_videoname.startswith("0"):
      frame_start = pred_frame_start["virat"]
      video_fps = video_fpss["virat"]
    else:
      frame_start = pred_frame_start["ethucy"]
      video_fps = video_fpss["ethucy"]

    # 2. gather all the bounding boxes
    future_bboxes = {}  # frame_idx -> annotator_id -> box
    for videoname in traj_ids_to_videonames[traj_id]:
      scene, moment_idx, x_agent_pid, dest_idx, \
          annotator_id, camera = videoname.split("_")
      with open(os.path.join(args.bbox_path, "%s.json" % videoname)) as f:
        bboxes = json.load(f)
      for box in bboxes:
        if (box["is_x_agent"] == 1) and (box["track_id"] == int(x_agent_pid)):
          frame_idx = box["frame_id"]
          #if not future_bboxes.has_key(frame_idx):
          if frame_idx not in future_bboxes:
            future_bboxes[frame_idx] = {}

          traj_key = (annotator_id, dest_idx, x_agent_pid)

          assert traj_key not in future_bboxes[frame_idx], \
              (annotator_id, frame_idx, videoname)
          # x,y,w,h
          future_bboxes[frame_idx][traj_key] = (box["bbox"], videoname)


    # 2. make the video frame by frame


    # open all the videos
    vcaps = []
    base_vcap = None
    frame_count = None
    for videoname in traj_ids_to_videonames[traj_id]:
      videofile = os.path.join(args.video_path, "%s.mp4" % videoname)
      vcap = cv2.VideoCapture(videofile)
      if not vcap.isOpened():
        raise Exception("Cannot open %s" % videofile)
      if videoname == base_videoname:
        frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
        base_vcap = vcap
      else:
        vcaps.append([videoname, vcap])

    assert frame_count is not None
    assert base_vcap is not None

    cur_frame = 0
    while cur_frame < frame_count:
      _, frame = base_vcap.read()
      im = frame.astype("float32")
      other_frames = {videoname: vcap.read()[1]
                      for videoname, vcap in vcaps}

      # overlay all other future box img of the x agent
      if cur_frame >= frame_start:
        # gather all the box image and blend to the crop on the base image
        for traj_key in future_bboxes[cur_frame]:
          annotator_id, dest_idx, x_agent_pid = traj_key
          box, videoname = future_bboxes[cur_frame][traj_key]
          if videoname == base_videoname:
            continue

          box = check_box(box)
          if box is None:
            print("warning, skipping %s at frame %s due to bad box: %s" % (
                videoname, cur_frame, future_bboxes[cur_frame][traj_key][0]))
            continue

          this_frame = other_frames[videoname]
          if this_frame is None:
            print("warning, %s frame %s is none but have box" % (
                videoname, cur_frame))
            continue
          this_frame = this_frame.astype("float32")


          x, y, w, h = box
          box_img = this_frame[y:(y+h), x:(x+w)]
          overlay = cv2.addWeighted(im[y:(y+h), x:(x+w)], 0.5, box_img, 0.5, 0)

          im[y:(y+h), x:(x+w)] = overlay

      else:
        # draw a bounding box for the x agent
        scene, moment_idx, x_agent_pid, dest_idx, \
          annotator_id, camera = base_videoname.split("_")
        traj_key = (annotator_id, dest_idx, x_agent_pid)
        box, videoname = future_bboxes[cur_frame][traj_key]
        box = check_box(box)
        if box is None:
          print("warning, skipping %s at frame %s due to bad box: %s" % (
              videoname, cur_frame, future_bboxes[cur_frame][traj_key][0]))
        else:
          x, y, w, h = box
          im = cv2.rectangle(im, (x, y), (x + w, y + h), color=(0, 0, 255),
                             thickness=2)

      target_frame = os.path.join(target_path, "%08d.jpg" % (cur_frame))
      cv2.imwrite(target_frame, im)

      cur_frame += 1
    # release all resources
    base_vcap.release()
    for _, vcap in vcaps:
      vcap.release()
    # make the video
    target_file = os.path.join(args.out_video_path, "%s.mp4" % traj_id)
    output = commands.getoutput("ffmpeg -y -framerate %s -i %s/%%08d.jpg %s" % (
      video_fps, target_path, target_file))
    output = commands.getoutput("rm -rf %s" % target_path)
