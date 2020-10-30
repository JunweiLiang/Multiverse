# coding=utf-8
"""Given the final dataset or the anchor dataset, compile prepared data."""

import argparse
import json
import os
import operator
import pickle
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("annotation_path")
parser.add_argument("split_path")
parser.add_argument("changelst", help="the resizing and rotation")
parser.add_argument("outpath")

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def filter_neg_boxes(bboxes):
  new_bboxes = []
  for bbox in bboxes:
    x, y, w, h = bbox["bbox"]
    coords = x, y, x + w, y + h
    bad = False
    for o in coords:
      if o < 0:
        bad = True
    if not bad:
      new_bboxes.append(bbox)
  return new_bboxes

# be consistent with next paper, merge some classes
class2classid = {
    "Pedestrian": 0,
    "Car": 1,
    "Bus": 1,
    "Cart": 1,
    "Biker": 8,
    "Skater": 8,
}


if __name__ == "__main__":
  args = parser.parse_args()

  args.drop_frame = {
      "sdd": 12,
  }

  target_resolution = (1920.0, 1080.0)



  filelst = {
      "train": [os.path.splitext(os.path.basename(line.strip()))[0]
                for line in open(os.path.join(args.split_path,
                                              "train.lst"), "r").readlines()],
      "val": [os.path.splitext(os.path.basename(line.strip()))[0]
              for line in open(os.path.join(args.split_path,
                                            "val.lst"), "r").readlines()],
      "test": [os.path.splitext(os.path.basename(line.strip()))[0]
               for line in open(os.path.join(args.split_path,
                                             "test.lst"), "r").readlines()],
  }

  args.traj_path = os.path.join(args.outpath, "traj_2.5fps")
  args.person_box_path = os.path.join(args.outpath, "anno_person_box")
  args.other_box_path = os.path.join(args.outpath, "anno_other_box")
  # we will have scene segmentation feature for each needed frame
  #args.scene_map_path = os.path.join(args.outpath, "anno_scene")

  frame_counts = []

  # load the changelst
  changelst = {}
  for line in open(args.changelst, "r").readlines():
    video_id, ori_resolution, rotated_90_clockwise = line.strip().split(",")
    rotated_90_clockwise = rotated_90_clockwise == "True"
    w, h = ori_resolution.split("x")
    w, h = int(w), int(h)
    if rotated_90_clockwise:
      w, h = h, w  # used for scaling later
    changelst[video_id] = (w, h, rotated_90_clockwise)

  def convert_bbox(bbox, vid):
    w, h, rotated_90_clockwise = changelst[vid]
    x1, y1, x2, y2 = bbox
    if rotated_90_clockwise:
      x1, y1, x2, y2 = y1, x1, y2, x2
      x1 = w - x1
      x2 = w - x2
    # rescaling
    x1 = target_resolution[0]/w * x1
    x2 = target_resolution[0]/w * x2
    y1 = target_resolution[1]/h * y1
    y2 = target_resolution[1]/h * y2

    return [x1, y1, x2, y2]

  def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2)/2.0, (y1 + y2)/2.0

  for split in tqdm(filelst, ascii=True):
    traj_path = os.path.join(args.traj_path, split)
    mkdir(traj_path)
    person_box_path = os.path.join(args.person_box_path, split)
    mkdir(person_box_path)
    other_box_path = os.path.join(args.other_box_path, split)
    mkdir(other_box_path)

    for video_id in tqdm(filelst[split]):
      scene, videoname = video_id.split("_")
      annotation_file = os.path.join(args.annotation_path, scene, videoname,
                                     "annotations.txt")
      anno_data = [line.strip().split()
                   for line in open(annotation_file).readlines()]

      drop_frame = args.drop_frame["sdd"]

      # 1. first pass, get the needed frames
      frame_idxs = {}
      for one in anno_data:
        # is a person and not outside of view
        if (one[-1].strip("\"") == "Pedestrian") and (one[-4] == "0"):
          frame_idxs[int(one[5])] = 1
      frame_idxs = sorted(frame_idxs.keys())
      needed_frame_idxs = frame_idxs[::drop_frame]
      if len(needed_frame_idxs) < 8 + 12:
        print("warning, %s video has only %s frames, skipped.." % (
            video_id, len(frame_idxs)))
        continue

      # save all the data into # frame_idx -> data
      frame_data = {}
      for one in anno_data:
        track_id, x1, y1, x2, y2, frame_idx, lost, _, _, classname = one
        track_id, x1, y1, x2, y2, frame_idx = [
            int(o) for o in [track_id, x1, y1, x2, y2, frame_idx]]
        if (frame_idx not in needed_frame_idxs) or (lost == "1"):
          continue
        if frame_idx not in frame_data:
          frame_data[frame_idx] = []
        frame_data[frame_idx].append({
            "class_name": classname.strip("\""),
            "track_id": track_id,
            "bbox": convert_bbox([x1, y1, x2, y2], video_id)
        })


      # 2. gather data for each frame_idx, each person_idx
      traj_data = []  # [frame_idx, person_idx, x, y]
      person_box_data = {}  # (frame_idx, person_id) -> boxes
      other_box_data = {}  # (frame_idx, person_id) -> other boxes + boxclasids
      for frame_idx in needed_frame_idxs:
        box_list = frame_data[frame_idx]
        box_list.sort(key=operator.itemgetter("track_id"))
        for i, box in enumerate(box_list):
          class_name = box["class_name"]
          track_id = box["track_id"]
          bbox = box["bbox"]
          if class_name == "Pedestrian":
            person_key = "%s_%d_%d" % (video_id, frame_idx, track_id)

            x, y = get_center(bbox)

            # ignore points outside of current resolution
            if (x > target_resolution[0]) or (y > target_resolution[1]):
              continue

            traj_data.append((frame_idx, float(track_id), x, y))

            person_box_data[person_key] = bbox

            all_other_boxes = [box_list[j]["bbox"]
                               for j in range(len(box_list)) if j != i]
            all_other_boxclassids = [class2classid[box_list[j]["class_name"]]
                                     for j in range(len(box_list)) if j != i]

            other_box_data[person_key] = (all_other_boxes,
                                          all_other_boxclassids)

      frame_counts.append(len(needed_frame_idxs))

      # save the data
      desfile = os.path.join(traj_path, "%s.txt" % video_id)

      delim = "\t"

      with open(desfile, "w") as f:
        for i, p, x, y in traj_data:
          f.writelines("%d%s%.1f%s%.6f%s%.6f\n" % (i, delim, p, delim, x,
                                                   delim, y))

      with open(os.path.join(person_box_path,
                             "%s.p" % video_id), "wb") as f:
        pickle.dump(person_box_data, f)

      with open(os.path.join(other_box_path,
                             "%s.p" % video_id), "wb") as f:
        pickle.dump(other_box_data, f)
  print("total file %s, min/max/avg frame count %s/%s/%s" % (
      len(frame_counts),
      min(frame_counts),
      max(frame_counts),
      np.mean(frame_counts)))
