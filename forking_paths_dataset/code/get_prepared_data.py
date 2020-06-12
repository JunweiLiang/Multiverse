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
parser.add_argument("dataset_path")
parser.add_argument("split_path")
parser.add_argument("outpath")

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def convert_bbox(bbox):
  x, y, w, h = bbox
  return [x, y, x + w, y + h]

def get_feet(bbox):
  x1, y1, x2, y2 = bbox
  return ((x1 + x2) / 2.0, y2)

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

if __name__ == "__main__":
  args = parser.parse_args()

  args.drop_frame = {
      "virat": 12,
      "ethucy": 10,
  }

  class2classid = {
      "Person": 0,
      "Vehicle": 1,
  }

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

  for split in tqdm(filelst, ascii=True):
    traj_path = os.path.join(args.traj_path, split)
    mkdir(traj_path)
    person_box_path = os.path.join(args.person_box_path, split)
    mkdir(person_box_path)
    other_box_path = os.path.join(args.other_box_path, split)
    mkdir(other_box_path)

    for videoname in tqdm(filelst[split]):
      bbox_json = os.path.join(args.dataset_path, "bbox", "%s.json" % videoname)
      with open(bbox_json, "r") as f:
        o_bboxes = json.load(f)
      bboxes = filter_neg_boxes(o_bboxes)
      if len(o_bboxes) != len(bboxes):
        print("warning, filter out negative boxes left %s/%s" % (
            len(bboxes), len(o_bboxes)))
      drop_frame = args.drop_frame["virat"]

      # 1. first pass, get the needed frames
      frame_data = {}  # frame_idx -> data
      for one in bboxes:
        if one["frame_id"] not in frame_data:
          frame_data[one["frame_id"]] = []
        frame_data[one["frame_id"]].append(one)
      frame_idxs = sorted(frame_data.keys())
      #assert frame_idxs[0] == 0
      needed_frame_idxs = frame_idxs[::drop_frame]
      if len(needed_frame_idxs) < 8 + 12:
        print("warning, %s video has only %s frames, skipped.." % (
            videoname, len(frame_idxs)))
        continue

      # 2. gather data for each frame_idx, each person_idx
      traj_data = []  # [frame_idx, person_idx, x, y]
      person_box_data = {}  # (frame_idx, person_id) -> boxes
      other_box_data = {}  # (frame_idx, person_id) -> other boxes + boxclasids
      for frame_idx in needed_frame_idxs:
        box_list = frame_data[frame_idx]
        # filter out negative boxes
        box_list.sort(key=operator.itemgetter("track_id"))
        for i, box in enumerate(box_list):
          class_name = box["class_name"]
          track_id = box["track_id"]
          is_x_agent = box["is_x_agent"]
          bbox = convert_bbox(box["bbox"])
          if class_name == "Person":
            person_key = "%d_%d" % (frame_idx, track_id)

            x, y = get_feet(bbox)
            traj_data.append((frame_idx, float(track_id), x, y))

            person_box_data[person_key] = bbox

            all_other_boxes = [convert_bbox(box_list[j]["bbox"])
                               for j in range(len(box_list)) if j != i]
            all_other_boxclassids = [class2classid[box_list[j]["class_name"]]
                                     for j in range(len(box_list)) if j != i]

            other_box_data[person_key] = (all_other_boxes,
                                          all_other_boxclassids)

      frame_counts.append(len(needed_frame_idxs))

      # save the data
      desfile = os.path.join(traj_path, "%s.txt" % videoname)

      delim = "\t"

      with open(desfile, "w") as f:
        for i, p, x, y in traj_data:
          f.writelines("%d%s%.1f%s%.6f%s%.6f\n" % (i, delim, p, delim, x,
                                                   delim, y))

      with open(os.path.join(person_box_path,
                             "%s.p" % videoname), "wb") as f:
        pickle.dump(person_box_data, f)

      with open(os.path.join(other_box_path,
                             "%s.p" % videoname), "wb") as f:
        pickle.dump(other_box_data, f)
  print("total file %s, min/max/avg frame count %s/%s/%s" % (
      len(frame_counts),
      min(frame_counts),
      max(frame_counts),
      np.mean(frame_counts)))
