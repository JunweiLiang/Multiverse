# coding=utf-8
# visualize converted annotation of SDD
import argparse
#import cPickle as pickle
import pickle
import cv2
import os
from tqdm import tqdm
from glob import glob

from get_prepared_data_sdd import class2classid

parser = argparse.ArgumentParser()
parser.add_argument("preparepath")
parser.add_argument("framepath")
parser.add_argument("targetpath")
parser.add_argument("--for_argoverse", action="store_true")
parser.add_argument("--vis_num_frame_per_video", default=3, type=int)

if __name__ == "__main__":
  args = parser.parse_args()

  splits = ["train", "val", "test"]
  if args.for_argoverse:
    splits = ["test"]
    from get_prepared_data_argoverse import class2classid

  classid2class = {v:k for k, v in class2classid.items()}

  traj_path = os.path.join(args.preparepath, "traj_2.5fps")
  person_box_path = os.path.join(args.preparepath, "anno_person_box")
  other_box_path = os.path.join(args.preparepath, "anno_other_box")

  for split in splits:
    traj_files = glob(os.path.join(traj_path, split, "*.txt"))

    for traj_file in tqdm(traj_files):
      video_id = os.path.splitext(os.path.basename(traj_file))[0]
      person_box_file = os.path.join(person_box_path, split, "%s.p" % video_id)
      other_box_file = os.path.join(other_box_path, split, "%s.p" % video_id)

      targetpath = os.path.join(args.targetpath, video_id)
      if not os.path.exists(targetpath):
        os.makedirs(targetpath)

      with open(person_box_file, "rb") as f:
        person_box_data = pickle.load(f)
      with open(other_box_file, "rb") as f:
        other_box_data = pickle.load(f)

      vis_frames = {}
      for i, line in enumerate(open(traj_file).readlines()):

        frame_idx, track_id, x, y = line.strip().split("\t")
        frame_idx, track_id = int(frame_idx), float(track_id)
        x, y = float(x), float(y)

        if frame_idx in vis_frames:
          continue
        vis_frames[frame_idx] = 1
        if len(vis_frames) > args.vis_num_frame_per_video:
          break

        person_key = "%s_%d_%d" % (video_id, frame_idx, track_id)

        person_box = person_box_data[person_key]
        other_boxes, other_boxclassids = other_box_data[person_key]

        frame_file = os.path.join(
            args.framepath, video_id, "%s_F_%08d.jpg" % (video_id, frame_idx))

        frame_data = cv2.imread(frame_file, cv2.IMREAD_COLOR)

        # the current person point
        frame_data = cv2.circle(
            frame_data, (int(x), int(y)), radius=5, color=(0, 0, 255))
        # the current person bounding box
        frame_data = cv2.rectangle(frame_data,
                                   (int(person_box[0]), int(person_box[1])),
                                   (int(person_box[2]), int(person_box[3])),
                                   color=(0, 0, 255),
                                   thickness=2)
        # the other boxes
        for box, box_classid in zip(other_boxes, other_boxclassids):
          box_label = classid2class[box_classid]

          frame_data = cv2.putText(frame_data, box_label,
                                   (int(box[0]), int(box[1])),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   color=(255, 0, 0))
          frame_data = cv2.rectangle(
              frame_data, (int(box[0]), int(box[1])),
              (int(box[2]), int(box[3])),
              color=(255, 0, 0), thickness=2)

        target_file = os.path.join(
            targetpath, "%s_F_%08d.jpg" % (video_id, frame_idx))
        cv2.imwrite(target_file, frame_data)

