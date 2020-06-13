# coding=utf-8
"""Given the person trajectory, get the vehicle points the """
import argparse
import os
import operator
#import cPickle as pickle
import pickle
import yaml
import numpy as np
from glob import glob
from tqdm import tqdm
from visualize_real_data import load_traj
from combine_traj import get_world_coordinates

parser = argparse.ArgumentParser()
parser.add_argument("traj_path", help="path to pedestrian dataset")
parser.add_argument("anno_path", help="yaml path")
parser.add_argument("h_path", help="path to homography matrix")
parser.add_argument("out_path")

# For running parallel jobs, set --job 4 --curJob k, where k=1/2/3/4
parser.add_argument("--job", type=int, default=1, help="total job")
parser.add_argument("--curJob", type=int, default=1,
                    help="this script run job Num")

actev_scene2imgsize = {
    "0002": (1280.0, 720.0),
    "0000": (1920.0, 1080.0),
    "0400": (1920.0, 1080.0),
    "0401": (1920.0, 1080.0),
    "0500": (1920.0, 1080.0),
}

scene2imgsize = actev_scene2imgsize


def load_yml_file_without_meta(yml_file):
  """Load the ActEV YAML annotation files."""
  with open(yml_file, "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    # get the meta index first
    mi = -1
    for i in range(len(data)):
      if "meta" not in data[i]:
        mi = i
        break
    assert mi >= 0

    return data[mi:]


def load_tracks(track_file, only=None):
  """Load track object type information."""
  trackid2object_ = {}
  data = load_yml_file_without_meta(track_file)
  for one in data:
    one = one["types"]  # added in v1_update
    # v4, changed to - { types: { id1: 0 , cset3: { Person: 1.0 } } }
    if "obj_type" not in one:
      one["obj_type"] = list(one["cset3"].keys())[0]
      assert len(one["cset3"].keys()) == 1
    if only is not None:
      if one["obj_type"] != only:
        continue
    trackid2object_[int(one["id1"])] = one["obj_type"]
  return trackid2object_


def load_boxes(box_file_, imgsize_):
  """Load bounding boxes."""
  boxes_ = []
  data = load_yml_file_without_meta(box_file_)
  for one in data:
    one = one["geom"]  # added in v1_update
    trackid_ = int(one["id1"])
    frame_index_ = int(one["ts0"])

    bbox_ = [float(a) for a in one["g0"].split()]

    src = one["src"]
    assert src == "truth", (src, one)

    # check box valid
    is_valid = valid_box(bbox_, imgsize_)
    if not is_valid:
      # modify box to be valid?
      bbox_ = modify_box(bbox_, imgsize_)
      assert valid_box(bbox_, imgsize_)
    # so box is [x1, y1, x2, y2]
    boxes_.append((trackid_, frame_index_, bbox_))

  return boxes_


def get_scene(videoname):
  """ActEV scene extractor from videoname."""
  s = videoname.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]


# actev boxes may contain some errors
# won't fix x,y reversed
def modify_box(bbox, imgsize):
  """Modify ActEV boxes."""
  w, h = imgsize
  x1, y1, x2, y2 = bbox
  x_min = min(x1, x2)
  x_max = max(x1, x2)
  y_min = min(y1, y2)
  y_max = max(y1, y2)

  x_min = min(w, x_min)
  x_max = min(w, x_max)
  y_min = min(h, y_min)
  y_max = min(h, y_max)

  return [x_min, y_min, x_max, y_max]


def valid_box(box, wh):
  """Check whether boxes are within the image bounds."""
  w, h = wh
  a = box_area(box)
  if a <= 0:
    return False
  if (box[0] > w) or (box[2] > w) or (box[1] > h) or (box[3] > h):
    return False
  return True


def box_area(box):
  """compute bbox area size in pixels."""
  x1, y1, x2, y2 = box
  w = x2 - x1
  h = y2 - y1
  return float(w) * h


def get_box_center(box):
  x1, y1, x2, y2 = box
  return [(x1+x2)/2.0, (y1+y2)/2.0]


def save_file(data, out_path, videoname):
  target_file = os.path.join(out_path, "%s.txt" % videoname)
  with open(target_file, "w") as f:
    for one in data:
      f.writelines("%s\n" % "\t".join(["%s" % x for x in one]))

if __name__ == "__main__":
  args = parser.parse_args()

  traj_files = glob(os.path.join(args.traj_path, "*.txt"))
  traj_files.sort()

  out_path_pixel = os.path.join(args.out_path, "pixel")
  out_path_world = os.path.join(args.out_path, "world")
  if not os.path.exists(out_path_pixel):
    os.makedirs(out_path_pixel)
    os.makedirs(out_path_world)

  # load the homography matrices
  h_dict = {}
  h_files = glob(os.path.join(args.h_path, "*.txt"))
  for h_file in h_files:
    scene = os.path.splitext(os.path.basename(h_file))[0]
    h_matrix = []
    with open(h_file, "r") as f:
      for line in f:
        h_matrix.append(line.strip().split(","))
    h_matrix = np.array(h_matrix, dtype="float")

    h_dict[scene] = h_matrix

  count = 0
  for traj_file in tqdm(traj_files):
    count += 1
    if (count % args.job) != (args.curJob - 1):
      continue

    # 1. load pedestrian dataset, so we know what frames to look at
    ped_trajs = load_traj(traj_file)
    frame_ids = np.unique(ped_trajs[:, 0]).tolist()
    frame_ids.sort()
    frame_ids = {int(frame_id):1 for frame_id in frame_ids}

    videoname = os.path.splitext(os.path.basename(traj_file))[0]
    scene = get_scene(videoname)
    h_matrix = h_dict[scene]
    imgsize = scene2imgsize[scene]

    box_file = os.path.join(args.anno_path, videoname + ".geom.yml")
    type_file = os.path.join(args.anno_path, videoname + ".types.yml")
    act_file = os.path.join(args.anno_path, videoname + ".activities.yml")

    # load each track id and its trajectories
    trackid2object = load_tracks(type_file, "Vehicle")
    # no cars in this video?

    # load traj boxes for the trackid
    boxes = load_boxes(box_file, imgsize)

    data_pixel = []  # frame_id, track_id, x, y
    data_world = []
    for box in boxes:
      track_id, frame_index, bbox = box
      #if trackid2object.has_key(track_id) and frame_ids.has_key(frame_index):
      if track_id in trackid2object and frame_index in frame_ids:
        center_xy = get_box_center(bbox)
        world_xy = get_world_coordinates(center_xy, h_matrix)
        if scene == "0002":
          # 0002 scene is origin 1280x720, image is resized to 1920x1080
          x, y = center_xy
          center_xy = [x * (1920 / 1280.0), y * (1920 / 1280.0)]
        world_xy = [-world_xy[0], world_xy[1]]  # actev h is mirrored
        data_pixel.append([frame_index, track_id] + center_xy)
        data_world.append([frame_index, track_id] + world_xy)

    data_pixel.sort(key=operator.itemgetter(0))
    data_world.sort(key=operator.itemgetter(0))
    save_file(data_pixel, out_path_pixel, videoname)
    save_file(data_world, out_path_world, videoname)

