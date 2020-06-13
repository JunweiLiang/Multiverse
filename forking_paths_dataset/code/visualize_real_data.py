# coding=utf-8
"""Given a start frame, visualize the pixel trajectory and world."""

import argparse
import os
import cv2
import math

import numpy as np
from combine_traj import get_world_coordinates

parser = argparse.ArgumentParser()
parser.add_argument("video_frame_path")
parser.add_argument("start_frame_idx", type=int)
parser.add_argument("traj_pixel_file")
parser.add_argument("traj_world_file")
parser.add_argument("vis_file")
parser.add_argument("--h_file", default=None, help="if set, will recompute the"
                                                    "traj using the h file")
parser.add_argument("--world_rotate", default=0.0, type=float,
                    help="rotate the points by degree for world vis.")

parser.add_argument("--obs_length", type=int, default=8)
parser.add_argument("--pred_length", type=int, default=12)

parser.add_argument("--vis_vehicle", action="store_true")
parser.add_argument("--vehicle_traj_pixel_file", default=None)
parser.add_argument("--vehicle_traj_world_file", default=None)


def load_traj(traj_file):
  data = [line.strip().split("\t") for line in open(traj_file).readlines()]
  return np.array(data, dtype="float32")  # [frame_id, person_id, x, y]

def get_traj(traj_data, frame_ids, person_id):
  """Select the traj matching the frame_ids and person_id."""
  person_data = traj_data[traj_data[:, 1] == person_id, :]
  filtered_data = person_data[np.isin(person_data[:, 0], frame_ids), :]
  return (person_id, filtered_data[:, 2:].tolist())

def get_scene(videoname_):
  """ActEV scene extractor from videoname."""
  s = videoname_.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]

def plot_trajs(img, trajs, color, show_person_id=False):
  """Plot traj on img with the person_id text."""
  # color is bgr tuple
  for person_id_, traj in trajs:
    traj = [(int(p1), int(p2)) for p1, p2 in traj]
    points = zip(traj[:-1], traj[1:])
    for p1, p2 in points:
      img = cv2.arrowedLine(img, tuple(p1), tuple(p2), color=color, thickness=2,
                            line_type=cv2.LINE_AA, tipLength=0.3)
    # show the person_id
    if show_person_id:
      img = cv2.putText(img, "#%s" % person_id_, tuple(traj[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        lineType=cv2.LINE_AA)
  return img


def rotate(traj_data, origin, radian):
  """Rotate the points. Simple trial-and-see affine transformation to get the
  world plane view similar to the video view
  """
  ox, oy = origin
  for i in range(len(traj_data)):
    px, py = traj_data[i, 2:]

    qx = ox + math.cos(radian) * (px - ox) - math.sin(radian) * (py - oy)
    qy = oy + math.sin(radian) * (px - ox) + math.cos(radian) * (py - oy)

    traj_data[i, 2:] = [qx, qy]
  return traj_data


if __name__ == "__main__":
  args = parser.parse_args()
  assert args.video_frame_path.endswith("/")


  # get the full trajectory that starts from this frame first
  traj_pixel_data = load_traj(args.traj_pixel_file)
  traj_world_data = load_traj(args.traj_world_file)

  if args.vis_vehicle:
    vehicle_traj_pixel = load_traj(args.vehicle_traj_pixel_file)
    vehicle_traj_world = load_traj(args.vehicle_traj_world_file)

  if args.h_file is not None:
    h_matrix = []
    with open(args.h_file, "r") as f:
      for line in f:
        h_matrix.append(line.strip().split(","))
    h_matrix = np.array(h_matrix, dtype="float")
    # recompute the world coordinates
    traj_world_data = [[fidx, pidx] + get_world_coordinates((x, y), h_matrix)
                       for fidx, pidx, x, y in traj_pixel_data]
    # actev h_matrix gets mirrored
    traj_world_data = [[fidx, pidx, -x, y]
                       for fidx, pidx, x, y in traj_world_data]
    traj_world_data = np.array(traj_world_data, dtype="float32")

  frame_ids = np.unique(traj_pixel_data[:, 0]).tolist()
  frame_ids.sort()
  f_idx = frame_ids.index(args.start_frame_idx)
  # we will draw pred_frame first then obs frame traj so there is 2 color?
  obs_frame_ids = frame_ids[f_idx:f_idx + args.obs_length]
  full_frame_ids = frame_ids[f_idx:f_idx + args.obs_length + args.pred_length]

  # all the person in the start idx
  person_ids = np.unique(
      traj_pixel_data[traj_pixel_data[:, 0] == args.start_frame_idx, 1])
  person_ids = person_ids.tolist()

  # (person_id, list of xy)
  obs_person_trajs = [get_traj(
      traj_pixel_data, obs_frame_ids, person_id) for person_id in person_ids]
  full_person_trajs = [get_traj(
      traj_pixel_data, full_frame_ids, person_id) for person_id in person_ids]

  # 1. pixel trajectory visualize on video frames.
  videoname = args.video_frame_path.split("/")[-2]
  video_frame_file = os.path.join(
      args.video_frame_path, "%s_F_%08d.jpg" % (videoname,
                                                args.start_frame_idx))
  print("%s start at %s, total %s people." % (
      videoname, args.start_frame_idx, len(person_ids)))
  video_frame = cv2.imread(video_frame_file, cv2.IMREAD_COLOR)
  h, w = video_frame.shape[:2]
  vis_pixel = plot_trajs(video_frame, full_person_trajs, (0, 255, 0),
                         show_person_id=True)
  vis_pixel = plot_trajs(vis_pixel, obs_person_trajs, (0, 255, 255))
  # plot the frame number
  vis_pixel = cv2.putText(vis_pixel, "#%s" % args.start_frame_idx, (0, h - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                          lineType=cv2.LINE_AA)

  # visualize from ground plane
  ground_image = np.zeros((h, w, 3), dtype="uint8")

  if args.world_rotate != 0:
    traj_world_data = rotate(
        traj_world_data, (0, 0), math.radians(args.world_rotate))
    if args.vis_vehicle:
      vehicle_traj_world = rotate(
          vehicle_traj_world, (0, 0), math.radians(args.world_rotate))

  # normalize the world coordinates to plot on h, w
  # TODO: in ActEV, the norm should be all videos in the scene
  min_x = np.amin(np.array(traj_world_data)[:, 2])
  max_x = np.amax(np.array(traj_world_data)[:, 2])
  min_y = np.amin(np.array(traj_world_data)[:, 3])
  max_y = np.amax(np.array(traj_world_data)[:, 3])
  length_x = max_x - min_x
  length_y = max_y - min_y
  traj_world_normalized = traj_world_data.copy()
  traj_world_normalized[:, 2] = w * (
      traj_world_normalized[:, 2] - min_x) / length_x
  traj_world_normalized[:, 3] = h * (
      traj_world_normalized[:, 3] - min_y) / length_y

  obs_person_trajs = [get_traj(
      traj_world_normalized, obs_frame_ids, person_id)
                      for person_id in person_ids]
  full_person_trajs = [get_traj(
      traj_world_normalized, full_frame_ids, person_id)
                       for person_id in person_ids]


  vis_world = plot_trajs(ground_image, full_person_trajs, (0, 255, 0),
                         show_person_id=True)
  vis_world = plot_trajs(vis_world, obs_person_trajs, (0, 255, 255))

  # plot the frame number
  vis_world = cv2.putText(vis_world, "#%s" % args.start_frame_idx, (0, h - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                          lineType=cv2.LINE_AA)
  # vehicle visualize
  if args.vis_vehicle:
    vehicle_ids = np.unique(
        vehicle_traj_pixel[vehicle_traj_pixel[:, 0] == args.start_frame_idx, 1])
    vehicle_ids = vehicle_ids.tolist()
    full_vehicle_trajs = [get_traj(
        vehicle_traj_pixel, full_frame_ids, vehicle_id)
                          for vehicle_id in vehicle_ids]
    vis_pixel = plot_trajs(vis_pixel, full_vehicle_trajs, (255, 0, 0))

    # ground plane
    # 1. normalizing
    vehicle_traj_world_norm = vehicle_traj_world.copy()
    vehicle_traj_world_norm[:, 2] = w * (
        vehicle_traj_world_norm[:, 2] - min_x) / length_x
    vehicle_traj_world_norm[:, 3] = h * (
        vehicle_traj_world_norm[:, 3] - min_y) / length_y
    full_vehicle_trajs = [get_traj(
        vehicle_traj_world_norm, full_frame_ids, vehicle_id)
                          for vehicle_id in vehicle_ids]
    vis_world = plot_trajs(vis_world, full_vehicle_trajs, (255, 255, 0))


  #cv2.imwrite(args.vis_world, vis_world)
  # concat two images
  vis = np.concatenate((vis_pixel, vis_world), axis=1)
  cv2.imwrite(args.vis_file, vis)
