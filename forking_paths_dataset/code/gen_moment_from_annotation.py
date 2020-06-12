# coding=utf-8
"""Given the annotation file and original moment, generate new moment data
  for recording
"""
import argparse
import json
import os
import operator

import numpy as np

from utils import reset_x_agent_key
from utils import interpolate_controls
from utils import make_moment_id

parser = argparse.ArgumentParser()
parser.add_argument("moment_filelst")
parser.add_argument("annotation_jsonlst", help="filepath annotator_id")
parser.add_argument("final_json")

# moment setup
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--annotation_fps", type=float, default=2.5)
parser.add_argument("--obs_length", type=int, default=12,
                    help="observation timestep, 12/2.5=4.8 seconds")
parser.add_argument("--pred_length", type=int, default=26,
                    help="observation timestep, 26/2.5=10.4 seconds")

if __name__ == "__main__":
  args = parser.parse_args()

  # compute the frame_ids for each moment
  args.frame_skip = int(args.video_fps / args.annotation_fps)  # 12/10
  # should be 0 -> 456 for 30fps
  args.moment_frame_ids = range(
      0, (args.obs_length + args.pred_length) * args.frame_skip,
      args.frame_skip)

  moment_data = []  # a stacked list as in annotate_carla
  for filename in open(args.moment_filelst, "r").readlines():
    with open(filename.strip(), "r") as f:
      this_moment_data = json.load(f)
      moment_data += this_moment_data
  reset_x_agent_key(moment_data)

  annotation = {}  # traj_key, id -> annotation
  for annotation_file in open(args.annotation_jsonlst, "r").readlines():
    annotation_file, annotator_id = annotation_file.strip().split()
    with open(annotation_file, "r") as f:
      this_annotation = json.load(f)
      # could have conflicts since we may want to merge
      for traj_key in this_annotation:
        #assert not annotation.has_key((traj_key, annotator_id)), \
        assert (traj_key, annotator_id) not in annotation, \
            "%s,%s already exists." % (traj_key, annotator_id)
        annotation[(traj_key, annotator_id)] = this_annotation[traj_key]

  # sort the annotation based on the moment_idx
  traj_keys = [(traj_key, annotator_id, int(traj_key.split("_")[1]))
               for traj_key, annotator_id in annotation]
  traj_keys.sort(key=operator.itemgetter(2))

  # stats of the annotated control length
  lengths = []
  num_moment = {}
  num_destinations = {}
  multi_future_count = {}  # moment_idx, x_agent_id -> num trajs

  new_moments = []
  for traj_key, annotator_id, _ in traj_keys:
    anno_key = (traj_key, annotator_id)
    # compile the annotation
    this_annotation = {}  # frame_id ->
    last_x_agent_frame_id = annotation[anno_key][-1][0]
    first_x_agent_frame_id = annotation[anno_key][0][0]

    lengths.append([
        first_x_agent_frame_id, last_x_agent_frame_id,
        len(annotation[anno_key])])
    for frame_id, direction_xyz, speed, location_xyz in annotation[anno_key]:
      this_annotation[frame_id] = (direction_xyz, speed, location_xyz)


    scene, moment_idx, x_agent_pid, dest_idx = traj_key.split("_")

    moment_idx, x_agent_pid, dest_idx = \
        int(moment_idx), int(x_agent_pid), int(dest_idx)

    # 1. get the original moment
    this_moment_data = moment_data[moment_idx].copy()
    # use the scene from the moment_data
    scene = this_moment_data["scenename"]

    num_moment[moment_idx] = 1
    x_agent_key = (moment_idx, x_agent_pid)
    num_destinations[x_agent_key] = len(this_moment_data["x_agents"])
    if x_agent_key not in multi_future_count:
      multi_future_count[x_agent_key] = 0
    multi_future_count[x_agent_key] += 1


    # 1. delete all controls of the x_agent after
    # 2. add the new control

    # reset the person_controls due to json
    person_controls = {}
    for frame_id in this_moment_data["ped_controls"]:
      person_controls[int(frame_id)] = \
          this_moment_data["ped_controls"][frame_id]
    # only save up to the last annotation frame
    new_person_controls = {}
    for frame_id in range(0, last_x_agent_frame_id + 1):

      if frame_id < first_x_agent_frame_id:
        # save everything
        if frame_id in person_controls:
          new_person_controls[frame_id] = person_controls[frame_id]
      else:
        # replace with x_agent annotation
        temp = []
        if frame_id in person_controls:
          for one in person_controls[frame_id]:
            if one[0] != x_agent_pid:
              temp.append(one)

        if frame_id in this_annotation:
          direction_xyz, speed, location_xyz = this_annotation[frame_id]
          """  # still keeps all the control so we could interpolate later
          if frame_id == last_x_agent_frame_id:
            temp.append([x_agent_pid, -1, location_xyz, None, None,
                         None, None])
          else:
          """
          temp.append([x_agent_pid, -1, location_xyz, direction_xyz, speed,
                       1/args.video_fps, False])
        if temp:
          new_person_controls[frame_id] = temp

    # interpolate the vehicle control
    vehicle_controls = interpolate_controls(
        this_moment_data["vehicle_controls"],
        args.video_fps)
    new_vehicle_controls = {}
    for frame_id in vehicle_controls:
      int_frame_id = int(frame_id)
      if int_frame_id <= last_x_agent_frame_id:
        new_vehicle_controls[int_frame_id] = vehicle_controls[frame_id]

    this_moment_data["ped_controls"] = new_person_controls
    this_moment_data["vehicle_controls"] = new_vehicle_controls
    this_moment_data["moment_id"] = make_moment_id(
        scene, moment_idx, x_agent_pid, dest_idx, annotator_id)

    new_moments.append(this_moment_data)

  multi_future_count = [multi_future_count[k]
                        for k in multi_future_count]
  print("start %s, end %s, min/max/mean length %.1f/%.1f/%.1f" % (
      sorted({o[0]:1 for o in lengths}.keys()),
      sorted({o[1]:1 for o in lengths}.keys()),
      min([o[2] for o in lengths]), max([o[2] for o in lengths]),
      np.mean([o[2] for o in lengths]),))
  print("Total %s final clips, %s moment, %s x_agents, %s destinations, " \
        "each has number of future (min/max/mean): %.1f/%.1f/%.1f" % (
            len(new_moments),
            len(num_moment),
            len(num_destinations),
            sum([num_destinations[k] for k in num_destinations]),
            min(multi_future_count),
            max(multi_future_count),
            np.mean(multi_future_count)))
  with open(args.final_json, "w") as f:
    json.dump(new_moments, f)
