# coding=utf-8
"""Utils for constructing moments."""

import cv2
import math
import os
import sys
import glob
import operator
import pygame

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla

import numpy as np

vehicle_z = {
    "0000": 0.2,
    "0401": 0.0,
    "0400": 0.0,
    "0500": 0.0
}


def make_moment_id(scene, moment_idx, x_agent_pid, dest_idx, annotator_id):
  return "%s_%d_%d_%d_%s" % (scene, moment_idx, x_agent_pid, dest_idx,
                             annotator_id)

def setup_walker_bps(world):
  # walker model that we use
  # 9-14 are kids, so we not using those
  walker_indexes = [1, 2, 3, 4, 5, 6, 7, 8]
  # the last item is the current cycled index of the bps
  walker_bps = [["walker.pedestrian.%04d" % o for o in walker_indexes], 0]
  walker_bps_list = [
      world.get_blueprint_library().find(one) for one in walker_bps[0]]
  walker_bps = [walker_bps_list, 0]
  return walker_bps

def setup_vehicle_bps(world):
  vehicle_bps = [
      [
          "vehicle.audi.a2",
          "vehicle.audi.etron",
          "vehicle.bmw.grandtourer",
          "vehicle.chevrolet.impala",
          "vehicle.citroen.c3",
          "vehicle.jeep.wrangler_rubicon",
          "vehicle.lincoln.mkz2017",
          "vehicle.nissan.micra",
          "vehicle.nissan.patrol",
      ], 0]
  vehicle_bps_list = [
      world.get_blueprint_library().find(one) for one in vehicle_bps[0]]
  vehicle_bps = [vehicle_bps_list, 0]
  return vehicle_bps


def get_bp(bps_):
  """Cycle through all available models."""
  bp_list, cur_index = bps_
  new_index = cur_index + 1
  if new_index >= len(bp_list):
    new_index = 0
  bps_[1] = new_index
  return bp_list[cur_index]

# seems some puddle on the ground makes the scene look perceptually more real.
realism_weather = carla.WeatherParameters(
    cloudyness=20.0,
    precipitation=0.0,
    sun_altitude_angle=65.0,
    precipitation_deposits=60.0,
    wind_intensity=80.0,
    sun_azimuth_angle=20.0)

# static scene info
static_scenes = {
    "zara01": {
        "fps": 25.0,
        "weather":{
            "cloudyness": 0.0,  # typo in the Carla 0.9.6 api
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "sun_altitude_angle": 85.0,
            "sun_azimuth_angle": 0.0,
            "wind_intensity": 80.0,
        },
        "static_cars":[],
        "map": "Town03_ethucy",
    },
    "eth": {
        "fps": 25.0,
        "weather":{
            "cloudyness": 0.0,  # typo in the Carla 0.9.6 api
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "sun_altitude_angle": 85.0,
            "sun_azimuth_angle": 0.0,
            "wind_intensity": 80.0,
        },
        "static_cars":[],
        "map": "Town03_ethucy",
    },
    "hotel": {
        "fps": 25.0,
        "weather":{
            "cloudyness": 0.0,  # typo in the Carla 0.9.6 api
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "sun_altitude_angle": 50.0,
            "sun_azimuth_angle": 270.0,
            "wind_intensity": 80.0,
        },
        "static_cars":[],
        "map": "Town03_ethucy",
    },
    "0000": {
        "fps": 30.0,
        "weather":{
            "cloudyness": 5.0,  # typo in the Carla 0.9.6 api
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "sun_altitude_angle": 120.0,
            "sun_azimuth_angle": 45.0,
            "wind_intensity": 80.0,
        },
        "static_cars":[],
        "map": "Town05_actev",
    },
    "0400": {
        "fps": 30.0,
        "weather":{
            "cloudyness": 5.0,  # typo in the Carla 0.9.6 api
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "sun_altitude_angle": 65.0,
            "sun_azimuth_angle": -20.0,
            "wind_intensity": 80.0,
        },
        "static_cars":[],
        "map": "Town05_actev",
    },
    "0401": {
        "fps": 30.0,
        "weather":{
            "cloudyness": 0.0,  # typo in the Carla 0.9.6 api
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "sun_altitude_angle": 50.0,
            "sun_azimuth_angle": 0.0,
            "wind_intensity": 80.0,
        },
        "static_cars":[],
        "map": "Town05_actev",
    },
    "0500": {
        "fps": 30.0,
        "weather":{
            "cloudyness": 0.0,  # typo in the Carla 0.9.6 api
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "sun_altitude_angle": 45.0,
            "sun_azimuth_angle": 0.0,
            "wind_intensity": 80.0,
        },
        "static_cars":[],
        "map": "Town05_actev",
    },
}
static_scenes["zara02"] = static_scenes["zara01"]


anchor_cameras = {
    "zara01": (carla.Transform(
        carla.Location(x=-33.863022, y=-56.820679, z=28.149984),
        carla.Rotation(pitch=-62.999184, yaw=-89.999214, roll=0.000053)), 30.0),
    "eth": (carla.Transform(
        carla.Location(x=42.360512, y=-29.728912, z=25.349985),
        carla.Rotation(pitch=-66.998413, yaw=-86.996346, roll=0.000057)), 85.0),
    "hotel": (carla.Transform(
        carla.Location(x=61.361576, y=-103.381432, z=22.765366),
        carla.Rotation(pitch=-63.188835, yaw=2.568912, roll=0.000136)), 45.0),
    "0000": (carla.Transform(
        carla.Location(x=-22.496830, y=-60.411972, z=12.070004),
        carla.Rotation(pitch=-30.999966, yaw=57.001354, roll=0.000025)), 65.0),
    "0400": (carla.Transform(
        carla.Location(x=-160.418839, y=33.280800, z=14.469944),
        carla.Rotation(pitch=-17.869482, yaw=54.943417, roll=0.000087)), 60.0),
    "0401": (carla.Transform(
        carla.Location(x=-120.234306, y=44.632133, z=10.556061),
        carla.Rotation(pitch=-26.056767, yaw=37.160381, roll=0.000123)), 70.0),
    "0500": (carla.Transform(
        carla.Location(x=-154.292542, y=-99.519508, z=26.075722),
        carla.Rotation(pitch=-23.600775, yaw=10.802525, roll=0.000002)), 30.0),
}
anchor_cameras["zara02"] = anchor_cameras["zara01"]

recording_cameras = {
    "zara01": [(carla.Transform(  # anchor-view
        carla.Location(x=-33.863022, y=-56.820679, z=28.149984),
        carla.Rotation(pitch=-62.999184, yaw=-89.999214, roll=0.000053)), 55.0),
               (carla.Transform(  # left
        carla.Location(x=-58.868965, y=-44.742245, z=28.149984),
        carla.Rotation(pitch=-37.998379, yaw=-49.998241, roll=0.000101)), 35.0),
               (carla.Transform(  # right
        carla.Location(x=-19.291773, y=-50.881275, z=28.149984),
        carla.Rotation(pitch=-45.998478, yaw=-118.998390, roll=0.00012)), 45.0),
               (carla.Transform(  # top-down
        carla.Location(x=-34.080410, y=-70.821098, z=54.799969),
        carla.Rotation(pitch=-84.997345, yaw=-88.997765, roll=0.000078)), 35.0),
    ],
    "eth": [(carla.Transform(  # anchor-view
        carla.Location(x=41.742561, y=-11.228123, z=22.499983),
        carla.Rotation(pitch=-38.997192, yaw=-89.996033, roll=0.000101)), 80.0),
               (carla.Transform(  # left
        carla.Location(x=13.706966, y=-17.478188, z=32.549980),
        carla.Rotation(pitch=-40.996883, yaw=-32.996113, roll=0.000096)), 55.0),
               (carla.Transform(  # right
        carla.Location(x=71.222115, y=-21.480639, z=32.499981),
        carla.Rotation(pitch=-39.996334, yaw=-156.996201, roll=0.000133)), 50.0),
               (carla.Transform(  # top-down
        carla.Location(x=40.933189, y=-24.631042, z=91.099937),
        carla.Rotation(pitch=-79.995010, yaw=-88.995430, roll=0.000216)), 50.0),
    ],
    "hotel": [(carla.Transform(  # anchor-view
        carla.Location(x=58.757435, y=-101.250473, z=25.415363),
        carla.Rotation(pitch=-64.188843, yaw=2.568922, roll=0.000135)), 65.0),
               (carla.Transform(  # left
        carla.Location(x=62.436810, y=-117.175545, z=19.665363),
        carla.Rotation(pitch=-40.187798, yaw=75.567665, roll=0.000085)), 65.0),
               (carla.Transform(  # right
        carla.Location(x=58.515789, y=-86.332535, z=19.665363),
        carla.Rotation(pitch=-38.187317, yaw=-59.432423, roll=0.000065)), 65.0),
               (carla.Transform(  # top-down
        carla.Location(x=66.663460, y=-102.476425, z=30.865358),
        carla.Rotation(pitch=-88.958252, yaw=-179.104660, roll=-179.895248)), 50.0),
    ],
    "0000": [(carla.Transform(  # anchor-view
        carla.Location(x=-21.545109, y=-61.469452, z=12.120005),
        carla.Rotation(pitch=-39.999821, yaw=65.000923, roll=0.000018)), 90.0),
               (carla.Transform(  # left
        carla.Location(x=-7.899245, y=-64.047493, z=12.120005),
        carla.Rotation(pitch=-37.999504, yaw=106.000496, roll=0.000020)), 90.0),
               (carla.Transform(  # right
        carla.Location(x=-38.025734, y=-52.780418, z=11.870004),
        carla.Rotation(pitch=-30.999201, yaw=32.000214, roll=0.000047)), 70.0),
               (carla.Transform(  # top-down
        carla.Location(x=-12.978075, y=-32.058861, z=48.219952),
        carla.Rotation(pitch=-87.999031, yaw=0.000000, roll=0.000000)), 70.0),
        #       (carla.Transform(  # dashboard view
        #carla.Location(x=-20.141300, y=-55.388958, z=2.220005),
        #carla.Rotation(pitch=0.000260, yaw=66.998703, roll=0.000035)), 90.0),
    ],
    "0400": [(carla.Transform(  # anchor-view
        carla.Location(x=-163.437454, y=26.059809, z=20.669943),
        carla.Rotation(pitch=-19.869471, yaw=36.942986, roll=0.000073)), 60.0),
               (carla.Transform(  # left
        carla.Location(x=-114.143768, y=7.405876, z=23.469940),
        carla.Rotation(pitch=-27.869444, yaw=81.942535, roll=0.000055)), 60.0),
               (carla.Transform(  # right
        carla.Location(x=-173.366577, y=70.659279, z=23.469940),
        carla.Rotation(pitch=-23.869440, yaw=-1.057342, roll=0.000046)), 55.0),
               (carla.Transform(  # top-down
        carla.Location(x=-107.249977, y=49.348232, z=101.969933),
        carla.Rotation(pitch=-83.868240, yaw=89.941933, roll=0.000096)), 55.0),
        #       (carla.Transform(  # dashboard view
        #carla.Location(x=-163.437454, y=26.059809, z=0.819942),
        #carla.Rotation(pitch=5.130530, yaw=42.942547, roll=0.000073)), 60.0),
    ],
    "0401": [(carla.Transform(  # anchor-view
        carla.Location(x=-128.780029, y=31.252804, z=16.156065),
        carla.Rotation(pitch=-26.056767, yaw=42.160397, roll=0.000124)), 80.0),
               (carla.Transform(  # left
        carla.Location(x=-101.373863, y=15.802762, z=16.156065),
        carla.Rotation(pitch=-26.056761, yaw=91.160004, roll=0.000150)), 75.0),
               (carla.Transform(  # right
        carla.Location(x=-139.725403, y=61.328167, z=16.156065),
        carla.Rotation(pitch=-30.818111, yaw=-1.363098, roll=0.000145)), 80.0),
               (carla.Transform(  # top-down
        carla.Location(x=-109.142944, y=58.624207, z=70.706039),
        carla.Rotation(pitch=-80.815720, yaw=0.636051, roll=0.000164)), 65.0),
        #       (carla.Transform(  # dashboard view
        #carla.Location(x=-128.780029, y=31.252802, z=1.306065),
        #carla.Rotation(pitch=3.943252, yaw=42.161617, roll=0.000125)), 80.0),
    ],
    "0500": [(carla.Transform(  # anchor-view
        carla.Location(x=-154.292542, y=-99.519508, z=26.075722),
        carla.Rotation(pitch=-23.600767, yaw=10.802557, roll=0.000002)), 35.0),
               (carla.Transform(  # left
        carla.Location(x=-150.165619, y=-129.959244, z=26.075722),
        carla.Rotation(pitch=-23.600767, yaw=42.802635, roll=0.000008)), 35.0),
               (carla.Transform(  # right
        carla.Location(x=-157.999283, y=-55.524170, z=26.075722),
        carla.Rotation(pitch=-23.600767, yaw=-33.197342, roll=0.000007)), 35.0),
               (carla.Transform(  # top-down
        carla.Location(x=-100.003044, y=-96.517174, z=52.925720),
        carla.Rotation(pitch=-78.599899, yaw=89.801888, roll=0.000000)), 70.0),
        #       (carla.Transform( # dashboard view
        #carla.Location(x=-144.576553, y=-97.665466, z=1.325722),
        #carla.Rotation(pitch=2.399254, yaw=16.803694, roll=0.000003)), 45.0),
    ],
}
recording_cameras["zara02"] = recording_cameras["zara01"]


anchor_cameras_annotation = {
    "zara01": (carla.Transform(
        carla.Location(x=-33.937153, y=-65.975639, z=13.199974),
        carla.Rotation(pitch=-63.998699, yaw=-90.999649, roll=0.000117)), 90.0),
    "eth": (carla.Transform(
        carla.Location(x=41.688187, y=-16.916178, z=25.349985),
        carla.Rotation(pitch=-44.997559, yaw=-86.996063, roll=0.000133)), 90.0),
    "hotel": (carla.Transform(
        carla.Location(x=62.348896, y=-101.509659, z=22.765366),
        carla.Rotation(pitch=-69.188515, yaw=-0.431061, roll=0.000136)), 90.0),
    "0000": (carla.Transform(
        carla.Location(x=-21.634167, y=-60.972176, z=12.070004),
        carla.Rotation(pitch=-30.999966, yaw=59.001438, roll=0.000028)), 90.0),
    "0400": (carla.Transform(
        carla.Location(x=-160.418839, y=33.280800, z=14.469944),
        carla.Rotation(pitch=-17.869482, yaw=54.943417, roll=0.000087)), 90.0),
    "0401": (carla.Transform(
        carla.Location(x=-120.234306, y=44.632133, z=10.556061),
        carla.Rotation(pitch=-26.056767, yaw=37.160381, roll=0.000123)), 90.0),
    "0500": (carla.Transform(
        carla.Location(x=-154.292542, y=-99.519508, z=26.075722),
        carla.Rotation(pitch=-23.600775, yaw=10.802525, roll=0.000002)), 90.0),
}
anchor_cameras_annotation["zara02"] = anchor_cameras_annotation["zara01"]


def reset_x_agent_key(moment_data):
  # reset all x_agents dict's key to int, since stupid json
  for i in range(len(moment_data)):
    this_moment_data = moment_data[i]
    new_x_agents = {}
    for key in this_moment_data["x_agents"]:
      new_key = int(float(key))
      new_x_agents[new_key] = this_moment_data["x_agents"][key]
    moment_data[i]["x_agents"] = new_x_agents


def interpolate_data_between(p1, p2):
  # p1, p2 is [frame_id, person_id, x, y, ..]
  data_points = []
  num_frames = int(p2[0] - p1[0])
  for i in range(num_frames - 1):
    new_data_point = [p1[0] + i + 1, p1[1]]
    for coor1, coor2 in zip(p1[2:], p2[2:]):
      inc = (coor2 - coor1) / num_frames
      this_coor = coor1 + inc * (i + 1)
      new_data_point.append(this_coor)
    data_points.append(new_data_point)
  return data_points


def interpolate_controls(controls, fps_):
  """Given low frame rate controls, interpolate."""
  # first, get the traj data
  # [frame_id, person_id, x, y, z]
  data = []
  for frame_id in controls:
    for pid, _, (x, y, z), _, _, _, is_stationary in controls[frame_id]:
      # json BS
      int_frame_id = int(float(frame_id))
      # need the is_stationary to keep things the same
      data.append([int_frame_id, int(pid), x, y, z, is_stationary])

  if len(data) == 0:
    return {}
  data.sort(key=operator.itemgetter(0))
  data = np.array(data, dtype="float64")

  person_ids = np.unique(data[:, 1]).tolist()
  # the frame_id in control data should be offset to the start of
  # the actual moment
  # frame_id always start from 0
  control_data = {}  # frame_id ->
  for person_id in person_ids:
    this_data = data[data[:, 1] == person_id, :]
    is_stationaries = this_data[:, -1]
    this_data = this_data[:, :-1]
    # here this_data should already be sorted by frame_id ASC
    if this_data.shape[0] <= 1:
      continue
    # interpolate the points in between
    # assuming constant velocity
    # don't interpolate if the second point is already stationary
    if is_stationaries[1] != 1.0:
      new_data = []
      new_stationaries = []
      for i in range(this_data.shape[0] - 1):
        j = i + 1
        # add the start point
        this_new = [this_data[i]]
        this_new += interpolate_data_between(this_data[i], this_data[j])
        new_data += this_new
        new_stationaries += [is_stationaries[i]] * len(this_new)
      new_data.append(this_data[-1])
      new_stationaries.append(is_stationaries[-1])
      this_data = np.array(new_data, dtype="float64")
      is_stationaries = np.array(new_stationaries, dtype="float64")

    for i in range(this_data.shape[0] - 1):
      frame_id = int(this_data[i, 0])
      j = i + 1
      is_stationary = is_stationaries[i]
      # direction vector
      direction_vector, speed, time_elasped = get_direction_and_speed(
          this_data[j], this_data[i], fps_)
      #if not control_data.has_key(frame_id):
      if frame_id not in control_data:
        control_data[frame_id] = []
      control_data[frame_id].append(
          [person_id, this_data[i, 0], this_data[i, 2:].tolist(),
           direction_vector, speed,
           time_elasped, is_stationary])

    last_frame_id = int(this_data[-1, 0])
    #if not control_data.has_key(last_frame_id):
    if last_frame_id not in control_data:
      control_data[last_frame_id] = []
    # signaling stop
    control_data[last_frame_id].append(
        [person_id, this_data[i, 0], this_data[-1, 2:].tolist(),
         None, None, None, None])


  # json bs
  new_control_data = {}
  for frame_id in control_data:
    string_frame_id = "%s" % frame_id
    new_control_data[string_frame_id] = control_data[frame_id]
  return new_control_data


def reset_bps(bps_):
  bps_[1] = 0


def get_controls(traj_file, start_frame, end_frame, fps_, interpolate=False,
              z_to=None, no_offset=False):
  """Gather the trajectories and convert to control data."""
  data = [o.strip().split("\t") for o in open(traj_file).readlines()]
  data = np.array(data, dtype="float64")  # [frame_id, person_id, x, y, z]

  control_data, total_frame_num = get_controls_from_traj_data(
      data, start_frame, end_frame, fps_, interpolate=interpolate,
      z_to=z_to, no_offset=no_offset)

  return control_data, total_frame_num

def get_controls_from_traj_data(data, start_frame, end_frame, fps_,
                                interpolate=False, z_to=None, no_offset=False):

  if z_to is not None:
    # for car traj, set all z coordinates to 0
    data[:, -1] = z_to

  frame_ids = np.unique(data[:, 0]).tolist()
  frame_ids.sort()
  if start_frame == -1:
    target_frame_ids = frame_ids
  else:
    if start_frame not in frame_ids:
      return {}, 0
    start_idx = frame_ids.index(start_frame)
    end_idx = frame_ids.index(end_frame)
    target_frame_ids = frame_ids[start_idx:end_idx]
  total_frame_num = int(target_frame_ids[-1] - target_frame_ids[0])

  filtered_data = data[np.isin(data[:, 0], target_frame_ids), :]
  # compute the direction vector and speed at each timestep
  # per person
  person_ids = np.unique(filtered_data[:, 1]).tolist()
  # the frame_id in control data should be offset to the start of
  # the actual moment
  # frame_id always start from 0
  control_data = {}  # frame_id ->
  # compute the absolute change between points so we can identify when
  # the traj is stationary like a parked car
  traj_change_future_seconds = 2.0  # for each frame, look at change in future
  traj_change_future_frames = fps_ * traj_change_future_seconds
  stationary_thres = 0.08  # from experience
  for person_id in person_ids:
    this_data = filtered_data[filtered_data[:, 1] == person_id, :]
    # here this_data should already be sorted by frame_id ASC
    if this_data.shape[0] <= 1:
      continue
    if interpolate:
      # interpolate the points in between
      # assuming constant velocity
      new_data = []
      for i in range(this_data.shape[0] - 1):
        j = i + 1
        # add the start point
        new_data.append(this_data[i])
        new_data += interpolate_data_between(this_data[i], this_data[j])
      new_data.append(this_data[-1])
      this_data = np.array(new_data, dtype="float64")

    is_stationary_before_end = False  # use this for last few frames
    for i in range(this_data.shape[0] - 1):
      # start from zero
      frame_id = int(this_data[i, 0] - target_frame_ids[0])
      if no_offset:
        frame_id = int(this_data[i, 0])
      j = i + 1
      # compute the future changes
      future_i = None
      for t in range(j, this_data.shape[0]):
        if this_data[t, 0] - this_data[i, 0] >= traj_change_future_frames:
          future_i = t
          break
      is_stationary = False
      if future_i is not None:
        diff = np.linalg.norm(this_data[future_i, 2:] - this_data[i, 2:])
        if diff <= stationary_thres:
          is_stationary = True
          is_stationary_before_end = True
      else:
        is_stationary = is_stationary_before_end
      # direction vector
      direction_vector, speed, time_elasped = get_direction_and_speed(
          this_data[j], this_data[i], fps_)
      #if not control_data.has_key(frame_id):
      if frame_id not in control_data:
        control_data[frame_id] = []
      control_data[frame_id].append(
          [person_id, this_data[i, 0], this_data[i, 2:].tolist(),
           direction_vector, speed,
           time_elasped, is_stationary])

    last_frame_id = int(this_data[-1, 0] - target_frame_ids[0])
    if no_offset:
      last_frame_id = int(this_data[-1, 0])
    #if not control_data.has_key(last_frame_id):
    if last_frame_id not in control_data:
      control_data[last_frame_id] = []
    # signaling stop
    control_data[last_frame_id].append(
        [person_id, this_data[i, 0], this_data[-1, 2:].tolist(),
         None, None, None, None])
  return control_data, total_frame_num


def cleanup_actors(actor_list, client):
  for actor in actor_list:
    if actor.type_id.startswith("sensor") and actor.is_alive:
      actor.stop()
  # finished, clean actors
  if actor_list:
    client.apply_batch(
        [carla.command.DestroyActor(x) for x in actor_list])


def control_data_to_traj(control_data):
  """Convert the control data back to trajectory data."""
  # person/vehicle ID -> a list of [frame_id, xyz, is_stationary]
  traj_data = {}
  frame_ids = {}
  for frame_id in control_data:
    for one in control_data[frame_id]:
      frame_id = int(frame_id)
      p_id, ori_frame_id, xyz, _, speed, time_elasped, is_stationary = \
          one
      if p_id not in traj_data:
        traj_data[p_id] = []
      traj_data[p_id].append({
          "frame_id": frame_id,
          "xyz": xyz,
          "is_stationary": is_stationary,
          "speed": speed})
      frame_ids[frame_id] = 1
  for p_id in traj_data:
    traj_data[p_id].sort(key=operator.itemgetter("frame_id"))
  return traj_data, sorted(frame_ids.keys())


speed_calibration = 1.22  # used to account for the acceleration period
def get_direction_and_speed(destination, current, fps_):
  """destination.xyz - current.xyz then normalize. also get the speed."""
  direction_vector = [
      destination[2] - current[2],
      destination[3] - current[3],
      0.0]
  vector_length = math.sqrt(sum([x**2 for x in direction_vector])) + \
      np.finfo(float).eps
  direction_vector = [x / vector_length for x in direction_vector]
  direction_vector = [float(x) for x in direction_vector]

  time_elasped = (destination[0] - current[0]) / fps_
  speed = vector_length / time_elasped * speed_calibration  # meter per second

  return direction_vector, speed, time_elasped


def get_scene(videoname_):
  """ActEV scene extractor from videoname."""
  s = videoname_.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]


class CollisionSensor(object):
  def __init__(self, parent_actor, actorid2info, world, verbose=False):
    self.world = world
    self.verbose = verbose
    self.parent = parent_actor
    self.actorid2info = actorid2info
    bp = world.get_blueprint_library().find("sensor.other.collision")
    self.sensor = world.spawn_actor(bp, carla.Transform(),
                                    attach_to=self.parent)
    self.parent_tag = actorid2info[parent_actor.id]
    self.history = []
    self.sensor.listen(self.on_collision)

  def on_collision(self, event):
    frame_id = event.frame
    other_actor_id = event.other_actor.id
    other_actor_carla_tag = event.other_actor.type_id
    other_actor_tag = None
    #if not self.actorid2info.has_key(other_actor_id):
    if other_actor_id not in self.actorid2info:
      if self.verbose:
        print("%s: %s collide with %s." % (
            frame_id,
            self.parent_tag, other_actor_carla_tag))
    else:
      other_actor_tag = self.actorid2info[other_actor_id]
      if self.verbose:
        print("%s: %s collide with %s." % (
            frame_id,
            self.parent_tag, other_actor_tag))

    self.history.append((frame_id, self.parent.id, other_actor_id,
                         self.parent_tag, other_actor_tag,
                         other_actor_carla_tag))


def setup_static(world, client, scene_elements, actor_list):
  this_weather = scene_elements["weather"]
  weather = carla.WeatherParameters(
      cloudyness=this_weather["cloudyness"],
      precipitation=this_weather["precipitation"],
      precipitation_deposits=this_weather["precipitation_deposits"],
      sun_altitude_angle=this_weather["sun_altitude_angle"],
      sun_azimuth_angle=this_weather["sun_azimuth_angle"],
      wind_intensity=this_weather["wind_intensity"])
  world.set_weather(weather)

  spawn_cmds = []
  for car in scene_elements["static_cars"]:
    car_location = carla.Location(x=car["location_xyz"][0],
                                  y=car["location_xyz"][1],
                                  z=car["location_xyz"][2])
    car_rotation = carla.Rotation(pitch=car["rotation_pyr"][0],
                                  yaw=car["rotation_pyr"][1],
                                  roll=car["rotation_pyr"][2])
    car_bp = world.get_blueprint_library().find(car["bp"])
    assert car_bp is not None, car_bp
    # the static car can be walked though
    spawn_cmds.append(
        carla.command.SpawnActor(
            car_bp, carla.Transform(
                location=car_location, rotation=car_rotation)).then(
                    carla.command.SetSimulatePhysics(
                        carla.command.FutureActor, False)))

  # spawn the actors needed for the static scene setup
  if spawn_cmds:
    response = client.apply_batch_sync(spawn_cmds)
    all_actors = world.get_actors([x.actor_id for x in response])
    actor_list += all_actors


def run_sim_for_one_frame(frame_id, ped_controls, vehicle_controls,
                          cur_peds, cur_ped_collisions,
                          cur_vehicles, cur_veh_init_f_vec, cur_veh_prev_yaw,
                          walker_bps, vehicle_bps,
                          world,
                          global_actor_list, actorid2info,
                          show_traj=False, verbose=False, max_yaw_change=60,
                          exit_if_spawn_fail=False,
                          no_collision_detector=False,
                          pid2actor={},
                          excepts=[]):
  """Given the controls and the current frame_id, run the simulation. Return
     the batch command to execute, return None if the any spawning failed
  """
  stats = {
      "vehicle_spawn_failed": False,
  }
  batch_cmds = []

  #if ped_controls.has_key(frame_id):
  if frame_id in ped_controls:
    this_control_data = ped_controls[frame_id]
    for person_id, _, xyz, direction_vector, speed, time_elasped, \
        is_static in this_control_data:
      if person_id in excepts:
        continue
      # last location reached, so delete this guy
      if direction_vector is None:
        #if cur_peds.has_key(person_id):
        if person_id in cur_peds:
          if not no_collision_detector:
            cur_ped_collisions[person_id].sensor.stop()
            batch_cmds.append(carla.command.DestroyActor(
                cur_ped_collisions[person_id].sensor))
            del cur_ped_collisions[person_id]
          batch_cmds.append(carla.command.DestroyActor(cur_peds[person_id]))
          del cur_peds[person_id]

      else:
        walker_control = carla.WalkerControl()
        walker_control.direction = carla.Vector3D(
            x=direction_vector[0],
            y=direction_vector[1],
            z=direction_vector[2])
        walker_control.speed = speed
        # new person, need to spawn
        #if not cur_peds.has_key(person_id):
        if person_id not in cur_peds:
          walker_bp = get_bp(walker_bps)
          # TODo: initial rotation?
          new_walker = world.try_spawn_actor(walker_bp, carla.Transform(
              location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2])))
          if new_walker is None:
            if verbose:
              print("%s: person %s failed to spawn." % (
                  frame_id, person_id))
            if exit_if_spawn_fail:
              return None, stats
            else:
              continue
          if verbose:
            print("%s: Spawned person id %s." % (
                frame_id, person_id))
          # Bug: walker will loose physics in 0.9.6
          #new_walker.set_simulate_physics(True)
          cur_peds[person_id] = new_walker
          global_actor_list.append(new_walker)

          # add a collision sensor
          actorid2info[new_walker.id] = ("Person", person_id)
          pid2actor[person_id] = new_walker

          if not no_collision_detector:
            collision_sensor = CollisionSensor(new_walker, actorid2info,
                                               world, verbose=verbose)
            global_actor_list.append(collision_sensor.sensor)
            cur_ped_collisions[person_id] = collision_sensor
          if show_traj:
            # show the track Id
            world.debug.draw_string(carla.Location(
                x=xyz[0], y=xyz[1], z=xyz[2]), "# %s" % person_id,
                                    draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0),
                                    life_time=30.0)

        this_walker_actor = cur_peds[person_id]

        if show_traj:
          delay = 0.0  # delay before removing traj
          p1 = carla.Location(x=xyz[0], y=xyz[1], z=xyz[2])
          next_xyz = [xyz[i] + direction_vector[i] * speed * time_elasped
                      for i in range(3)]
          p2 = carla.Location(x=next_xyz[0], y=next_xyz[1], z=next_xyz[2])
          world.debug.draw_arrow(p1, p2, thickness=0.1, arrow_size=0.1,
                                 color=carla.Color(r=255),
                                 life_time=time_elasped + delay)

        if is_static:
          # stop the walker
          batch_cmds.append(carla.command.ApplyWalkerControl(
              this_walker_actor, carla.WalkerControl()))
          continue
        batch_cmds.append(carla.command.ApplyWalkerControl(
            this_walker_actor, walker_control))
        #velocity = carla.Vector3D(
        #    x=direction_vector[0] * speed * time_elasped,
        #    y=direction_vector[1] * speed * time_elasped,
        #    z=direction_vector[2] * speed * time_elasped)
        # Carla Bug: set_velocity does not work on 0.9.6
        #this_walker_actor.set_velocity(velocity)


  # add vehicle
  #if vehicle_controls.has_key(frame_id):
  if frame_id in vehicle_controls:
    this_control_data = vehicle_controls[frame_id]
    for vehicle_id, _, xyz, direction_vector, speed, time_elasped, \
        is_static in this_control_data:
      # last location reached, so delete this guy
      if direction_vector is None:
        #if cur_vehicles.has_key(vehicle_id):
        if vehicle_id in cur_vehicles:
          this_vehicle_actor = cur_vehicles[vehicle_id]
          batch_cmds.append(carla.command.DestroyActor(
              this_vehicle_actor))
          del cur_vehicles[vehicle_id]
      else:

        #if not cur_vehicles.has_key(vehicle_id):
        if vehicle_id not in cur_vehicles:
          vehicle_bp = get_bp(vehicle_bps)

          new_vehicle = world.try_spawn_actor(vehicle_bp, carla.Transform(
              location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2])))
          if new_vehicle is None:
            if verbose:
              print("%s: vehicle %s failed to spawn." % (
                  frame_id, vehicle_id))
            #if exit_if_spawn_fail:
            #  return None
            #else:
            #  continue
            # somehow 0400/0401 car spawn always fail at the first few frames
            stats["vehicle_spawn_failed"] = True
            continue  # dont' exit even if car fails
          if verbose:
            print("%s: Spawned vehicle id %s." % (
                frame_id, vehicle_id))
          # if car physics is false, the car will not
          # be affected by set_velocity
          # Todo: set physics to be true and use a PID controller
          # collision sensor on walker will not fire when
          # the walker collide with car
          new_vehicle.set_simulate_physics(False)
          #new_vehicle.set_transform(carla.Transform(
          #    location=carla.Location(x=xyz[0]+0.5, y=xyz[1], z=xyz[2]),
          #    rotation=carla.Rotation(yaw=270)))
          cur_vehicles[vehicle_id] = new_vehicle
          actorid2info[new_vehicle.id] = ("Vehicle", vehicle_id)
          global_actor_list.append(new_vehicle)

          # initial rotation
          current_transform = new_vehicle.get_transform()
          current_forward_vector = \
              current_transform.rotation.get_forward_vector()
          cur_veh_init_f_vec[vehicle_id] = \
              current_forward_vector

          if show_traj:
            # show the track Id
            world.debug.draw_string(carla.Location(
                x=xyz[0], y=xyz[1], z=xyz[2]), "# %s" % vehicle_id,
                                    draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0),
                                    life_time=30.0)
        this_vehicle_actor = cur_vehicles[vehicle_id]

        if show_traj:
          delay = 5.0  # delay before removing traj
          p1 = carla.Location(x=xyz[0], y=xyz[1], z=xyz[2])
          next_xyz = [xyz[i] + direction_vector[i] * speed * time_elasped
                      for i in range(3)]
          p2 = carla.Location(x=next_xyz[0], y=next_xyz[1], z=next_xyz[2])
          world.debug.draw_arrow(p1, p2, thickness=0.1, arrow_size=0.1,
                                 color=carla.Color(r=255),
                                 life_time=time_elasped + delay)
        if is_static:
          continue
        # Todo: add a PID controller to get car from point A to B
        initial_forward_vector = cur_veh_init_f_vec[vehicle_id]
        v0 = np.array(
            [initial_forward_vector.x, initial_forward_vector.y])
        v1 = np.array([direction_vector[0], direction_vector[1]])
        # the angle of degree between the current direction vector and
        # the initial vehicle model forward vector
        yaw_degree = np.math.atan2(
            np.linalg.det([v0, v1]), np.dot(v0, v1))
        yaw_degree = np.rad2deg(yaw_degree)

        # smoothing out the sudden big change of yaw degree
        #if not cur_veh_prev_yaw.has_key(vehicle_id):
        if vehicle_id not in cur_veh_prev_yaw:
          cur_veh_prev_yaw[vehicle_id] = yaw_degree
        else:
          prev_yaw = cur_veh_prev_yaw[vehicle_id]
          if abs(prev_yaw - yaw_degree) > max_yaw_change:
            yaw_degree = prev_yaw
          else:
            cur_veh_prev_yaw[vehicle_id] = yaw_degree

        batch_cmds.append(carla.command.ApplyTransform(
            this_vehicle_actor, carla.Transform(
                location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]),
                rotation=carla.Rotation(roll=0,
                                        pitch=0,
                                        yaw=yaw_degree))))
  return batch_cmds, stats


def cross(carla_vector3d_1, carla_vector3d_2):
  """Cross product."""
  return carla.Vector3D(
      x=carla_vector3d_1.y * carla_vector3d_2.z -
      carla_vector3d_1.z * carla_vector3d_2.y,
      y=carla_vector3d_1.z * carla_vector3d_2.x -
      carla_vector3d_1.x * carla_vector3d_2.z,
      z=carla_vector3d_1.x * carla_vector3d_2.y -
      carla_vector3d_1.y * carla_vector3d_2.x)


def get_degree_of_two_vectors(vec1, vec2):
  x1, y1 = vec1
  x2, y2 = vec2
  dot = x1*x2 + y1*y2
  det = x1*y2 - y1*x2
  angle_rad = np.arctan2(det, dot)
  return np.rad2deg(angle_rad)


def parse_carla_depth(depth_image):
  """Parse Carla depth image."""
  # 0.9.6: The image codifies the depth in 3 channels of the RGB color space,
  # from less to more significant bytes: R -> G -> B.
  # depth_image is [h, w, 3], last dim is RGB order
  depth_image = depth_image.astype("float32")
  normalized = (depth_image[:, :, 0] + depth_image[:, :, 1]*256 + \
      depth_image[:, :, 2]*256*256) / (256 * 256 * 256 - 1)
  return 1000 * normalized


def compute_intrinsic(img_width, img_height, fov):
  """Compute intrinsic matrix."""
  intrinsic = np.identity(3)
  intrinsic[0, 2] = img_width / 2.0
  intrinsic[1, 2] = img_height / 2.0
  intrinsic[0, 0] = intrinsic[1, 1] = img_width / (2.0 * np.tan(fov *
                                                                np.pi / 360.0))
  return intrinsic


def compute_extrinsic_from_transform(transform_):
  """
  Creates extrinsic matrix from carla transform.
  This is known as the coordinate system transformation matrix.
  """

  rotation = transform_.rotation
  location = transform_.location
  c_y = np.cos(np.radians(rotation.yaw))
  s_y = np.sin(np.radians(rotation.yaw))
  c_r = np.cos(np.radians(rotation.roll))
  s_r = np.sin(np.radians(rotation.roll))
  c_p = np.cos(np.radians(rotation.pitch))
  s_p = np.sin(np.radians(rotation.pitch))
  matrix = np.matrix(np.identity(4))  # matrix is needed
  # 3x1 translation vector
  matrix[0, 3] = location.x
  matrix[1, 3] = location.y
  matrix[2, 3] = location.z
  # 3x3 rotation matrix
  matrix[0, 0] = c_p * c_y
  matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
  matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
  matrix[1, 0] = s_y * c_p
  matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
  matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
  matrix[2, 0] = s_p
  matrix[2, 1] = -c_p * s_r
  matrix[2, 2] = c_p * c_r
  # [3, 3] == 1, rest is zero
  return matrix


def save_rgb_image(rgb_np_img, save_file):
  """Convert the RGB numpy image for cv2 to save."""
  # RGB np array
  image_to_save = rgb_np_img
  image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
  save_path = os.path.dirname(save_file)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  cv2.imwrite(save_file, image_to_save)


def to_xyz(carla_vector3d):
  """Return xyz coordinates as list from carla vector3d."""
  return [carla_vector3d.x, carla_vector3d.y, carla_vector3d.z]


def make_text_surface(text, offset):
  """Put text on a black bg pygame surface."""
  font = pygame.font.Font(pygame.font.get_default_font(), 20)
  text_width, text_height = font.size(text)

  surface = pygame.Surface((text_width, text_height))
  surface.fill((0, 0, 0, 0))  # black bg
  text_texture = font.render(text, True, (255, 255, 255))  # white color
  surface.blit(text_texture, (0, 0))  # upper-left corner

  return surface, offset + surface.get_height()


def get_2d_bbox(bbox_3d, max_w, max_h):
  """Given the computed [8, 3] points with depth, get the one bbox."""
  if all(bbox_3d[:, 2] > 0):
    # make one 2d bbox from 8 points
    x1 = round(np.min(bbox_3d[:, 0]), 3)
    y1 = round(np.min(bbox_3d[:, 1]), 3)
    x2 = round(np.max(bbox_3d[:, 0]), 3)
    y2 = round(np.max(bbox_3d[:, 1]), 3)
    if x1 > max_w or y1 > max_h:
      return None
    if x1 < 0:
      x1 = 0
    if y1 < 0:
      y1 = 0
    if x2 > max_w:
      x2 = max_w
    if y2 > max_h:
      y2 = max_h
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]
  else:
    return None

def get_3d_bbox(actor_, camera_actor):
  """Get the 8 point coordinates of the actor in the camera view."""
  # 1. get the 8 vertices of the actor box
  vertices = np.zeros((8, 4), dtype="float")
  extent = actor_.bounding_box.extent  # x, y, z extension from the center
  vertices[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
  vertices[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
  vertices[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
  vertices[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
  vertices[4, :] = np.array([extent.x, extent.y, extent.z, 1])
  vertices[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
  vertices[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
  vertices[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
  # vertices coordinates are relative to the actor (0, 0, 0)
  # now get the world coordinates
  # the bounding_box.location is all 0?
  center_transform = carla.Transform(actor_.bounding_box.location)
  center_rt = compute_extrinsic_from_transform(center_transform)
  # actor_ is in the world coordinates
  actor_rt = compute_extrinsic_from_transform(actor_.get_transform())
  # dont know why
  bbox_rt = np.dot(actor_rt, center_rt)

  # vertices relative to the bbox center
  # bbox center relative to the parent actor
  # so :
  # [4, 8] # these are in the world coordinates
  world_vertices = np.dot(bbox_rt, np.transpose(vertices))

  # now we transform vertices in world
  # to the camera 3D coordinate system
  camera_rt = compute_extrinsic_from_transform(camera_actor.get_transform())
  camera_rt_inv = np.linalg.inv(camera_rt)
  # [3, 8]
  x_y_z = np.dot(camera_rt_inv, world_vertices)[:3, :]
  # wadu hek? why?, unreal coordinates problem?
  # email me (junweil@cs.cmu.edu) if you know why
  y_minus_z_x = np.concatenate(
      [x_y_z[1, :], - x_y_z[2, :], x_y_z[0, :]], axis=0)

  # then we dot the intrinsic matrix then we got the pixel coordinates and ?
  # [8, 3]
  actor_bbox = np.transpose(np.dot(camera_actor.intrinsic, y_minus_z_x))
  # last dim keeps the scale factor?
  actor_bbox = np.concatenate(
      [actor_bbox[:, 0] / actor_bbox[:, 2],
       actor_bbox[:, 1] / actor_bbox[:, 2], actor_bbox[:, 2]], axis=1)

  return actor_bbox


# -------------- visualization

def draw_boxes(im, boxes, labels=None, colors=None, font_scale=0.6,
               font_thick=1, box_thick=1, bottom_text=False):
  """Draw boxes with labels on an image."""

  # boxes need to be x1, y1, x2, y2
  if not boxes:
    return im

  boxes = np.asarray(boxes, dtype="int")

  FONT = cv2.FONT_HERSHEY_SIMPLEX
  FONT_SCALE = font_scale

  if labels is not None:
    assert len(labels) == len(boxes)
  if colors is not None:
    assert len(labels) == len(colors)

  im = im.copy()

  for i in range(len(boxes)):
    box = boxes[i]

    color = (218, 218, 218)
    if colors is not None:
      color = colors[i]

    lineh = 2  # for box enlarging, replace with text height if there is label
    if labels is not None:
      label = labels[i]

      # find the best placement for the text
      ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, font_thick)
      bottom_left = [box[0] + 1, box[1] - 0.3 * lineh]
      top_left = [box[0] + 1, box[1] - 1.3 * lineh]
      if top_left[1] < 0:  # out of image
        top_left[1] = box[3] - 1.3 * lineh
        bottom_left[1] = box[3] - 0.3 * lineh

      textbox = [int(top_left[0]), int(top_left[1]),
                 int(top_left[0] + linew), int(top_left[1] + lineh)]

      if bottom_text:
        cv2.putText(im, label, (box[0] + 2, box[3] - 4),
                    FONT, FONT_SCALE, color=color)
      else:
        cv2.putText(im, label, (textbox[0], textbox[3]),
                    FONT, FONT_SCALE, color=color)  #, lineType=cv2.LINE_AA)

    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                  color=color, thickness=box_thick)
  return im

