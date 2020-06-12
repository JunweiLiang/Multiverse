# coding=utf-8
"""After starting a carla server off-sceen, use this to connect as spectator."""

from __future__ import print_function

import argparse
import cv2
import datetime
import json
import math
import sys
import os
import glob
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla
import pygame
import time

from utils import setup_static
from utils import setup_walker_bps
from utils import setup_vehicle_bps
from utils import anchor_cameras
from utils import get_bp
from utils import control_data_to_traj
from utils import compute_intrinsic
from utils import cross
from utils import parse_carla_depth
from utils import compute_extrinsic_from_transform
from utils import get_degree_of_two_vectors
from utils import save_rgb_image
from utils import get_3d_bbox
from utils import make_text_surface
from utils import get_direction_and_speed
from utils import run_sim_for_one_frame
from utils import cleanup_actors
from utils import reset_x_agent_key
from utils import reset_bps

parser = argparse.ArgumentParser()

parser.add_argument("moment_json", help="control data of the moment")
parser.add_argument("new_moment_json", help="save the new stuff to new json")

# interface setup
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--res", default="1280x720")

parser.add_argument("--is_actev", action="store_true")

# moment setup
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--annotation_fps", type=float, default=2.5)
parser.add_argument("--obs_length", type=int, default=12,
                    help="observation timestep, 12/2.5=4.8 seconds")
parser.add_argument("--pred_length", type=int, default=26,
                    help="observation timestep, 26/2.5=10.4 seconds")


class Camera(object):
  """Camera object to have a surface."""
  def __init__(self, camera_actor, camera_type="rgb",
               image_type=carla.ColorConverter.Raw,
               width=None, height=None, fov=None):
    self.camera_actor = camera_actor
    self.image_type = image_type

    self.last_image_frame_num = None  # the frame num of the image
    self.last_image_seconds = None  # the seconds since beginning of eposide?
    self.rgb_image = None  # last RGB image
    self.pygame_surface = None  # last RGB image made pygame surface

    self.camera_type = camera_type

    # initialize
    camera_actor.listen(self.parse_image)
    if self.camera_type == "rgb":
      self.camera_actor.intrinsic = compute_intrinsic(width, height, fov)

  # callback for sensor.listen()
  # this is called whenever a data comes from CARLA server
  def parse_image(self, image):
    """Process one camera captured image."""
    # parse the image data into a pygame surface for display or screenshot
    # raw image is BGRA
    # if image_type is segmentation, here will convert to the pre-defined color
    image.convert(self.image_type)

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # BGR -> RGB
    self.rgb_image = array
    self.pygame_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    self.last_image_seconds = image.timestamp
    self.last_image_frame_num = image.frame


def set_camera_fov(args_, client_, new_fov):
  """Replace the current camera actor with new camera actor with new fov."""

  args_.camera_bp.set_attribute("fov", "%s" % new_fov)
  args_.camera_depth_bp.set_attribute("fov", "%s" % new_fov)

  # destroy the original actor and make a new camera object
  args_.rgb_camera.camera_actor.stop()
  args_.depth_camera.camera_actor.stop()
  commands_ = [
      # destroy the previous actor first
      carla.command.DestroyActor(args_.depth_camera.camera_actor.id),
      carla.command.DestroyActor(args_.rgb_camera.camera_actor.id),
      # spawn the new actor
      carla.command.SpawnActor(
          args_.camera_bp, carla.Transform(), args_.spectator),
      carla.command.SpawnActor(
          args_.camera_depth_bp, carla.Transform(), args_.spectator),
  ]
  response_ = client_.apply_batch_sync(commands_)
  camera_actor_ids_ = [r.actor_id for r in response_[-2:]]
  camera_, camera_depth_ = world.get_actors(
      camera_actor_ids_)

  args_.rgb_camera = Camera(camera_, width=args_.width,
                            height=args_.height,
                            fov=new_fov,
                            camera_type="rgb")

  args_.depth_camera = Camera(
      camera_depth_, camera_type="depth")

  args_.prev_camera_fov = new_fov

"""
Camera control:
  r: reset camera transform to zeros
  n/m: zooming camera
  w/a/s/d/u/i/arrows: camera movements
  t: show current camera transform

Moment high-level:
  select moments: []
  toggle saving this moment: p
  save all moment or unsave all: o
  duplicate the current moment: l
  go to anchor view: v

Moment editing:
  selecting actor: ,.
  delete selected actor: backspace
  toggle showing static actors: space
  toggle showing traj: enter

Actor Trajectory Editing:
  delete the current last timestep: q
  click anywhere: add the new control point to the selected actor
  press: e  then click mouse will add new actor instead
  press: 1 to toggle between car and person new actor
  set all the person/vehicle control point is_stationary to be True: f/c

Play the moment: g
will block most controls until the moment is finished (except the camera
movement controls)

Annotation Related
  set the current agent as x agent: x
  delete the last destination: z

"""
def keyboard_control(args_, pygame_clock, world_, client_, runtime_bucket,
                     moment_data, saved_idx, global_actor_list):
  """Process all the keyboard event since last tick."""
  # the rgb camera and seg camera are in the args
  # since we may need to change the fov for them
  # by rebuild the camera actor

  # return True to exit
  ms_since_last_tick = pygame_clock.get_time()
  # TOdo: change the following. dont get the transform at each time to avoid
  # stuttering due to the frame lag
  # set a global variable of the rotation instead
  prev_rotation = args_.spectator.get_transform().rotation
  prev_location = args_.spectator.get_transform().location
  global_up_vector = carla.Vector3D(x=0, z=1, y=0)
  # a normalized x,y,z, between 0~1
  forward_vector = prev_rotation.get_forward_vector()
  left_vector = cross(forward_vector, global_up_vector)
  global_forward_vector = cross(global_up_vector, left_vector)

  # get all event from event queue
  # empty out the event queue
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      return True
    # click a point to add traj or add new actor
    elif event.type == pygame.MOUSEBUTTONUP:
      if runtime_bucket["playing_moment"]:
        continue
      # get the clicking xyz first
      # get the depth map in meters
      depth_in_meters = parse_carla_depth(args_.depth_camera.rgb_image)
      pos_x, pos_y = pygame.mouse.get_pos()
      click_point = np.array([pos_x, pos_y, 1])

      intrinsic = args_.rgb_camera.camera_actor.intrinsic
      # 3d point in the camera coordinates
      click_point_3d = np.dot(np.linalg.inv(intrinsic), click_point)
      click_point_3d *= depth_in_meters[pos_y, pos_x]
      # why? this is because unreal transform is (y, -z , x)???
      y, z, x = click_point_3d
      z = -z
      click_point_3d = np.array([x, y, z, 1])
      click_point_3d.reshape([4, 1])

      # transform to the world origin
      camera_rt = compute_extrinsic_from_transform(
          args_.rgb_camera.camera_actor.get_transform())
      click_point_world_3d = np.dot(camera_rt, click_point_3d)
      x, y, z = click_point_world_3d.tolist()[0][:3]  # since it is np.matrix
      xyz = [x, y, z + 0.1]  # slightly above ground

      if runtime_bucket["waiting_for_click"]:
        runtime_bucket["waiting_for_click"] = False
        # adding a new actor at the click location
        # or add a new destination for the x_agent

        new_actor_type = runtime_bucket["new_actor_type"]

        if new_actor_type == "destination":
          # check whether the currently selected guy is x_agent
          p_id = runtime_bucket["selected"]
          if p_id in moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"]:
            # check whether the destination make sense
            last_obs = runtime_bucket[
                "person_data"][runtime_bucket["selected"]][args.obs_length-1]
            last_obs_speed = last_obs["speed"]

            # meter / second
            # pred_length is 26 timestep, 10.4 seconds
            max_dest_dist = last_obs_speed * \
                (args.pred_length / args.annotation_fps)

            last_obs_xyz = last_obs["xyz"]
            diff = [(xyz[i] - last_obs_xyz[i])**2 for i in range(2)]  # ignore z
            diff = math.sqrt(sum(diff))
            if diff > max_dest_dist:
              print("Destination too far away. Dist to last obs: %s" % diff)
            else:
              this_moment_data = moment_data[runtime_bucket["cur_moment_idx"]]
              this_moment_data["x_agents"][p_id].append(xyz)
              print("Set destination #%s for person #%s. Dist to last obs: %s" \
                  % (len(this_moment_data["x_agents"][p_id]), p_id, diff))

        else:
          new_p_id = runtime_bucket["new_p_id"]
          new_frame_id = 0  # new actor always start from first frame?
          new_control_point = [new_p_id, new_frame_id, xyz, None, None, None,
                               False]

          add_new_control_point(
              moment_data, new_control_point, new_frame_id,
              runtime_bucket,
              is_vehicle=runtime_bucket["new_actor_type"] == "vehicle")

          # select the new guy
          runtime_bucket["selected"] = new_p_id
          runtime_bucket["is_vehicle"] = \
              runtime_bucket["new_actor_type"] == "vehicle"

          runtime_bucket["new_p_id"] += 1

      else:
        # check whether there is a selected actor first
        all_actors = sorted(runtime_bucket["person_data"].keys()) + \
            sorted(runtime_bucket["vehicle_data"].keys())
        if runtime_bucket["selected"] is not None and \
              (runtime_bucket["selected"] in all_actors):
          p_id = runtime_bucket["selected"]
          if not runtime_bucket["is_vehicle"]:
            traj_data = runtime_bucket["person_data"][p_id]
          else:
            traj_data = runtime_bucket["vehicle_data"][p_id]

          # the current last frame id
          last_frame_id = traj_data[-1]["frame_id"]
          new_frame_id = last_frame_id + args.frame_skip

          change_type = "ped_controls"
          if runtime_bucket["is_vehicle"]:
            change_type = "vehicle_controls"

          # modify the moment_data
          # a new control point meaning, we need to recompute the
          # direction vector for the last timestep, then add the new point
          # 1. recompute last control point
          prev_controls = \
              moment_data[runtime_bucket["cur_moment_idx"]][change_type]
          last_frame_id_str = "%s" % last_frame_id  # json's fault
          # find the control point idx to change
          last_control_point_idx = None
          for i, one in enumerate(prev_controls[last_frame_id_str]):
            if one[0] == p_id:
              last_control_point_idx = i
              break
          if last_control_point_idx is not None:
            direction_vector, speed, time_elasped = get_direction_and_speed(
                [new_frame_id, p_id] + xyz,
                [last_frame_id, p_id] + traj_data[-1]["xyz"],
                args.video_fps)
            _, ori_frame_id, _, _, _, _, is_stationary = \
                prev_controls[last_frame_id_str][last_control_point_idx]
            moment_data[runtime_bucket["cur_moment_idx"]][change_type][
                last_frame_id_str][last_control_point_idx] = \
                    [p_id, ori_frame_id, traj_data[-1]["xyz"],
                     direction_vector, speed, time_elasped, is_stationary]

          # 2. add the new point
          new_control_point = [p_id, new_frame_id, xyz, None,
                               None, None, False]
          add_new_control_point(
              moment_data, new_control_point, new_frame_id,
              runtime_bucket,
              is_vehicle=runtime_bucket["is_vehicle"])


    elif event.type == pygame.KEYUP:
      if event.key == pygame.K_r:
        # reset rotation
        args_.spectator.set_transform(carla.Transform(
            rotation=carla.Rotation(pitch=0.0,
                                    yaw=0.0,
                                    roll=0.0),
            location=prev_location))
      elif event.key == pygame.K_t:
        # print out the camera transform
        print(args_.spectator.get_transform())
        print("camera FOV: %s" % \
            args_.rgb_camera.camera_actor.attributes["fov"])
      # an ugly way to change the camera fov
      elif (event.key == pygame.K_n) or (event.key == pygame.K_m):
        if event.key == pygame.K_n:
          new_fov = args_.prev_camera_fov - 5.0
        else:
          new_fov = args_.prev_camera_fov + 5.0
        new_fov = new_fov if new_fov >= 5.0 else 5.0
        new_fov = new_fov if new_fov <= 175.0 else 175.0

        set_camera_fov(args_, client_, new_fov)


      # ------------ selecting moments
      elif (event.key == pygame.K_LEFTBRACKET) or \
          (event.key == pygame.K_RIGHTBRACKET):
        if runtime_bucket["playing_moment"]:
          continue
        get_idx = 1
        if event.key == pygame.K_LEFTBRACKET:
          get_idx = -1
        cur_moment_idx = runtime_bucket["cur_moment_idx"]
        cur_moment_idx += get_idx
        if cur_moment_idx >= len(moment_data):
          cur_moment_idx = 0
        elif cur_moment_idx < 0:
          cur_moment_idx = len(moment_data) - 1

        person_data, vehicle_data = init_moment(
        world_, client_, moment_data, cur_moment_idx,
        global_actor_list)
        runtime_bucket["person_data"] = person_data
        runtime_bucket["vehicle_data"] = vehicle_data
        runtime_bucket["selected"] = sorted(person_data.keys())[0]
        runtime_bucket["is_vehicle"] = False
        runtime_bucket["cur_moment_idx"] = cur_moment_idx

        # cleanup static actors
        all_actors = list(runtime_bucket["static_peds"].values()) + \
            list(runtime_bucket["static_vehicles"].values())
        client.apply_batch(
            [carla.command.DestroyActor(x) for x in all_actors])
        runtime_bucket["static_peds"] = {}
        runtime_bucket["static_vehicles"] = {}
        runtime_bucket["static_spawned"] = False

      # toggle saving this moment or not
      elif event.key == pygame.K_p:
        cur_moment_idx = runtime_bucket["cur_moment_idx"]
        #if saved_idxs.has_key(cur_moment_idx):
        if cur_moment_idx in saved_idxs:
          del saved_idxs[cur_moment_idx]
        else:
          saved_idxs[cur_moment_idx] = 1
      # toggle saving all moment or not
      elif event.key == pygame.K_o:
        # saving all
        if len(saved_idxs) < len(moment_data):
          for i in range(len(moment_data)):
            saved_idxs[i] = 1
        else:
          cur_saved_idxs = list(saved_idxs.keys())
          for i in cur_saved_idxs:
            del saved_idxs[i]
      elif event.key == pygame.K_v:
        # go the current scene's anchor camera setting
        scene = moment_data[runtime_bucket["cur_moment_idx"]]["scenename"]
        camera_location_preset = anchor_cameras[scene][0]
        args_.spectator.set_transform(camera_location_preset)
        set_camera_fov(args_, client_, anchor_cameras[scene][1])
      # copy the current moment to the end (so it does not affect saved_idxs)
      elif event.key == pygame.K_l:
        cur_moment = moment_data[runtime_bucket["cur_moment_idx"]].copy()
        moment_data.append(cur_moment)
        print("Copied current moment to idx %s" % (len(moment_data)-1))
      # toggle showing traj
      elif event.key == pygame.K_RETURN:
        if runtime_bucket["show_traj"]:
          runtime_bucket["show_traj"] = False
        else:
          runtime_bucket["show_traj"] = True
      # toggle showing the current static actors
      elif event.key == pygame.K_SPACE:
        if runtime_bucket["playing_moment"]:
          continue
        all_actors = list(runtime_bucket["static_peds"].values()) + \
            list(runtime_bucket["static_vehicles"].values())

        if all_actors:  # delete all static actors
          client.apply_batch(
            [carla.command.DestroyActor(x) for x in all_actors])
          runtime_bucket["static_peds"] = {}
          runtime_bucket["static_vehicles"] = {}
          # static_spawned is still true so no actor is generated
        else:
          runtime_bucket["static_spawned"] = False

      #  ---------------------- editing the moment
      # selecting actor
      elif (event.key == pygame.K_COMMA) or (event.key == pygame.K_PERIOD):
        get_idx = 1
        if event.key == pygame.K_COMMA:
          get_idx = -1
        select_new_actor(runtime_bucket, get_idx)

      # deleting an actor
      elif (event.key == pygame.K_BACKSPACE):
        if runtime_bucket["playing_moment"]:
          continue
        all_actors = sorted(runtime_bucket["person_data"].keys()) + \
            sorted(runtime_bucket["vehicle_data"].keys())
        if (len(runtime_bucket["person_data"]) == 1) and \
            not runtime_bucket["is_vehicle"]:
          print("Cannot delete the last person.")
        elif runtime_bucket["selected"] is not None and \
              (runtime_bucket["selected"] in all_actors):
          delete_p_id = runtime_bucket["selected"]
          delete_type = "ped_controls"
          if runtime_bucket["is_vehicle"]:
            delete_type = "vehicle_controls"

          # modify the moment_data
          this_moment_data = moment_data[runtime_bucket["cur_moment_idx"]]
          prev_controls = this_moment_data[delete_type]
          new_controls = {}
          for frame_id in prev_controls:
            temp = []
            for one in prev_controls[frame_id]:
              p_id, _, _, _, _, _, _ = one
              if p_id != delete_p_id:
                temp.append(one)
            new_controls[frame_id] = temp
          this_moment_data[delete_type] = new_controls
          # change the data directly
          moment_data[runtime_bucket["cur_moment_idx"]] = this_moment_data

          # delete the x_agents
          if delete_p_id in \
              moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"]:
            del moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"][
                delete_p_id]

          # delete the static guy
          check_static = runtime_bucket["static_peds"]
          if runtime_bucket["is_vehicle"]:
            check_static = runtime_bucket["static_vehicles"]
          if delete_p_id in check_static:
            check_static[delete_p_id].destroy()
            del check_static[delete_p_id]

          # get the new traj data
          person_data, vehicle_data = get_trajs(
              moment_data, runtime_bucket["cur_moment_idx"])
          runtime_bucket["person_data"] = person_data
          runtime_bucket["vehicle_data"] = vehicle_data

          # select a new guy
          select_new_actor(runtime_bucket, 1)

      # ----------------------------- editing the actor trajectory
      # toggle setting all points of the person/vehicle to is_stationary=True
      # except the first point.
      # this is useful for stationary person/vehicle in the scene
      elif (event.key == pygame.K_f) or (event.key == pygame.K_c):
        if runtime_bucket["playing_moment"]:
          continue
        set_all_is_stationary_to = True
        if event.key == pygame.K_c:
          set_all_is_stationary_to = False
        # check selected or not
        all_actors = sorted(runtime_bucket["person_data"].keys()) + \
            sorted(runtime_bucket["vehicle_data"].keys())
        if runtime_bucket["selected"] is not None and \
              (runtime_bucket["selected"] in all_actors):
          p_id = runtime_bucket["selected"]

          change_type = "ped_controls"
          if runtime_bucket["is_vehicle"]:
            change_type = "vehicle_controls"

          prev_controls = \
              moment_data[runtime_bucket["cur_moment_idx"]][change_type]
          # not changing the stationary status of the first frame
          first_frame = sorted([int(fid) for fid in prev_controls])[0]
          for frame_id in prev_controls:
            if frame_id != ("%s" % first_frame):  # json's fault
              for i, one in enumerate(prev_controls[frame_id]):
                if one[0] == p_id:
                  one[-1] = set_all_is_stationary_to
                  moment_data[runtime_bucket["cur_moment_idx"]][change_type][
                      frame_id][i] = one

          print("Set %s all traj's stationary to %s" % (
              p_id, set_all_is_stationary_to))
          # get the new traj data
          person_data, vehicle_data = get_trajs(
              moment_data, runtime_bucket["cur_moment_idx"])
          runtime_bucket["person_data"] = person_data
          runtime_bucket["vehicle_data"] = vehicle_data

      # press this then click to add new actor
      # toggle
      elif event.key == pygame.K_e:
        if runtime_bucket["waiting_for_click"]:
          runtime_bucket["waiting_for_click"] = False
        else:
          runtime_bucket["waiting_for_click"] = True
      # press this then toggle new actor car or person or destination
      elif event.key == pygame.K_1:
        types = ["person", "vehicle", "destination"]
        cur_idx = types.index(runtime_bucket["new_actor_type"])
        new_idx = cur_idx + 1
        if new_idx >= len(types):
          new_idx = 0
        runtime_bucket["new_actor_type"] = types[new_idx]
      # delete the current selected actor's last traj
      elif event.key == pygame.K_q:
        if runtime_bucket["playing_moment"]:
          continue
        all_actors = sorted(runtime_bucket["person_data"].keys()) + \
            sorted(runtime_bucket["vehicle_data"].keys())
        if runtime_bucket["selected"] is not None and \
              (runtime_bucket["selected"] in all_actors):
          delete_p_id = runtime_bucket["selected"]

          if not runtime_bucket["is_vehicle"]:
            traj_data = runtime_bucket["person_data"][delete_p_id]
          else:
            traj_data = runtime_bucket["vehicle_data"][delete_p_id]

          if len(traj_data) == 1:
            print("Cannot delete with only 1 timestep left.")
          else:
            # change the actual control data and recompute everything
            delete_frame_id = "%s" % traj_data[-1]["frame_id"]  # json's fault
            delete_type = "ped_controls"
            if runtime_bucket["is_vehicle"]:
              delete_type = "vehicle_controls"
            # todo: need to set the new last timestep to be None

            # modify the moment_data
            prev_controls = \
                moment_data[runtime_bucket["cur_moment_idx"]][delete_type]
            delete_idx = None
            for i, one in enumerate(prev_controls[delete_frame_id]):
              p_id, _, _, _, _, _, _ = one
              if p_id == delete_p_id:
                delete_idx = i
                break
            if delete_idx is not None:
              # Yikes.
              del moment_data[
                  runtime_bucket["cur_moment_idx"]][delete_type][
                      delete_frame_id][delete_idx]

            # get the new traj data
            person_data, vehicle_data = get_trajs(
                moment_data, runtime_bucket["cur_moment_idx"])
            runtime_bucket["person_data"] = person_data
            runtime_bucket["vehicle_data"] = vehicle_data
      elif event.key == pygame.K_g:
        # play the moment
        # always set to true
        # until the moment finished will automatically be false
        runtime_bucket["playing_moment"] = True

      # ------------- x agent setting

      # toggle setting the current agent as x agent
      elif event.key == pygame.K_x:
        if runtime_bucket["playing_moment"]:
          continue
        # check selected or not
        all_actors = sorted(runtime_bucket["person_data"].keys()) + \
            sorted(runtime_bucket["vehicle_data"].keys())
        if runtime_bucket["selected"] is not None and \
              (runtime_bucket["selected"] in all_actors):

          p_id = runtime_bucket["selected"]
          if runtime_bucket["is_vehicle"]:
            print("Cannot set vehicle as x_agent for now.")
            continue

          # unset x agent.
          if p_id in moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"]:
            print("Deleted person #%s as x_agent" % p_id)
            del moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"][p_id]
          else:
            # set the actor as x agent
            # need to check whether the trajectory is full length
            frame_ids = [o["frame_id"]
                         for o in runtime_bucket["person_data"][p_id]]
            if frame_ids[0] != 0:
              print("X_agent person needs to start from 0 frame")
              continue
            if len(frame_ids) < args.obs_length + args.pred_length:
              print("X_agent traj length need to be %s" % (
                  args.obs_length + args.pred_length))
              continue
            print("Set person #%s as x_agent" % p_id)
            moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"][p_id] = []
      # delete the last destination for this x_agent
      elif event.key == pygame.K_z:
        if runtime_bucket["playing_moment"]:
          continue
        this_moment_data = moment_data[runtime_bucket["cur_moment_idx"]]
        p_id = runtime_bucket["selected"]

        if p_id in this_moment_data["x_agents"]:
          if this_moment_data["x_agents"][p_id]:
            print("Deleted person #%s destination." % p_id)
            del this_moment_data["x_agents"][p_id][-1]

  # get a big dict of what key is pressed now, so to avoid hitting forward
  # multiple times to go forward for a distance
  step = 0.1 * ms_since_last_tick  # this is from experimenting
  keys = pygame.key.get_pressed()
  if keys[pygame.K_w]:
    args_.spectator.set_location(prev_location + step * global_forward_vector)
  if keys[pygame.K_s]:
    args_.spectator.set_location(prev_location - step * global_forward_vector)
  if keys[pygame.K_a]:
    args_.spectator.set_location(prev_location + step * left_vector)
  if keys[pygame.K_d]:
    args_.spectator.set_location(prev_location - step * left_vector)
  if keys[pygame.K_u]:
    args_.spectator.set_location(prev_location + step * 0.5 * global_up_vector)
  if keys[pygame.K_i]:
    args_.spectator.set_location(prev_location - step * 0.5 * global_up_vector)
  if keys[pygame.K_UP]:
    args_.spectator.set_transform(carla.Transform(
        rotation=carla.Rotation(pitch=prev_rotation.pitch + 1.0,
                                yaw=prev_rotation.yaw,
                                roll=prev_rotation.roll),
        location=prev_location))
  if keys[pygame.K_DOWN]:
    args_.spectator.set_transform(carla.Transform(
        rotation=carla.Rotation(pitch=prev_rotation.pitch - 1.0,
                                yaw=prev_rotation.yaw,
                                roll=prev_rotation.roll),
        location=prev_location))
  if keys[pygame.K_LEFT]:
    args_.spectator.set_transform(carla.Transform(
        rotation=carla.Rotation(pitch=prev_rotation.pitch,
                                yaw=prev_rotation.yaw - 1.0,
                                roll=prev_rotation.roll),
        location=prev_location))
  if keys[pygame.K_RIGHT]:
    args_.spectator.set_transform(carla.Transform(
        rotation=carla.Rotation(pitch=prev_rotation.pitch,
                                yaw=prev_rotation.yaw + 1.0,
                                roll=prev_rotation.roll),
        location=prev_location))

  return False


def add_new_control_point(moment_data, new_control_point, new_frame_id,
                          runtime_bucket, is_vehicle=False):
  """Add new control point of a person/vehicle to the moment data.
    Will also recompute the runtime_bucket's person/vehicle traj data
  """
  change_type = "ped_controls"
  if is_vehicle:
    change_type = "vehicle_controls"
  prev_controls = \
        moment_data[runtime_bucket["cur_moment_idx"]][change_type]
  new_frame_id = "%s" % new_frame_id  # json's fault
  #if prev_controls.has_key(new_frame_id):
  if new_frame_id in prev_controls:
    # add to the existing frame
    moment_data[runtime_bucket["cur_moment_idx"]][change_type][
        new_frame_id].append(new_control_point)
  else:
    # a new frame
    moment_data[runtime_bucket["cur_moment_idx"]][change_type][
        new_frame_id] = [new_control_point]

  # get the new traj data
  person_data, vehicle_data = get_trajs(
      moment_data, runtime_bucket["cur_moment_idx"])
  runtime_bucket["person_data"] = person_data
  runtime_bucket["vehicle_data"] = vehicle_data

def select_new_actor(runtime_bucket, get_idx):
  # all the current actor list
  # here we assume the person_id and vehicle id does not conflict
  # since they are track_id
  all_actors = sorted(runtime_bucket["person_data"].keys()) + \
      sorted(runtime_bucket["vehicle_data"].keys())
  if all_actors:
    # there is nothing selected currently
    if (runtime_bucket["selected"] is None) or \
        (runtime_bucket["selected"] not in all_actors):
      runtime_bucket["selected"] = all_actors[0]
      runtime_bucket["is_vehicle"] = len(
          runtime_bucket["person_data"].keys()) == 0
    else:
      cur_selected_idx = all_actors.index(runtime_bucket["selected"])
      cur_selected_idx += get_idx
      if cur_selected_idx >= len(all_actors):
        cur_selected_idx = 0
      elif cur_selected_idx < 0:
        cur_selected_idx = len(all_actors) - 1
      runtime_bucket["selected"] = all_actors[cur_selected_idx]
      runtime_bucket["is_vehicle"] = \
          cur_selected_idx >= len(runtime_bucket["person_data"].keys())

def xyz_to_carla(xyz):
  return carla.Location(x=xyz[0], y=xyz[1], z=xyz[2])

def plot_trajs_carla(world, traj_data, args, is_vehicle=False):
  vehicle_traj_color = ((0, 255, 255), (255, 0, 255))
  person_traj_color = ((255, 255, 0), (0, 255, 0))
  color = person_traj_color
  if is_vehicle:
    color = vehicle_traj_color
  red = carla.Color(r=255, g=0, b=0)

  for pid in traj_data:
    trajs = traj_data[pid]

    # show the person id at the beginning
    string = "Person #%s" % pid
    if is_vehicle:
      string = "Vehicle #%s" % pid

    world.debug.draw_string(
        xyz_to_carla(trajs[0]["xyz"]), string,
        draw_shadow=False, color=red,
        life_time=1.0/args.video_fps)

    # just a point:
    if len(trajs) == 1:
      frame_id = trajs[0]["frame_id"]
      # color for observation
      this_color = color[0]
      if frame_id >= args.moment_frame_ids[args.obs_length]:
        # color for prediction period
        this_color = color[1]
      this_color = carla.Color(
          r=this_color[0], g=this_color[1], b=this_color[2])
      world.debug.draw_point(
          xyz_to_carla(trajs[0]["xyz"]),
          color=this_color, size=0.1, life_time=1.0/args.video_fps)

    # assuming the trajectory is sorted in time
    for p1, p2 in zip(trajs[:-1], trajs[1:]):
      frame_id = p2["frame_id"]

      # color for observation
      this_color = color[0]
      if frame_id >= args.moment_frame_ids[args.obs_length]:
        # color for prediction period
        this_color = color[1]
      this_color = carla.Color(
          r=this_color[0], g=this_color[1], b=this_color[2])

      p1_xyz = xyz_to_carla(p1["xyz"])
      p2_xyz = xyz_to_carla(p2["xyz"])

      world.debug.draw_arrow(
          p1_xyz, p2_xyz,
          thickness=0.1,
          arrow_size=0.1, color=this_color, life_time=1.0/args.video_fps)

      if p2["is_stationary"]:
        world.debug.draw_point(
            p2_xyz, color=red, size=0.1, life_time=1.0/args.video_fps)


def spawn_actor(bps, xyz, world):
  bp = get_bp(bps)
  actor = world.try_spawn_actor(bp, carla.Transform(
      location=carla.Location(x=xyz[0],
                              y=xyz[1],
                              z=xyz[2])))
  return actor


def spawn_static(traj_data, static_dict, bps, frame_idx, world):
  for pid in traj_data:
    actor = spawn_actor(bps, traj_data[pid][frame_idx]["xyz"], world)
    if actor is None:
      print("Person/vehicle %s failed to spawn at frame %s" %
            (pid, frame_idx))
    else:
      static_dict[pid] = actor


def plot_actor_3d_bbox(world, actor, color, fps):
  color = carla.Color(r=color[0], g=color[1], b=color[2])
  # get the current transform (location + rotation)
  transform = actor.get_transform()
  # bounding box is relative to the actor
  bounding_box = actor.bounding_box
  bounding_box.location += transform.location  # from relative to world
  world.debug.draw_box(bounding_box, transform.rotation,
                       color=color, life_time=1.0/fps)


def get_trajs(moment_data, moment_idx):
  # person_id -> list of points
  person_data, _ = control_data_to_traj(
      moment_data[moment_idx]["ped_controls"])
  vehicle_data, _ = control_data_to_traj(
      moment_data[moment_idx]["vehicle_controls"])
  return person_data, vehicle_data


def init_moment(world, client, moment_data, moment_idx, global_actor_list):
  settings = world.get_settings()
  settings.synchronous_mode = True
  settings.fixed_delta_seconds = 1.0 / \
      moment_data[moment_idx]["static_scene"]["fps"]
  world.apply_settings(settings)
  setup_static(world, client, moment_data[moment_idx]["static_scene"],
               global_actor_list)
  person_data, vehicle_data = get_trajs(moment_data, moment_idx)

  return person_data, vehicle_data


if __name__ == "__main__":
  args = parser.parse_args()

  # compute the frame_ids for each moment
  args.frame_skip = int(args.video_fps / args.annotation_fps)  # 12/10
  args.moment_frame_ids = range(
      0, (args.obs_length + args.pred_length) * args.frame_skip,
      args.frame_skip)

  args.width, args.height = [int(x) for x in args.res.split("x")]

  args.rgb_camera = None
  args.depth_camera = None
  runtime_bucket = None

  with open(args.moment_json, "r") as f:
    moment_data = json.load(f)
  if not moment_data:
    print("Data is empty.")
    sys.exit()

  # reset all x_agents dict's key to int, since stupid json
  reset_x_agent_key(moment_data)

  # go through the whole thing to find the largest trackid for new actors
  # todo: remember this when generating the files
  max_p_id = -1
  for o in moment_data:
    for frame_id in o["ped_controls"]:
      for on in o["ped_controls"][frame_id]:
        max_p_id = max(on[0], max_p_id)
    for frame_id in o["vehicle_controls"]:
      for on in o["vehicle_controls"][frame_id]:
        max_p_id = max(on[0], max_p_id)

  # saved the moment idxs
  # currently you can add new moment by copying the current moment
  saved_idxs = {}

  # assuming the filename is the scene name
  #scene = os.path.splitext(os.path.basename(args.moment_json))[0].split(".")[0]
  scene = moment_data[0]["scenename"]
  camera_location_preset = anchor_cameras[scene][0]
  args.fov = anchor_cameras[scene][1]

  map_name = "Town03_ethucy"
  if args.is_actev:
    map_name = "Town05_actev"

  try:
    # record all the actor so we could clean them
    global_actor_list = []

    # connect to carla server
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.load_world(map_name)

    walker_bps = setup_walker_bps(world)
    vehicle_bps = setup_vehicle_bps(world)

    blueprint_library = world.get_blueprint_library()

    # the world will have a spectator already in it with actor id == 1
    spectator = world.get_spectator()

    pygame.init()

    # pygame screen
    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    # configure the rgb camera
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "%s" % args.width)
    camera_bp.set_attribute("image_size_y", "%s" % args.height)
    camera_bp.set_attribute("fov", "%s" % args.fov)
    # Set the time in seconds between sensor captures

    camera_bp.set_attribute("sensor_tick", "%s" % 0.0)
    camera_bp.set_attribute("enable_postprocess_effects", "true")
    # no motion blur
    camera_bp.set_attribute("motion_blur_intensity", "%s" % 0.0)

    # 2.2 is default, 1.5 is the carla default spectator gamma (darker)
    camera_bp.set_attribute("gamma", "%s" % 1.7)

    # depth camera
    camera_depth_bp = blueprint_library.find("sensor.camera.depth")
    camera_depth_bp.set_attribute("image_size_x", "%s" % args.width)
    camera_depth_bp.set_attribute("image_size_y", "%s" % args.height)
    camera_depth_bp.set_attribute("fov", "%s" % args.fov)
    camera_depth_bp.set_attribute("sensor_tick", "%s" % 0.0)


    # the rgb camera actor and the seg camera actor
    spawn_commands = [
        carla.command.SpawnActor(camera_bp, carla.Transform(), spectator),
        carla.command.SpawnActor(camera_depth_bp, carla.Transform(), spectator),
    ]
    response = client.apply_batch_sync(spawn_commands)
    camera_actor_ids = [x.actor_id for x in response]
    camera, camera_depth = world.get_actors(camera_actor_ids)

    rgb_camera = Camera(camera, width=args.width, height=args.height,
                        fov=args.fov, camera_type="rgb")

    depth_camera = Camera(camera_depth, camera_type="depth")

    # save this for changing fov
    args.camera_bp = camera_bp
    args.camera_depth_bp = camera_depth_bp
    args.prev_camera_fov = args.fov
    args.spectator = spectator
    args.rgb_camera = rgb_camera
    args.depth_camera = depth_camera

    spectator.set_transform(camera_location_preset)


    client_clock = pygame.time.Clock()

    #sign_bp = blueprint_library.find("static.prop.pergola")
    #sign_bp = blueprint_library.find("static.prop.streetsign")
    #sign = world.spawn_actor(sign_bp, carla.Transform(location=carla.Location(
    #    x=-17.803186866382454, y=-28.65965775909937, z=0.245477528999292)))
    #global_actor_list.append(sign)

    # some variables
    runtime_bucket = {
        "selected": None,
        "is_vehicle": False,
        "cur_moment_idx": 0,
        "person_data": None,
        "vehicle_data": None,

        "playing_moment": False,
        "playing_moment_fidx": 0,
        "moment_vars": {
            "cur_peds": {},
            "cur_ped_collisions": {},
            "cur_vehicles": {},
            "cur_vehicle_initial_forward_vector": {},
            "cur_vehicle_prev_yaw": {},
            "actorid2info": {},
            "local_actor_list": [],
        },

        "static_peds": {},
        "static_vehicles": {},
        "static_spawned": False,
        "waiting_for_click": False,  # flag to add new actor
        "new_actor_type": "person",
        "new_p_id": max_p_id + 100,  # the new actor's trackid start
        "show_traj": True,
    }
    person_data, vehicle_data = init_moment(
        world, client, moment_data, runtime_bucket["cur_moment_idx"],
        global_actor_list)
    runtime_bucket["person_data"] = person_data
    runtime_bucket["vehicle_data"] = vehicle_data
    runtime_bucket["selected"] = sorted(person_data.keys())[0]

    # this tick applies all the static stuff
    static_frame_id = world.tick()

    # one frame loop
    # moment_data and runtime_bucket will be changed based on keyboard input
    while True:
      # keyboard and mouse control
      if keyboard_control(
          args, client_clock, world, client, runtime_bucket, moment_data,
          saved_idxs, global_actor_list):
        break

      cur_moment_idx = runtime_bucket["cur_moment_idx"]

      assert runtime_bucket["person_data"], "person control for %s moment is" \
                                            " empty" % cur_moment_idx

      # keep the simulation run under this fps
      client_clock.tick_busy_loop(args.video_fps)


      if runtime_bucket["playing_moment"]:

        # delete the static actors at any time
        if runtime_bucket["static_spawned"]:
          actor_list = list(runtime_bucket["static_peds"].values()) + \
              list(runtime_bucket["static_vehicles"].values())
          client.apply_batch(
              [carla.command.DestroyActor(x) for x in actor_list])
          runtime_bucket["static_peds"], runtime_bucket["static_vehicles"] = \
              {}, {}
          runtime_bucket["static_spawned"] = False

        # compute the total frame to simulate for this moment

        # simluate from "playing_moment_fidx" 0 to the last frame (the exp max
        # or the current ped max)
        all_frame_ids = sorted([int(fidx) for fidx in moment_data[
            runtime_bucket["cur_moment_idx"]]["ped_controls"]])
        if (runtime_bucket["playing_moment_fidx"] > \
            args.moment_frame_ids[-1]) or (
                runtime_bucket["playing_moment_fidx"] > all_frame_ids[-1]):
          runtime_bucket["playing_moment"] = False
          runtime_bucket["playing_moment_fidx"] = 0
          # clean up and reset all the variables
          cleanup_actors(
              list(runtime_bucket["moment_vars"]["cur_peds"].values()) + \
              [x.sensor for x in runtime_bucket["moment_vars"][
                  "cur_ped_collisions"].values()] + \
              list(runtime_bucket["moment_vars"]["cur_vehicles"].values()),
              client)

          reset_bps(walker_bps)
          reset_bps(vehicle_bps)

          runtime_bucket["moment_vars"] = {
              "cur_peds": {},
              "cur_ped_collisions": {},
              "cur_vehicles": {},
              "cur_vehicle_initial_forward_vector": {},
              "cur_vehicle_prev_yaw": {},
              "actorid2info": {},
              "local_actor_list": [],
          }
        else:
          ped_controls = moment_data[runtime_bucket["cur_moment_idx"]][
              "ped_controls"]
          vehicle_controls = moment_data[runtime_bucket["cur_moment_idx"]][
              "vehicle_controls"]
          # grab the control data of this frame if any
          batch_cmds, sim_stats = run_sim_for_one_frame(
              "%s" % runtime_bucket["playing_moment_fidx"],  # json's fault
              ped_controls, vehicle_controls,
              runtime_bucket["moment_vars"]["cur_peds"],
              runtime_bucket["moment_vars"]["cur_ped_collisions"],
              runtime_bucket["moment_vars"]["cur_vehicles"],
              runtime_bucket["moment_vars"][
                  "cur_vehicle_initial_forward_vector"],
              runtime_bucket["moment_vars"][
                  "cur_vehicle_prev_yaw"],
              walker_bps, vehicle_bps,
              world,
              runtime_bucket["moment_vars"]["local_actor_list"],
              runtime_bucket["moment_vars"]["actorid2info"],
              show_traj=False, verbose=True,
              max_yaw_change=90, exit_if_spawn_fail=False)
          if batch_cmds:
            response = client.apply_batch_sync(batch_cmds)
          runtime_bucket["playing_moment_fidx"] += 1
      else:
        # not playing, then put the person on the trajectory start
        if not runtime_bucket["static_spawned"]:
          spawn_static(runtime_bucket["person_data"],
                       runtime_bucket["static_peds"], walker_bps,
                       0, world)
          spawn_static(runtime_bucket["vehicle_data"],
                       runtime_bucket["static_vehicles"],
                       vehicle_bps, 0, world)
          runtime_bucket["static_spawned"] = True
          reset_bps(walker_bps)
          reset_bps(vehicle_bps)

        # show the selected person
        if runtime_bucket["selected"] is not None:
          # 1. the static model
          check_static = runtime_bucket["static_peds"] \
                         if not runtime_bucket["is_vehicle"] \
                         else runtime_bucket["static_vehicles"]
          if runtime_bucket["selected"] in check_static:
            plot_actor_3d_bbox(world, check_static[runtime_bucket["selected"]],
                               (0, 0, 255), args.video_fps)
          # 2. just plot the big dot
          traj_data = runtime_bucket["person_data"] \
                      if not runtime_bucket["is_vehicle"] \
                      else runtime_bucket["vehicle_data"]
          world.debug.draw_point(
              xyz_to_carla(traj_data[runtime_bucket["selected"]][0]["xyz"]),
              color=carla.Color(b=255), size=0.1, life_time=1.0/args.video_fps)

          # continously draw the destination points for the current selected actor
          # if it is x_agent
          if runtime_bucket["selected"] in \
              moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"]:
            destinations = moment_data[runtime_bucket["cur_moment_idx"]][
                "x_agents"][runtime_bucket["selected"]]
            dest_color = carla.Color(r=255, g=100, b=0)
            # plot them as arrows to show the deleting order
            if len(destinations) == 1:
              world.debug.draw_point(
                  xyz_to_carla(destinations[0]),
                  color=dest_color, size=0.1, life_time=1.0/args.video_fps)

            for p1, p2 in zip(destinations[:-1], destinations[1:]):
              p1_xyz = xyz_to_carla(p1)
              p2_xyz = xyz_to_carla(p2)

              world.debug.draw_arrow(
                  p1_xyz, p2_xyz,
                  thickness=0.1,
                  arrow_size=0.1, color=dest_color,
                  life_time=1.0/args.video_fps)


      # continuously drawing the trajectory of the moment
      if runtime_bucket["show_traj"]:
        plot_trajs_carla(world, runtime_bucket["person_data"], args,
                         is_vehicle=False)
        plot_trajs_carla(world, runtime_bucket["vehicle_data"], args,
                         is_vehicle=True)

      # update camera image at the pygame screen
      if args.rgb_camera.pygame_surface is not None:
        display.blit(args.rgb_camera.pygame_surface, (0, 0))

      #  ------ show the current moment stats in text on the screen
      show_stats = {
          "moment": "%s/%s" % (cur_moment_idx + 1,
                               len(moment_data)),
          "is_saved": "%s / (%s)" % (
              cur_moment_idx in saved_idxs, len(saved_idxs)),
          "original_frame_id": \
              moment_data[cur_moment_idx]["original_start_frame_id"],
          "vehicle_spawn_failed": \
              moment_data[cur_moment_idx]["vehicle_spawn_failed"],
      }
      #if moment_data[cur_moment_idx].has_key("moment_id"):
      if "moment_id" in moment_data[cur_moment_idx]:
        show_stats["moment_id"] = moment_data[cur_moment_idx]["moment_id"]
      if runtime_bucket["playing_moment"]:
        show_stats["Blocking"] = True
      show_stats = ", ".join(
          #["%s: %s" % (k, v) for k, v in show_stats.iteritems()])
          ["%s: %s" % (k, v) for k, v in show_stats.items()])
      text_surface, text_offset = make_text_surface(show_stats, 0)
      display.blit(text_surface, (0, 0))
      traj_data = runtime_bucket["person_data"] \
                  if not runtime_bucket["is_vehicle"] \
                  else runtime_bucket["vehicle_data"]
      selected_data = traj_data[runtime_bucket["selected"]]
      show_stats = {
          "num_person": len(runtime_bucket["person_data"]),
          "num_veh": len(runtime_bucket["vehicle_data"]),
          "x": len(
              moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"]),
          "selected": "Person #%s" % runtime_bucket["selected"]
                      if not runtime_bucket["is_vehicle"]
                      else "Vehicle #%s" % runtime_bucket["selected"],
          "control_steps": len(selected_data),
          "frame_rage": "%s-%s" % (selected_data[0]["frame_id"],
                                   selected_data[-1]["frame_id"]),
      }
      if runtime_bucket["waiting_for_click"]:
        show_stats["next_click_new_actor"] = runtime_bucket["new_actor_type"]
      show_stats = ", ".join(
          #["%s: %s" % (k, v) for k, v in show_stats.iteritems()])
          ["%s: %s" % (k, v) for k, v in show_stats.items()])
      text_surface, text_offset2 = make_text_surface(show_stats, text_offset)
      display.blit(text_surface, (0, text_offset))
      # annotation info
      if runtime_bucket["selected"] in \
          moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"]:
        selected_traj = runtime_bucket["person_data"][
            runtime_bucket["selected"]]
        show_stats = {}
        if len(selected_traj) > args.obs_length:
          show_stats = {
              "Last_obs_speed": "%.4f" % \
                  selected_traj[args.obs_length-1]["speed"],
              "Max_dest_dist": "%.4f" % (
                  selected_traj[args.obs_length-1]["speed"] * (
                      args.pred_length / args.annotation_fps)),
          }

        show_stats["is_x_agent"] = "%s destinations" % len(
            moment_data[runtime_bucket["cur_moment_idx"]]["x_agents"][
                runtime_bucket["selected"]])
        show_stats = ", ".join(
            #["%s: %s" % (k, v) for k, v in show_stats.iteritems()])
            ["%s: %s" % (k, v) for k, v in show_stats.items()])
        text_surface, _ = make_text_surface(show_stats, text_offset2)
        display.blit(text_surface, (0, text_offset2))

      pygame.display.flip()

      server_frame_id = world.tick()

  finally:
    # save all the changes to the new json
    all_new_moments = []
    for i in saved_idxs:
      all_new_moments.append(moment_data[i])

    with open(args.new_moment_json, "w") as f:
      json.dump(all_new_moments, f)

    # finished, clean actors

    # destroy the camera actor separately
    if args.rgb_camera is not None and args.rgb_camera.camera_actor is not None:
      args.rgb_camera.camera_actor.stop()
      global_actor_list.append(args.rgb_camera.camera_actor)
    if args.depth_camera is not None and \
        args.depth_camera.camera_actor is not None:
      args.depth_camera.camera_actor.stop()
      global_actor_list.append(args.depth_camera.camera_actor)

    if runtime_bucket is not None:
      global_actor_list += list(runtime_bucket["static_peds"].values())
      global_actor_list += list(runtime_bucket["static_vehicles"].values())
    client.apply_batch(
        [carla.command.DestroyActor(x) for x in global_actor_list])

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    pygame.quit()
