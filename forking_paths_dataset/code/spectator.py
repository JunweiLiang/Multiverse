# coding=utf-8
"""After starting a carla server off-sceen, use this to connect as spectator."""

from __future__ import print_function

import argparse
import cv2
import datetime
import json
import sys
import os
import glob
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla
import pygame

from utils import anchor_cameras
from utils import compute_intrinsic
from utils import cross
from utils import parse_carla_depth
from utils import compute_extrinsic_from_transform
from utils import get_degree_of_two_vectors
from utils import save_rgb_image
from utils import get_3d_bbox
from utils import get_2d_bbox
from utils import make_text_surface

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--res", default="1280x720")
parser.add_argument("--fov", type=float, default=90)
parser.add_argument("--gamma", type=float, default=1.7)

parser.add_argument("--hide_server_info", action="store_true",
                    help="hide server info so it doesn't block view.")

parser.add_argument("--save_screenshot_path", default="screenshot")
parser.add_argument("--save_video_path", default=None)
parser.add_argument("--save_seg_path", default=None)
parser.add_argument("--save_bbox_json", default=None)
parser.add_argument("--save_seg_as_img", action="store_true",
                    help="save the scene segmentation as image instead")

parser.add_argument("--set_weather", action="store_true")
parser.add_argument("--weather_night", action="store_true")
parser.add_argument("--weather_rain", action="store_true")

# actev needs map Town05_actev, eth/ucy is Town03_ethucy
parser.add_argument("--change_map", default=None,)

# Town03 (default)
parser.add_argument("--go_to_zara_anchor", action="store_true",
                    help="go to ZARA anchor camera location")
parser.add_argument("--go_to_eth_anchor", action="store_true",
                    help="go to ETH anchor camera location")
parser.add_argument("--go_to_hotel_anchor", action="store_true",
                    help="go to HOTEL anchor camera location")

# actev scenes
parser.add_argument("--go_to_0000_anchor", action="store_true",
                    help="go to scene 0000 anchor camera location")
parser.add_argument("--go_to_0400_anchor", action="store_true",
                    help="go to scene 0400 anchor camera location")
parser.add_argument("--go_to_0401_anchor", action="store_true",
                    help="go to scene 0401 anchor camera location")
parser.add_argument("--go_to_0500_anchor", action="store_true",
                    help="go to scene 0500 anchor camera location")

parser.add_argument("--go_to_scene", default=None,
                    help="check the 4 view camera parameters")
parser.add_argument("--go_to_camera_num", default=0, type=int)

class Camera(object):
  """Camera object to have a surface."""
  def __init__(self, camera_actor, save_path=None, camera_type="rgb",
               seg_save_img=False, image_type=carla.ColorConverter.Raw,
               width=None, height=None, fov=None, recording=False):
    self.camera_actor = camera_actor
    self.image_type = image_type

    self.last_image_frame_num = None  # the frame num of the image
    self.last_image_seconds = None  # the seconds since beginning of eposide?
    self.rgb_image = None  # last RGB image
    self.pygame_surface = None  # last RGB image made pygame surface

    self.recording = recording
    self.save_path = save_path

    self.camera_type = camera_type
    self.seg_save_img = seg_save_img  # whether to save segmentation as image

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

    # save rgb or seg feature to disk
    if self.save_path is not None:
      if self.recording:
        if (self.camera_type == "seg") and not self.seg_save_img:
          array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
          array = np.reshape(array, (image.height, image.width, 4))
          # the pixel class is in the red channel
          array = array[:, :, 2]  # [H, W]
          np.save(os.path.join(self.save_path, "%08d.npy" % image.frame), array)
        elif self.camera_type == "rgb":
          image.save_to_disk(
              os.path.join(self.save_path, "%08d.jpg" % image.frame))

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # BGR -> RGB
    self.rgb_image = array
    self.pygame_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    self.last_image_seconds = image.timestamp
    self.last_image_frame_num = image.frame


"""
keyboard control:
r: reset camera transform to zeros
t: print out current camera transform and fov
y: reset to the preset camera transform if any
p: save segmentation and rgb screenshot
b: start recording videos
x: toggle showing client side bbox
n/m: zooming camera
w/a/s/d/u/i/arrows: camera movements

mouse control:
click point: get the 3D point at the clicking location
"""
def keyboard_control(args_, pygame_clock, save_screenshot_path,
                     vehicles_, walkers_, world_, client_):
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

  # up_vector = cross(left_vector, forward_vector)

  # get all event from event queue
  # empty out the event queue
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      return True
    elif event.type == pygame.MOUSEBUTTONUP:
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
      red = carla.Color(r=255, g=0, b=0)
      green = carla.Color(r=0, g=255, b=0)
      blue = carla.Color(r=0, g=0, b=255)
      if args_.last_click_point is None:
        args_.last_click_point = carla.Location(x=x, y=y, z=z)
        print("Click origin location xyz: %s %s %s" % (x, y, z))
        world.debug.draw_point(
            args_.last_click_point, color=red, size=0.3, life_time=30)
      else:
        this_click_point = carla.Location(x=x, y=y, z=z)
        world.debug.draw_arrow(
            args_.last_click_point, this_click_point, thickness=0.2,
            arrow_size=0.2, color=red, life_time=30)
        # draw the (x=1.0, y=0.0) and (x=0.0, y=1.0) point translated to the
        # origin point, and compute the rotation degree needed
        ref_point = (1.0 + args_.last_click_point.x, 0 + args_.last_click_point.y)
        ref_vec = (ref_point[0] - args_.last_click_point.x, ref_point[1] - args_.last_click_point.y)
        this_vec = (x - args_.last_click_point.x, y - args_.last_click_point.y)
        angle_degree = get_degree_of_two_vectors(this_vec, ref_vec)
        # notice here the degree is negative.
        print("Click second location xyz: %s, Degree between (1, 0) "
              "and this vector: %s" % ([x, y, z], -angle_degree))
        world.debug.draw_arrow(
            args_.last_click_point, carla.Location(x=ref_point[0], y=ref_point[1], z=z),
            thickness=0.1,
            arrow_size=0.1, color=green, life_time=30)
        ref_point = (0.0 + args_.last_click_point.x, 1.0 + args_.last_click_point.y)
        ref_vec = (ref_point[0] - args_.last_click_point.x, ref_point[1] - args_.last_click_point.y)
        world.debug.draw_arrow(
            args_.last_click_point, carla.Location(x=ref_point[0], y=ref_point[1], z=z),
            thickness=0.1,
            arrow_size=0.1, color=blue, life_time=30)
        args_.last_click_point = None


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

      # back to preset location
      elif event.key == pygame.K_y:
        if args_.preset is not None:
          args_.spectator.set_transform(args_.preset)

      # save the image in camera_to_save to save_path
      elif event.key == pygame.K_p:
        # in asyn mode, the frame_number might be different for the two
        # cameras
        save_rgb_image(args_.seg_camera.rgb_image,
                       os.path.join(save_screenshot_path,
                                    "%08d.seg.jpg" %
                                    args_.seg_camera.last_image_frame_num))
        save_rgb_image(args_.rgb_camera.rgb_image,
                       os.path.join(save_screenshot_path,
                                    "%08d.rgb.jpg" %
                                    args_.rgb_camera.last_image_frame_num))

        print("Saved seg and rgb image to %s, framenum %s, timestamp %s, " % (
            save_screenshot_path, args_.rgb_camera.last_image_frame_num,
            args_.rgb_camera.last_image_seconds))

      # toggle saving to videos
      elif event.key == pygame.K_b:
        if not args_.rgb_camera.recording:
          print(
              "Start saving video frames to %s" %
              args_.rgb_camera.save_path)
          args_.rgb_camera.recording = True
          print(
              "\tScene segmentation to %s" %
              args_.seg_camera.save_path)
          args_.seg_camera.recording = True
        else:
          print("stop saving video frame & seg feature.")
          args_.rgb_camera.recording = False
          args_.seg_camera.recording = False

      elif event.key == pygame.K_x:
        # before recording the video, remember to hit this button to get all
        # actor bboxes
        # update the actors in the scene, like after running build moment

        # reset all the vehicle and walker actor list
        # the actual box will be computed in client loop
        vehicles_[:] = []
        walkers_[:] = []
        if args_.show_actor_box:
          args_.show_actor_box = False
          print("stop showing box.")
        else:
          args_.show_actor_box = True
          for v in world_.get_actors().filter('vehicle.*'):
            vehicles_.append(v)
          for w in world.get_actors().filter('walker.*'):
            walkers_.append(w)
          print("Start showing actor boxes: got %s vehicles, %s walkers." %
                (len(vehicles_), len(walkers_)))

      # an ugly way to change the camera fov
      elif (event.key == pygame.K_n) or (event.key == pygame.K_m):
        if event.key == pygame.K_n:
          new_fov = args.prev_camera_fov - 5.0
        else:
          new_fov = args.prev_camera_fov + 5.0
        new_fov = new_fov if new_fov >= 5.0 else 5.0
        new_fov = new_fov if new_fov <= 175.0 else 175.0

        prev_recording = args_.rgb_camera.recording

        args_.camera_bp.set_attribute("fov", "%s" % new_fov)
        args_.camera_seg_bp.set_attribute("fov", "%s" % new_fov)
        args_.camera_depth_bp.set_attribute("fov", "%s" % new_fov)

        # destroy the original actor and make a new camera object
        args_.rgb_camera.camera_actor.stop()
        args_.seg_camera.camera_actor.stop()
        args_.depth_camera.camera_actor.stop()
        commands_ = [
            # destroy the previous actor first
            carla.command.DestroyActor(args_.depth_camera.camera_actor.id),
            carla.command.DestroyActor(args_.seg_camera.camera_actor.id),
            carla.command.DestroyActor(args_.rgb_camera.camera_actor.id),
            # spawn the new actor
            carla.command.SpawnActor(
                args_.camera_bp, carla.Transform(), args_.spectator),
            carla.command.SpawnActor(
                args_.camera_seg_bp, carla.Transform(), args_.spectator),
            carla.command.SpawnActor(
                args_.camera_depth_bp, carla.Transform(), args_.spectator),
        ]
        response_ = client_.apply_batch_sync(commands_)
        camera_actor_ids_ = [r.actor_id for r in response_[-3:]]
        camera_, camera_seg_, camera_depth_ = world.get_actors(
            camera_actor_ids_)

        args_.rgb_camera = Camera(camera_, width=args_.width,
                                  height=args_.height, recording=prev_recording,
                                  fov=new_fov, save_path=args_.save_video_path,
                                  camera_type="rgb")

        args_.seg_camera = Camera(
            camera_seg_, save_path=args_.save_seg_path, camera_type="seg",
            seg_save_img=args_.save_seg_as_img, recording=prev_recording,
            image_type=carla.ColorConverter.CityScapesPalette)
        args_.depth_camera = Camera(
            camera_depth_, save_path=None, camera_type="depth",
            recording=prev_recording)

        args_.prev_camera_fov = new_fov

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


class Info(object):
  """Need an object to record some info."""
  def __init__(self):
    self.server_clock = pygame.time.Clock()
    self.server_fps = 0
    self.simulation_time = 0
    self.frame_num = 0

  def on_world_tick(self, timestamp):
    """Carla world.on_tick()."""
    self.server_clock.tick()
    self.server_fps = self.server_clock.get_fps()
    # the number of frame elapsed since the simulator launch
    self.frame_num = timestamp.frame
    # seconds after the simulator has been launch
    self.simulation_time = timestamp.elapsed_seconds


if __name__ == "__main__":
  args = parser.parse_args()

  args.show_actor_box = False

  args.last_click_point = None

  args.rgb_camera = None
  args.seg_camera = None
  args.depth_camera = None

  args.width, args.height = [int(x) for x in args.res.split("x")]


  # so try-finally so we always destroy the actors
  try:
    # record all the actor so we could clean them
    global_actor_list = []

    # connect to carla server
    client = carla.Client(args.host, args.port)

    client.set_timeout(2.0)

    if args.change_map is not None:
      #print(client.get_available_maps())
      world = client.load_world(args.change_map)
      print("Changed map.")
    else:
      world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # ---------------------- set weather to sunny
    weather = carla.WeatherParameters(
        cloudyness=0.0,
        precipitation=0.0,
        sun_altitude_angle=85.0)
    #from utils import realism_weather
    #weather = realism_weather
    """
    from utils import static_scenes
    weather = carla.WeatherParameters(
        cloudyness=static_scenes["0000"]["weather"]["cloudyness"],
        precipitation=static_scenes["0000"]["weather"]["precipitation"],
        sun_altitude_angle=static_scenes["0000"]["weather"]["sun_altitude_angle"],
        precipitation_deposits=static_scenes["0000"]["weather"]["precipitation_deposits"],
        wind_intensity=static_scenes["0000"]["weather"]["wind_intensity"],
        sun_azimuth_angle=static_scenes["0000"]["weather"]["sun_azimuth_angle"])
    """
    if args.weather_night:
      world.set_weather(carla.WeatherParameters.ClearSunset)
    elif args.weather_rain:
      weather = carla.WeatherParameters.HardRainNoon
      world.set_weather(weather)
      # preset rain does not work here but works in manual_control.py
    elif args.set_weather:
      world.set_weather(weather)


    # the world will have a spectator already in it with actor id == 1
    spectator = world.get_spectator()
    args.spectator = spectator

    pygame.init()

    # pygame screen
    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    # path to save all the frames
    if args.save_video_path is not None:
      if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)
    if args.save_seg_path is not None:
      if not os.path.exists(args.save_seg_path):
        os.makedirs(args.save_seg_path)

    # camera preset locations
    preset = None
    if args.go_to_zara_anchor:
      preset = anchor_cameras["zara01"][0]
      args.fov = anchor_cameras["zara01"][1]
    # ETH
    if args.go_to_eth_anchor:
      preset = anchor_cameras["eth"][0]
      args.fov = anchor_cameras["eth"][1]

    # HOTEL
    if args.go_to_hotel_anchor:
      preset = anchor_cameras["hotel"][0]
      args.fov = anchor_cameras["hotel"][1]

    # Actev
    # map is Town05_actev
    if args.go_to_0000_anchor:
      preset = anchor_cameras["0000"][0]
      args.fov = anchor_cameras["0000"][1]

    if args.go_to_0400_anchor:
      preset = anchor_cameras["0400"][0]
      args.fov = anchor_cameras["0400"][1]

    if args.go_to_0401_anchor:
      preset = anchor_cameras["0401"][0]
      args.fov = anchor_cameras["0401"][1]

    if args.go_to_0500_anchor:
      preset = anchor_cameras["0500"][0]
      args.fov = anchor_cameras["0500"][1]

    if args.go_to_scene is not None:
      from utils import recording_cameras
      preset = recording_cameras[args.go_to_scene][args.go_to_camera_num][0]
      args.fov = recording_cameras[args.go_to_scene][args.go_to_camera_num][1]

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

    # ----- setup a scene segmentation camera to save screen shot
    camera_seg_bp = blueprint_library.find(
        "sensor.camera.semantic_segmentation")
    camera_seg_bp.set_attribute("image_size_x", "%s" % args.width)
    camera_seg_bp.set_attribute("image_size_y", "%s" % args.height)
    camera_seg_bp.set_attribute("fov", "%s" % args.fov)
    camera_seg_bp.set_attribute("sensor_tick", "%s" % 0.0)

    # depth camera
    camera_depth_bp = blueprint_library.find("sensor.camera.depth")
    camera_depth_bp.set_attribute("image_size_x", "%s" % args.width)
    camera_depth_bp.set_attribute("image_size_y", "%s" % args.height)
    camera_depth_bp.set_attribute("fov", "%s" % args.fov)
    camera_depth_bp.set_attribute("sensor_tick", "%s" % 0.0)

    # save this camera bp for changing fov
    args.camera_bp = camera_bp
    args.camera_seg_bp = camera_seg_bp
    args.camera_depth_bp = camera_depth_bp
    args.prev_camera_fov = args.fov

    # the rgb camera actor and the seg camera actor
    spawn_commands = [
        carla.command.SpawnActor(camera_bp, carla.Transform(), spectator),
        carla.command.SpawnActor(camera_seg_bp, carla.Transform(), spectator),
        carla.command.SpawnActor(camera_depth_bp, carla.Transform(), spectator),
    ]
    response = client.apply_batch_sync(spawn_commands)
    camera_actor_ids = [x.actor_id for x in response]
    camera, camera_seg, camera_depth = world.get_actors(camera_actor_ids)

    # the object that save the camera image data every tick
    rgb_camera = Camera(camera, width=args.width, height=args.height,
                        fov=args.fov, save_path=args.save_video_path,
                        camera_type="rgb")

    seg_camera = Camera(
        camera_seg, save_path=args.save_seg_path, camera_type="seg",
        seg_save_img=args.save_seg_as_img,
        image_type=carla.ColorConverter.CityScapesPalette)
    depth_camera = Camera(
        camera_depth, save_path=None, camera_type="depth")


    # save this for changing fov
    args.rgb_camera = rgb_camera
    args.seg_camera = seg_camera
    args.depth_camera = depth_camera

    server_info = Info()
    world.on_tick(server_info.on_world_tick)

    args.preset = preset
    if preset is not None:
      spectator.set_transform(preset)

    # draw pedestrian waypoints
    # it is a location object
    #if args.draw_ped_point:
    #  for i in xrange(100):
    #    point = world.get_random_location_from_navigation()
    #    world.debug.draw_string(point, 'O', draw_shadow=False,
    #                            color=carla.Color(r=255, g=0, b=0),
    #                            life_time=120.0, persistent_lines=True)


    # other actor in the scene, so we could use them to draw bbox
    vehicles = []
    walkers = []

    bbox_data = {}  # frame -> bbox, class

    client_clock = pygame.time.Clock()
    # game loop
    while True:
      # same as tick(60), keep the loop run under 60 fps
      client_clock.tick_busy_loop(30)

      # keyboard and mousse control
      exit_now = keyboard_control(
          args, client_clock,
          args.save_screenshot_path,
          vehicles, walkers, world, client)

      if exit_now:
        break

      # update camera image at the pygame screen
      if args.rgb_camera.pygame_surface is not None:
        display.blit(args.rgb_camera.pygame_surface, (0, 0))

      # draw bounding box for all walker
      bb_surface = pygame.Surface((args.width, args.height))
      bb_surface.set_colorkey((0, 0, 0))  # white
      this_frame_bboxes = []
      for i, actor in enumerate(vehicles + walkers):
        if i < len(vehicles):
          class_name = "Vehicle"
        else:
          class_name = "Person"
        # [8, 3], last dim is depth,
        bbox_3d = get_3d_bbox(actor, args.rgb_camera.camera_actor)
        bbox = get_2d_bbox(bbox_3d, args.width, args.height)
        # all point in front of the camera
        if bbox is not None:

          if args.rgb_camera.recording:
            this_frame_bboxes.append({
                "bbox": bbox,
                "class_name": class_name,
                "track_id": actor.id,
            })
          if args.show_actor_box:
            # last param is line width
            pygame.draw.rect(bb_surface, (0, 255, 0), bbox, 2)
      if args.show_actor_box:
        display.blit(bb_surface, (0, 0))

      if this_frame_bboxes:
        if args.rgb_camera.last_image_frame_num:
          bbox_data[int(args.rgb_camera.last_image_frame_num)] = \
              this_frame_bboxes

      # show server info
      if not args.hide_server_info:
        # put text on the screen over the camera image
        camera_fov = args.rgb_camera.camera_actor.attributes["fov"]
        text_surface1, text_offset = make_text_surface(
            "Camera location: x: %s, y: %s, z: %s, fov: %s" %
            (spectator.get_location().x, spectator.get_location().y,
             spectator.get_location().z, camera_fov), 0)
        display.blit(text_surface1, (0, 0))

        text_surface2, text_offset2 = make_text_surface(
            "Camera rotation: pitch: %s, roll: %s, yaw: %s" % (
                spectator.get_transform().rotation.pitch,
                spectator.get_transform().rotation.roll,
                spectator.get_transform().rotation.yaw), text_offset)
        display.blit(text_surface2, (0, text_offset))

        # time and other info
        text_surface3, text_offset3 = make_text_surface(
            "Server time: %s, frame num: %s,"
            " server fps: %.4f, client fps: %.4f" % (
                datetime.timedelta(seconds=int(server_info.simulation_time)),
                server_info.frame_num,
                server_info.server_fps,
                client_clock.get_fps()), text_offset2)

        display.blit(text_surface3, (0, text_offset2))

      pygame.display.flip()

  finally:
    # save the bbox into json if asked
    if args.save_bbox_json is not None:
      with open(args.save_bbox_json, "w") as f:
        json.dump(bbox_data, f)
      print("saved bbox json with %s frame of data" % (len(bbox_data)))
    # finished, clean actors
    # destroy the camera actor separately
    if args.rgb_camera is not None and args.rgb_camera.camera_actor is not None:
      args.rgb_camera.camera_actor.stop()
      global_actor_list.append(args.rgb_camera.camera_actor)
    if args.seg_camera is not None and args.seg_camera.camera_actor is not None:
      args.seg_camera.camera_actor.stop()
      global_actor_list.append(args.seg_camera.camera_actor)
    if args.depth_camera is not None and \
        args.depth_camera.camera_actor is not None:
      args.depth_camera.camera_actor.stop()
      global_actor_list.append(args.depth_camera.camera_actor)

    client.apply_batch(
        [carla.command.DestroyActor(x) for x in global_actor_list])

    pygame.quit()
