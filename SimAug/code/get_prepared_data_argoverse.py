# coding=utf-8
# get prepare data from argoverse dataset into trajectory, person_box, other_box
import argparse
import cv2
import os
import operator
import json
import pickle
import numpy as np

from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("datapath", help="under this would be the video_ids folder")
parser.add_argument("newframepath")
parser.add_argument("outpath")

class2classid = {
    "VEHICLE": 1,
    "PEDESTRIAN": 0,
    "ON_ROAD_OBSTACLE": 3,
    "LARGE_VEHICLE": 1,
    "BICYCLE": 8,
    "BICYCLIST": 8,
    "BUS": 1,
    "OTHER_MOVER": 3,
    "TRAILER": 1,
    "MOTORCYCLIST": 8,
    "MOPED": 8,
    "MOTORCYCLE": 8,
    #"STROLLER": 8,
    "EMERGENCY_VEHICLE": 1,
    #"ANIMAL": 14,
    #"WHEELCHAIR": 15,
    "SCHOOL_BUS": 1,
}

def get_center(bbox):
  x1, y1, x2, y2 = bbox
  return (x1 + x2)/2.0, (y1 + y2)/2.0

def find_closest(frame_file, label_files):
  label_int_idxs = [os.path.splitext(os.path.basename(one))[0]
                    for one in label_files]
  label_int_idxs = sorted([int(one.split("_")[-1]) for one in label_int_idxs])
  label_int_idxs = np.array(label_int_idxs)

  frame_int = int(
      os.path.splitext(os.path.basename(frame_file))[0].split("_")[-1])

  closest_ind = np.argmin(np.absolute(label_int_idxs - frame_int))
  closest_label_file = label_files[closest_ind]

  return closest_label_file


from scipy.spatial.transform import Rotation
# these are from argoverse dataset API
def get_2d_box_from_3d_box(label, camera_params):

  def transform_point_cloud(point_cloud, transform_matrix):
    num_pts = point_cloud.shape[0]
    homogeneous_pts = np.hstack([point_cloud, np.ones((num_pts, 1))])
    transformed_point_cloud = homogeneous_pts.dot(transform_matrix.T)
    return transformed_point_cloud[:, :3]

  def get_3d_points(label):
    tr_x = label["center"]["x"]
    tr_y = label["center"]["y"]
    tr_z = label["center"]["z"]
    translation = np.array([tr_x, tr_y, tr_z])

    rot_w = label["rotation"]["w"]
    rot_x = label["rotation"]["x"]
    rot_y = label["rotation"]["y"]
    rot_z = label["rotation"]["z"]
    quaternion = np.array([rot_w, rot_x, rot_y, rot_z])
    quaternion = quat2rotmat(quaternion)

    transform_matrix = get_transform_matrix(quaternion, translation)

    length = label["length"]
    width = label["width"]
    height = label["height"]

    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners_object_frame = np.vstack((x_corners, y_corners, z_corners)).T

    return transform_point_cloud(corners_object_frame, transform_matrix)


  # [8, 3]  # all 8 box corners
  points_3d = get_3d_points(label)
  num_pts = points_3d.shape[0]
  # [4, 8]
  points_3d_h = np.hstack([points_3d, np.ones((num_pts, 1))]).T

  R = camera_params.extrinsic[:3, :3]
  t = camera_params.extrinsic[:3, 3]
  transform_matrix = get_transform_matrix(R, t)

  points_egovehicle = points_3d_h.T[:, :3]
  # [8, 3] # coordinates in the camera's coordinate system
  uv_cam = transform_point_cloud(points_egovehicle, transform_matrix)

  # [8, 3]
  # project to the camera image plane
  # last dim negative means behind the camera
  uvh = proj_cam_to_uv(uv_cam, camera_params)

  # [4]
  box_2d = get_2d_bbox(uvh, camera_params.img_width, camera_params.img_height)

  return box_2d

def proj_cam_to_uv(uv_cam, camera_params):
  num_points = uv_cam.shape[0]
  uvh = np.zeros((num_points, 3))
  # (x_transformed_m, y_transformed_m, z_transformed_m)

  for idx in range(num_points):
    x_transformed_m = uv_cam[idx, 0]
    y_transformed_m = uv_cam[idx, 1]
    z_transformed_m = uv_cam[idx, 2]

    z_transformed_fixed_m = z_transformed_m

    # If we're behind the camera, z value (homogeneous coord w in image plane)
    # will be negative. If we are on the camera, there would be division by zero
    # later. To prevent that, move an epsilon away from zero.

    Z_EPSILON = 1.0e-4
    if np.absolute(z_transformed_m) <= Z_EPSILON:
        z_transformed_fixed_m = np.sign(z_transformed_m) * Z_EPSILON

    pinhole_x = x_transformed_m / z_transformed_fixed_m
    pinhole_y = y_transformed_m / z_transformed_fixed_m

    K = camera_params.intrinsic
    u_px = K[0, 0] * pinhole_x + K[0, 1] * pinhole_y + K[0, 2]

    v_px = K[1, 1] * pinhole_y + K[1, 2]

    uvh[idx] = np.array([u_px, v_px, z_transformed_m])

  #uv = uvh[:, :2]
  return uvh


def quat2rotmat(q):
  assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
  w, x, y, z = q
  q_scipy = np.array([x, y, z, w])
  return Rotation.from_quat(q_scipy).as_dcm()

def get_transform_matrix(rotation, translation):
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = translation
    return transform_matrix

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
    if x2 < 0 or y2 < 0:
      return None
    if x1 < 0:
      x1 = 0
    if y1 < 0:
      y1 = 0
    if x2 > max_w:
      x2 = max_w
    if y2 > max_h:
      y2 = max_h
    return [x1, y1, x2, y2]
  else:
    return None

class Camera_params:
  def __init__(self, camera_config):
    all_camera_data = camera_config["camera_data_"]
    for camera_data in all_camera_data:
        if "image_raw_ring_front_center" in camera_data["key"]:
            camera_config = camera_data["value"]
            break
    vehicle_SE3_sensor = camera_config["vehicle_SE3_camera_"]
    egovehicle_t_camera = np.array(vehicle_SE3_sensor["translation"])
    egovehicle_q_camera = vehicle_SE3_sensor["rotation"]["coefficients"]
    egovehicle_R_camera = quat2rotmat(egovehicle_q_camera)

    extrinsic = get_transform_matrix(
        egovehicle_R_camera.T, egovehicle_R_camera.T.dot(-egovehicle_t_camera))

    intrinsic_matrix = np.zeros((3, 4))
    intrinsic_matrix[0, 0] = camera_config["focal_length_x_px_"]
    intrinsic_matrix[0, 1] = camera_config["skew_"]
    intrinsic_matrix[0, 2] = camera_config["focal_center_x_px_"]
    intrinsic_matrix[1, 1] = camera_config["focal_length_y_px_"]
    intrinsic_matrix[1, 2] = camera_config["focal_center_y_px_"]
    intrinsic_matrix[2, 2] = 1.0

    img_width = 1920.0
    img_height = 1200.0

    distortion_coef = camera_config["distortion_coefficients_"]

    self.extrinsic = extrinsic
    self.intrinsic = intrinsic_matrix
    self.img_height = img_height
    self.img_width = img_width
    self.distortion_coef = distortion_coef

if __name__ == "__main__":
  args = parser.parse_args()

  video_paths = glob(os.path.join(args.datapath, "*"))

  # we will clipping the original image from 1920x1200 to 1920x1080
  clip_height = 120.0
  drop_frame = 12  # original is 30 fps
  target_resolution = (1920.0, 1080.0)

  def clip_box(box):
    x1, y1, x2, y2 = box
    y1 -= clip_height
    y2 -= clip_height
    y1 = 0.0 if y1 < 0.0 else y1
    y2 = 0.0 if y2 < 0.0 else y2
    return [x1, y1, x2, y2]

  for video_path in tqdm(video_paths):
    video_id = video_path.strip("/").split("/")[-1]

    track_labels = os.path.join(video_path, "per_sweep_annotations_amodal", "*")
    track_labels = sorted(glob(track_labels))

    rgb_frames = os.path.join(video_path, "ring_front_center", "*")
    rgb_frames = sorted(glob(rgb_frames))

    camera_json = os.path.join(
        video_path, "vehicle_calibration_info.json")
    with open(camera_json) as f:
      camera_json = json.load(f)
    camera_params = Camera_params(camera_json)

    # 1. collect the tracklet info
    anno_data = []  # track_id, box, frame_id
    trackid_mapping = {}  # we will have new track_id

    for frame_idx, rgb_frame in enumerate(rgb_frames):
      track_label = find_closest(rgb_frame, track_labels)

      with open(track_label) as f:
        labels = json.load(f)
      for label in labels:
        classname = label["label_class"]

        track_uuid = label["track_label_uuid"]
        if track_uuid not in trackid_mapping:
          trackid_mapping[track_uuid] = len(trackid_mapping)
        track_id = trackid_mapping[track_uuid]

        if "occlusion" in label:
          occlusion = label["occlusion"]
        else:
          occlusion = 0
        if occlusion == 100:
          continue

        # [x1, y1, x2, y2]
        track_box = get_2d_box_from_3d_box(label, camera_params)
        if track_box is None:  # box is behind camera
          continue

        track_box = clip_box(track_box)

        anno_data.append([track_id, track_box, frame_idx, classname])

    # 2. get the needed frameIdx after dropping some (original 30 fps)
    frame_idxs = {}
    for one in anno_data:
      if one[-1] == "PEDESTRIAN":
        frame_idxs[int(one[-2])] = 1
    frame_idxs = sorted(frame_idxs.keys())
    needed_frame_idxs = frame_idxs[::drop_frame]
    if len(needed_frame_idxs) < 8 + 12:
      print("warning, %s video has only %s frames, skipped.." % (
          video_id, len(frame_idxs)))
      continue

    # 3. save all the data into # frame_idx -> data
    frame_data = {}
    for one in anno_data:
      track_id, (x1, y1, x2, y2), frame_idx, classname = one
      if frame_idx not in needed_frame_idxs:
        continue
      if classname not in class2classid:  # ignore some classes
        continue
      if frame_idx not in frame_data:
        frame_data[frame_idx] = []
      frame_data[frame_idx].append({
          "class_name": classname,
          "track_id": track_id,
          "bbox": [x1, y1, x2, y2]
      })

    # 4. gather data for each frame_idx, each person_idx
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
        if class_name == "PEDESTRIAN":
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

    # save the data
    # 1. clip the frame
    target_frame_path = os.path.join(args.newframepath, video_id)
    if not os.path.exists(target_frame_path):
      os.makedirs(target_frame_path)
    for frame_idx in needed_frame_idxs:
      ori_frame_file = rgb_frames[frame_idx]
      im = cv2.imread(ori_frame_file, cv2.IMREAD_COLOR)
      im = im[int(clip_height):, :]

      cv2.imwrite(os.path.join(
          target_frame_path, "%s_F_%08d.jpg" % (video_id, frame_idx)),
                  im)

    # 2. save the annotations
    traj_path = os.path.join(args.outpath, "traj_2.5fps", "test")
    person_box_path = os.path.join(args.outpath, "anno_person_box", "test")
    other_box_path = os.path.join(args.outpath, "anno_other_box", "test")
    if not os.path.exists(traj_path):
      os.makedirs(traj_path)
    if not os.path.exists(person_box_path):
      os.makedirs(person_box_path)
    if not os.path.exists(other_box_path):
      os.makedirs(other_box_path)

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
