
# The Forking Paths Dataset

Download the dataset according to [this](../README.md#the-forking-paths-dataset).

## Annotations
This dataset is for multi-future trajectory prediction. The common experiment setting for trajectory prediction is to let the model observe a period of time (3.2 seconds) and then predict the future 4.8 seconds. We set the observation time period in this dataset to be 4.8 seconds and the future period to be up to 10.4 seconds. For videos of the same scenario (we call it a "moment" in the code), the observation period would be the same and the future period would be different. See Section 4 in the paper for more details.

+ The dataset includes 3000 videos. The filenames are in [scene]\_[moment_id]\_[controlled_agent_id]\_[destination_idx]\_[annotator_id]\_[camera_idx] format. There are 508 unique observation trajectories, which are identified by [scene]\_[moment_id]\_[controlled_agent_id]\_[camera_idx] \(so the destination and annotator variances make the multi-future trajectories\).

+ The bounding box annotations are in JSON format, which include annotations for every agent for every time frame. FPS for ETH/UCY scene (zara01, eth, hotel) is 25.0 and the rest is 30.0. The JSON files look like this:
```
[
  {
    "class_name": "Person",  # either "Person" or "Vehicle"
    "is_x_agent": 1,  # whether it is a controlled agent
    "bbox": [1535.1, 55.443, 32.5920000000001, 33.919],  # [x, y, w, h]
    "frame_id": 0,  # 0-indexed time frame index.
    "track_id": 9.0,  # ID of the person/vehicle
  },
  ...
]
```

+ The scene semantic segmentation ground truth is encoded in MP4 videos. See [here](https://carla.readthedocs.io/en/0.9.6/cameras_and_sensors/) for scene classes and coloring.

## Prepare Data
Here are instructions to prepare data for testing like the [Next-Prediction](https://github.com/google/next-prediction) repo.

+ Assuming you have the dataset in this folder
```
$ tar -zxvf ForkingPaths_dataset_v1.tgz
```

+ Step 1, get data split (in the paper we use the whole Forking Paths dataset for testing. The model is trained on VIRAT/ActEV dataset, same as [Next-Prediction](https://github.com/google/next-prediction)).
```
$ python code/get_split_path.py next_x_v1_dataset/rgb_videos/ \
next_x_v1_dataset/data_splits
```

+ Step 2, get trajectories and bounding boxes in format that is compatible with the [Next-Prediction](https://github.com/google/next-prediction) repo and the [Social-GAN](https://github.com/agrimgupta92/sgan) repo.
```
$ python code/get_prepared_data_multifuture.py next_x_v1_dataset/ \
next_x_v1_dataset/data_splits/ next_x_v1_dataset_prepared_data/obs_data \
next_x_v1_dataset_prepared_data/multifuture
```
`next_x_v1_dataset_prepared_data/obs_data/traj_2.5fps` will contain trajectory
files that are inputs to Social-GAN.

+ Step 3, extract RGB frames from the videos and convert scene segmentation videos to Numpy format.

```
$ python code/get_frames_and_scene_seg.py next_x_v1_dataset_prepared_data/obs_data/traj_2.5fps/ \
next_x_v1_dataset/rgb_videos/ next_x_v1_dataset/seg_videos/ \
next_x_v1_dataset_prepared_data/obs_data/rgb_frames \
next_x_v1_dataset_prepared_data/obs_data/scene_seg/ \
next_x_v1_dataset_prepared_data/obs_data/bad_video.lst \
--scene_h 36 --scene_w 64 --is_multifuture
```

Now you can follow [this](../TESTING.md) to run multi-future inferencing with the Multiverse model. Optionally, to run [Next-Prediction](https://github.com/google/next-prediction) on this dataset, you can follow [this](https://github.com/JunweiLiang/next-prediction/blob/master/code/prepare_data/README.md#step-5-get-person-appearance-features) to extract person appearance features based on the RGB frames and person bounding boxes we get from Step 2 and 3.

## Visualize the Dataset
To get the cool multi-future visualization like the following, run this:
```
$ python code/visualize_multifuture_dataset.py next_x_v1_dataset/rgb_videos/ \
 next_x_v1_dataset/bbox/ next_x_v1_dataset_prepared_data/multifuture/test/ \
 multifuture_visualization
```
<div align="center">
  <div style="">
      <img src="../images/zara01_1_252_cam1.gif" height="300px" />
  </div>
</div>


## Record More Annotations
In this section, I will show you how to use our code and [CARLA](https://carla.org) simulator to get more human annotations like the following.

<div align="center">
  <div style="">
      <img src="../images/carla_annotation_multifuture_interface.gif" height="300px" />
  </div>
  <p style="font-weight:bold;font-size:0.9em;">
    This is the human annotation procedure. Annotators are provided with a bird-eye view of the scene first and then they are asked to control the agent to go to the destination. More to see in <a href="https://youtu.be/FJTJquN2Kj4" target="_blank">this video</a>
  </p>
</div>


