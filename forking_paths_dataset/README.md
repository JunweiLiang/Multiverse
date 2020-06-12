
# The Forking Paths Dataset

Download the dataset according to [this](../README.md#the-forking-paths-dataset). This dataset is created based on CARLA 0.9.6. For CARLA/UE4 veterans, here are the resources listed in the rest of the sections for downloading: \[[CARLA_0.9.6_compiled](http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz) from [CARLA](https://github.com/carla-simulator/carla/releases/tag/0.9.6)\], \[[Our_edited_maps](https://next.cs.cmu.edu/multiverse/dataset/multiverse_maps_and_statics.tgz)\]

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

+ The scene semantic segmentation ground truth is encoded in MP4 videos. See [here](https://carla.readthedocs.io/en/0.9.6/cameras_and_sensors/) for scene classes and coloring. See [this code](code/get_frames_and_scene_seg.py) for examples of converting segmentation videos to Numpy format.

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

### Some basics
In order to simulate a **scenario** (in the code it is referred to as "moment") where a number of "Person" and "Vehicle" agents navigate in the scene for a period of time, the simulator needs to know exactly how to control each agent at each time frame. For "Person" agent the control means the direction and velocity. For "Vehicle" agent, we teleport them to the desire location and remove the physics simulation for simplicity (As of CARLA 0.9.6, it is not trivial to accurately convert direction and velocity to vehicle controls like throttling and steering. And teleporting looks smooth enough if we do it at every time frame.) So basically we need: 1. the **static map**, 2. the **full control records** of every agents at all time frames, for the simulation to run. To get human-annotated multi-future trajectories, the idea is to first recreate a plausible scenario that resembles the real-world, and then ask a human annotator to "drop-in" or "embody" a "Person" agent, and control such an agent to continue to a destination. The control record of the human annotator along with other agents' are saved as a JSON file. We leave multi-human simultaneous annotation to future work.

### Step 1, prepare and test the CARLA simulator
Get the CARLA simulator from [here](http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz) and our edited maps from [here](https://next.cs.cmu.edu/multiverse/dataset/multiverse_maps_and_statics.tgz) or by:
```
$ wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
$ wget https://next.cs.cmu.edu/multiverse/dataset/multiverse_maps_and_statics.tgz
```
Put the maps into the CARLA package:
```
$ mkdir CARLA_0.9.6/; cd CARLA_0.9.6/; tar -zxvf ../CARLA_0.9.6.tar.gz
$ cd ../; tar -zxvf multiverse_maps_and_statics.tgz;
$ cp multiverse_maps_and_statics/Town0* CARLA_0.9.6/CarlaUE4/Content/Carla/Maps/
$ cp multiverse_maps_and_statics/Road/Town05_TerrainNode_125.u* CARLA_0.9.6/CarlaUE4/Content/Carla/Static/Road/RoadsTown05/
$ cp multiverse_maps_and_statics/Vegetation/Town05_TerrainNode_125.u* CARLA_0.9.6/CarlaUE4/Content/Carla/Static/Vegetation/
```
Now you should be able to start a CARLA simulator server. The package should work out-of-the-box. I have tested it on RTX 2060/TITAN X/GTX 1080 TI GPU machine with Nvidia-430 or above drivers and on Ubuntu 16/18. Start CARLA server by:
```
$ cd CARLA_0.9.6/; ./CarlaUE4.sh -opengl -carla-port=23015
```
A spectator window should popup. Then open another terminal to test the new maps. Start an observer client with a PTZ camera to play around:
```
# install pygame
$ python code/spectator.py --port 23015  --change_map Town05_actev
```
Both windows should be switched to a new map. For full keyboard and mouse controls of the spectator window, see [here](code/spectator.py#L136). Now, press Ctr+C to exit both terminals.


### Step 2, get recreated scenarios
To add more human annotations, we start with recreated scenarios (We will talk about how to create scenarios from real-world videos or from scratch in the next section). Download the scenarios from [here](https://next.cs.cmu.edu/multiverse/dataset/multiverse_scenarios_v1.tgz) or by:

```
$ wget https://next.cs.cmu.edu/multiverse/dataset/multiverse_scenarios_v1.tgz
# sha256sum: f25a02f3a362c8e05823f17b20e5c12224be0559849f46ae3143abc1828f8051
```

We'll need to make two file lists since we use two different maps.
```
$ tar -zxvf multiverse_scenarios_v1.tgz
$ cd multiverse_scenarios_v1/
$ ls $PWD/0* > actev.lst
$ ls $PWD/eth.fixed.json $PWD/zara01.fixed.json $PWD/hotel.fixed.json > ethucy.lst
```

### Step 3, human annotation
Start the CARLA server in the background:
```
$ cd CARLA_0.9.6/; DISPLAY= ./CarlaUE4.sh -opengl -carla-port=23015
```

You can change the port to others. Now, open another terminal and:
```
$ python code/annotate_carla.py multiverse_scenarios_v1/actev.lst actev.junwei.json \
actev.junwei.log.json --video_fps 30.0 --annotation_fps 2.5 --obs_length 12 \
--pred_length 26 --is_actev --port 23015
```

Now a pygame window should pop up and there will be instructions in the window for annotators. For ETHUCY, change to `ethucy.lst` and `--video_fps 25.0` and remove `--is_actev`.


### Step 4, data cleaning
Suppose you have a couple of annotators and each of them generates a JSON file from Step 3, we need a file list of these JSONs and their annotator ID:
```
$ echo "$PWD/actev.junwei.json 27" >> actev_annotations.lst
...
```

Now, make a single "moment" record JSON:
```
$ python code/gen_moment_from_annotation.py multiverse_scenarios_v1/actev.lst \
actev_annotations.lst actev.final.json --video_fps 30.0 --annotation_fps 2.5 \
--obs_length 12 --pred_length 26
```

Now, before recording the final videos, we should clean the data by manually looking at all the annotated trajectories and remove outliers. Start the server if it is not running:
```
$ cd CARLA_0.9.6/; DISPLAY= ./CarlaUE4.sh -opengl -carla-port=23015
```

Start the "moment" editor client:
```
$ python code/moment_editor.py actev.final.json actev.final.checked.json \
 --video_fps 30.0 --is_actev --annotation_fps 2.5 --port 23015
```

Click "[" or "]" to cycle through the annotated trajectories. Click "g" to replay each annotated trajectory. Click "o" to approve all trajectories. See [here](code/moment_editor.py#L139) for full controls. Close the window and a new JSON file is saved to `actev.final.checked.json`

<div align="center">
  <div style="">
      <img src="../images/moment_editor_example1.gif" height="300px" />
  </div>
  <p style="font-weight:bold;font-size:0.9em;">
    You can use the moment editor to edit or even create from scratch a person-vehicle scenario. The vehicle movements will be interpolated in the final recordings so it will be much smoother. See more in <a href="https://youtu.be/MktcrwkbNC4" target="_blank">this video</a>.
  </p>
</div>

### Step 5, now that we have the annotations, we could record videos!

Now we can start recording videos and get ground truth annotations including bounding boxes and scene semantic segmentation.
```
$ python code/record_annotation.py --is_actev --res 1920x1080 --video_fps 30.0 \
 --annotation_fps 2.5 --obs_length 12 --pred_length 26 actev.final.checked.json \
 new_dataset --port 23015
```

For ETHUCY, remove `--is_actev` and change `--video_fps 25.0`. The recording is done in the background and 4 cameras are used simultaneously to record the simulation. The output folder should have the same structure as our released dataset.
