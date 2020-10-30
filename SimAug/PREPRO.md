# Preprocessing for SDD/Argoverse

These instructions create `data_*.npz` files for SDD and Argoverse.

## SDD

First download the videos from the [website](https://cvgl.stanford.edu/projects/uav_data/).

1. Resize and rotate videos to 1920x1080 and remember the changes. Need ffmpeg.
```
# assuming all SDD videos are under `videos` folder. Get the videos into a list first
$ find $PWD/videos -name "*.mov" > all_videos.lst
$ python code/resize_rotate_sdd.py all_videos.lst resized_videos resized.lst
```

2. Get the random 5-fold data split used in the paper. We'll just use 1 fold.
```
$ find $PWD/resized_videos -name "*.mp4" > resized_videos.lst
$ wget https://next.cs.cmu.edu/data/sdd_data_splits_eccv2020.tgz
$ tar -zxvf sdd_data_splits_eccv2020.tgz
```

You can also generate this yourself.
```
$ python code/get_sdd_splits.py resized_videos.lst data_splits --n_fold 5
```

3. Prepare data of trajectory and boxes based on rotation and resize
```
$ python code/get_prepared_data_sdd.py annotations/ data_splits/fold_1 resized.lst prepared_data_fold1
```

4. Get video frames (need opencv)
```
$ python code/get_frames_sdd.py resized_videos.lst prepared_data_fold1/traj_2.5fps/ \
resized_videos_frames --use_2level
```

Optionally, you can visualize the annotations:
```
$ python code/visualize_sdd_annotation.py prepared_data_fold1/ resized_videos_frames \
 vis_gt --vis_num_frame_per_video 3
```

5. Get scene segmentation features in npy files:
```
# download the deeplab ADE20k model
$ wget http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz

$ tar -zxvf deeplabv3_xception_ade20k_train_2018_05_29.tar.gz

$ find $PWD/resized_videos_frames -name "*.jpg" > resized_videos_frames.lst

$ python code/extract_scene_seg.py resized_videos_frames.lst \
deeplabv3_xception_ade20k_train/frozen_inference_graph.pb scene_seg_36x64 \
--every 1 --down_rate 8.0 --job 1 --curJob 1 --gpuid 0 --save_two_level
```

6. Preprocess and get all the npz files
```
$ python code/preprocess.py prepared_data_fold1/traj_2.5fps/ prepro_fold1 \
--obs_len 8 --pred_len 12 --add_scene --scene_feat_path scene_seg_36x64/ \
--direct_scene_feat --scene_id2name packed_prepro/scene36_64_id2name_top10.json \
 --scene_h 36 --scene_w 64 --grid_strides 2,4 --video_h 1080 --video_w 1920 \
 --add_grid --add_all_reg
```

Now you can follow [this](TRAINING.md#sdd) to do training and testing.

## Argoverse

We only utilize the videos from the official validation set for testing. First, download the 3D-tracking annotations and frames (validation set) from [Argoverse](https://s3.amazonaws.com/argoai-argoverse/tracking_val_v1.1.tar.gz).


1. Sample and rename the frames as well as getting the tracking annotation into formats we like (transform from 3D to 2D).
```
$ python code/get_prepared_data_argoverse.py argoverse-tracking/val/ \
val_frames_renamed prepared_data_val
```
Based on the data I downloaded from Argoverse in Sept. 2019, the code will skip 8 videos and result with only 16 of the validation videos due to video length.

Optionally, you can visualize the annotations:
```
$ python code/visualize_sdd_annotation.py prepared_data_val/ val_frames_renamed/ \
vis_gt --vis_num_frame_per_video 10 --for_argoverse
```

2. Get scene segmentation features
```
$ find $PWD/val_frames_renamed/ -name "*.jpg" > val_frames_renamed.lst
$ python code/extract_scene_seg.py val_frames_renamed.lst deeplabv3_xception_ade20k_train/frozen_inference_graph.pb \
scene_seg_36x64_argoverse --every 1 --down_rate 8.0 --job 1 --gpuid 0 --save_two_level
```

3. Proprocess!
```
$ python code/preprocess.py prepared_data_val/traj_2.5fps/ argoverse_prepro --obs_len 8 \
 --pred_len 12 --add_scene --scene_feat_path scene_seg_36x64_argoverse/ --direct_scene_feat \
 --scene_id2name packed_prepro/scene36_64_id2name_top10.json --scene_h 36 \
 --scene_w 64 --grid_strides 2,4 --video_h 1080 --video_w 1920 \
 --add_grid --add_all_reg
```

Now you can follow [this](TESTING.md) to do testing.
