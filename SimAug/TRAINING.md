## SimAug Training
We experimented on the multi-view reconstructed ActEV dataset with the SimAug algorithm. First download the dataset according to [here](./README.md#dataset) and extract the dataset.

First download the pre-processed test data according to [here](./TESTING.md#single-future-trajectory-prediction).

### Step 1: Pre-processing
Preprocess the data for SimAug training.

Get data split:
```
$ python code/get_split_path.py anchor_actev_4view_videos/rgb_videos/ \
anchor_actev_4view_videos/data_splits --is_anchor --ori_split_path packed_prepro/actev_data_splits/
```

Get trajectories:
```
$ python code/get_prepared_data.py anchor_actev_4view_videos \
anchor_actev_4view_videos/data_splits/ anchor_actev_4view_videos/prepared_data
```

Get RGB frames and scene segmentation features:
```
$ python code/get_frames_and_scene_seg.py anchor_actev_4view_videos/prepared_data/traj_2.5fps/ \
anchor_actev_4view_videos/rgb_videos/ anchor_actev_4view_videos/seg_videos/ \
anchor_actev_4view_videos/prepared_data/rgb_frames anchor_actev_4view_videos/prepared_data/scene_seg/ \
anchor_actev_4view_videos/prepared_data/bad_video.lst --scene_h 36 --scene_w 64
```

Remove videos with bad trajectories:
```
$ cd anchor_actev_4view_videos/prepared_data/
$ cat bad_video.lst |while read line;do rm -rf traj_2.5fps/${line}.* ;done
$ cd ../../
```
total video 5600, 1488 bad (0.2657)

Preprocess data into npz files for efficient training:
```
$ python code/preprocess.py anchor_actev_4view_videos/prepared_data/traj_2.5fps/ \
anchor_actev_4view_videos/prepro --obs_len 8 --pred_len 12 --add_scene \
--scene_feat_path anchor_actev_4view_videos/prepared_data/scene_seg/ \
--direct_scene_feat --scene_id2name packed_prepro/scene36_64_id2name_top10.json \
--scene_h 36 --scene_w 64 --grid_strides 2,4 --video_h 1080 --video_w 1920 --add_grid --add_all_reg
```

The files will be under `anchor_actev_4view_videos/prepro`.

### Step 2: Train the model using SimAug
You can train your model by running:

```
$ python code/train.py anchor_actev_4view_videos/prepro simaug_my_model/ modelname \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 --enc_hidden_size 256 \
--dec_hidden_size 256 --activation_func tanh --keep_prob 1.0 --num_epochs 30 \
--batch_size 12 --init_lr 0.3 --use_gnn --learning_rate_decay 0.95 \
--num_epoch_per_decay 8.0 --grid_loss_weight 1.0 --grid_reg_loss_weight 0.5 \
--save_period 3000 --scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,0 --val_grid_num 0 --train_w_onehot \
--adv_epsilon 0.1 --mixup_alpha 0.2 --multiview_train --multiview_exp 3 --gpuid 0
```

This will take 36 hours on a GTX 1080 TI GPU. Check the [code](code/train.py) for other hyper-parameters to try. Then you can follow the [testing guide](TESTING.md) to test the model on three different datasets.


## Finetuning on SDD/ActEV

First download the pre-processed train/val data for SDD and actev:
Google Drive [link](https://drive.google.com/file/d/1U7--eOKdjrfwLk9uqgr6jcCsRoAF8YtS/view?usp=sharing)
```
$ tar -zxvf packed_prepro_trainval_sddactev_eccv2020.tgz
```

Or you can follow [this instruction](PREPRO.md) to get the train/val.npz for SDD yourself. To get ActEV prepro files, we utilize the [Multiverse Model](https://github.com/JunweiLiang/Multiverse/blob/master/TRAINING.md#step-1-preprocess) repo.

Then download SimAug-trained model if you have not done so:
```
$ wget https://next.cs.cmu.edu/data/packed_models_eccv2020.tgz
$ tar -zxvf packed_models_eccv2020.tgz
```

### SDD

Finetuning from the SimAug-trained model (single model):
```
$ python code/train.py sddactev_trainval/sdd_fold1/ simaug_my_finetune \
sdd_modelname --load_from packed_models/best_simaug_model/00/best/ --wd 0.001 \
--runId 0 --obs_len 8 --pred_len 12 --emb_size 32 --enc_hidden_size 256 \
--dec_hidden_size 256 --activation_func tanh --keep_prob 1.0 --num_epochs 5 \
--batch_size 20 --init_lr 0.01 --use_gnn --learning_rate_decay 0.95 \
--num_epoch_per_decay 2.0 --grid_loss_weight 1.0 --grid_reg_loss_weight 0.1 \
--save_period 1000 --scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim \
64 --scene_grid_strides 2,4 --use_grids 1,0 --val_grid_num 0 --train_w_onehot \
--gpuid 0
```

Test:
```
$ python code/test.py packed_prepro/sdd_prepro/ simaug_my_finetune sdd_modelname \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 --enc_hidden_size 256 \
--dec_hidden_size 256 --activation_func tanh --keep_prob 1.0 --num_epochs 5 \
--batch_size 20 --init_lr 0.01 --use_gnn --learning_rate_decay 0.95 \
--num_epoch_per_decay 2.0 --grid_loss_weight 1.0 --grid_reg_loss_weight 0.1 \
--save_period 1000 --scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim \
64 --scene_grid_strides 2,4 --use_grids 1,0 --gpuid 0 --save_output sdd_finetune.p
```

Eval:
```
$ python code/evaluate_sdd.py packed_prepro/resized.lst sdd_out_finetune.p
```

### ActEV

Finetuning from the SimAug-trained model (single model):
```
$ python code/train.py sddactev_trainval/actev/ simaug_my_finetune actev_modelname \
 --load_from packed_models/best_simaug_model/00/best/ --wd 0.001 --runId 0 \
 --obs_len 8 --pred_len 12 --emb_size 32 --enc_hidden_size 256 --dec_hidden_size \
 256 --activation_func tanh --keep_prob 1.0 --num_epochs 30 --batch_size 20 \
 --init_lr 0.05 --use_gnn --learning_rate_decay 0.95 --num_epoch_per_decay 2.0 \
 --grid_loss_weight 1.0 --grid_reg_loss_weight 0.1 --save_period 1000 --scene_h 36 \
 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 --scene_grid_strides 2,4 \
 --use_grids 1,0 --val_grid_num 0 --train_w_onehot --gpuid 0
```

Test:
```
$ python code/test.py packed_prepro/actev_prepro/ simaug_my_finetune \
actev_modelname --wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 \
--enc_hidden_size 256 --dec_hidden_size 256 --activation_func tanh --keep_prob 1.0 \
--num_epochs 30 --batch_size 20 --init_lr 0.05 --use_gnn --learning_rate_decay 0.95 \
 --num_epoch_per_decay 2.0 --grid_loss_weight 1.0 --grid_reg_loss_weight 0.1 \
 --save_period 1000 --scene_h 36 --scene_w 64 --scene_conv_kernel 3 \
 --scene_conv_dim 64 --scene_grid_strides 2,4 --use_grids 1,0 --gpuid 0
```
