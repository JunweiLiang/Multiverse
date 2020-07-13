## Training
We experimented on the [ActEv dataset](https://actev.nist.gov) by following the [Next-prediction](https://github.com/google/next-prediction) evaluation protocol.
First download the processed annotations from the [Next-prediction](https://github.com/google/next-prediction) repo.
```
$ wget https://next.cs.cmu.edu/data/072019_prepare_data/072019_next_my_prepare_nopersoncnn.tgz
$ tar -zxvf 072019_next_my_prepare_nopersoncnn.tgz
```

Then download pretrained models by running the script
`bash scripts/download_single_models.sh`.

### Step 1: Preprocess
Preprocess the data for training and testing.
The following is for ActEv experiments.

```
$ python code/preprocess.py prepared_data/traj_2.5fps/ actev_preprocess \
--obs_len 8 --pred_len 12 --add_kp --kp_path prepared_data/anno_kp/ \
--add_scene --scene_feat_path prepared_data/actev_all_video_frames_scene_seg_every30_36x64/ \
--scene_map_path prepared_data/anno_scene/ --scene_id2name prepared_data/scene36_64_id2name_top10.json \
--scene_h 36 --scene_w 64 --video_h 1080 --video_w 1920 --add_grid \
--add_person_box --person_box_path prepared_data/anno_person_box/ \
--add_other_box --other_box_path prepared_data/anno_other_box/ \
--add_activity --activity_path prepared_data/anno_activity/ \
--person_boxkey2id_p prepared_data/person_boxkey2id.p --add_all_reg
```

### Step 2: Train the model
You can train your model by running:

```
$ python code/train.py actev_preprocess multiverse-models new_train/ \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 \
--enc_hidden_size 256 --dec_hidden_size 256 --activation_func tanh \
--keep_prob 1.0 --num_epochs 80 --batch_size 20 --init_lr 0.3 --use_gnn \
--use_scene --learning_rate_decay 0.95 --num_epoch_per_decay 2.0 \
--grid_loss_weight 1.0 --grid_reg_loss_weight 0.2 --save_period 2000 \
--scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,1 --val_grid_num 0 --train_w_onehot --gpuid 0
```

By default this will train the multiverse model, periodically saving model
files to `multiverse-models/new_train/00/save` at the current working
directory.
The script will also periodically evaluate the model on the
validation set and save the latest best model to
`multiverse-models/new_train/00/best`.
On a TITAN X GPU it takes about 48 hours to finish training.
Detailed commands of the training script:

### Training options

- `--batch_size`: How many trajectories to use in each minibatch during training.
- `--num_epochs`: Number of training epochs.
- `--init_lr`: Initial Learning rate.
- `--optimizer`: Optimizer to use. Default is AdaDelta.
- `--grid_loss_weight`: Weight for grid classification loss.
- `--grid_reg_loss_weight`: Weight for grid regress loss.

###  Basic model options

- `--emb_size`: Embedding size.
- `--enc_hidden_size`: Encoder hidden size.
- `--dec_hidden_size`: Decoder hidden size.
- `--activation_func`: Activation function.
You could choose from relu/lrelu/tanh.

### Step 4: Test the model - Single Future Trajectory Prediction
You can use following command to test the newly trained model:

```
$ python code/test.py actev_preprocess multiverse-models new_train/ \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 \
--enc_hidden_size 256 --dec_hidden_size 256 --activation_func tanh \
--keep_prob 1.0 --num_epochs 80 --batch_size 20 --init_lr 0.3 --use_gnn \
--use_scene --learning_rate_decay 0.95 --num_epoch_per_decay 2.0 \
--grid_loss_weight 1.0 --grid_reg_loss_weight 0.2 --save_period 2000 \
--scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,0 --val_grid_num 0 --gpuid 0 --load_best
```
The best model on the validation set will be used.

### Step 5: Multi-future Trajectory Prediction
Please follow [this](TESTING.md#multi-future-trajectory-prediction) for multi-future trajectory prediction inferencing.
