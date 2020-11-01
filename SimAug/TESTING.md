
## Single Future Trajectory Prediction
We experimented on the [ActEv dataset](https://actev.nist.gov), [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) and [Argoverse Dataset](https://www.argoverse.org/) by following the [Next-prediction](https://github.com/google/next-prediction) evaluation protocol as stated in the paper.

First download the pre-processed data for all the datasets:
```
$ wget https://next.cs.cmu.edu/data/packed_prepro_eccv2020.tgz
$ tar -zxvf packed_prepro_eccv2020.tgz
```

Then download SimAug-trained model:
```
$ wget https://next.cs.cmu.edu/data/packed_models_eccv2020.tgz
$ tar -zxvf packed_models_eccv2020.tgz
```

### Test SimAug-Trained Model
The following is for ActEv experiments.

```
$ python code/test.py packed_prepro/actev_prepro/ packed_models/ best_simaug_model \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 --enc_hidden_size 256 \
--dec_hidden_size 256 --activation_func tanh --keep_prob 1.0 --num_epochs 30 \
--batch_size 12 --init_lr 0.3 --use_gnn --learning_rate_decay 0.95 --num_epoch_per_decay 5.0 \
--grid_loss_weight 1.0 --grid_reg_loss_weight 0.5 --save_period 3000 \
--scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,0 --val_grid_num 0 --gpuid 0 --load_best
```
The evaluation result should be:
<table>
  <tr>
    <td>Grid0_Accuracy</td>
    <td>ADE</td>
    <td>FDE</td>
  </tr>
  <tr>
    <td>0.51071</td>
    <td>21.730333</td>
    <td>42.223895</td>
  </tr>
</table>

The following is for Stanford Drone experiments.

```
$ python code/test.py packed_prepro/sdd_prepro/ packed_models/ best_simaug_model \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 --enc_hidden_size 256 \
--dec_hidden_size 256 --activation_func tanh --keep_prob 1.0 --num_epochs 30 \
--batch_size 12 --init_lr 0.3 --use_gnn --learning_rate_decay 0.95 --num_epoch_per_decay 5.0 \
--grid_loss_weight 1.0 --grid_reg_loss_weight 0.5 --save_period 3000 \
--scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,0 --val_grid_num 0 --gpuid 0 --load_best \
--save_output sdd_out.p
```

Since we resize all the videos to 1920x1080 in the pre-processing, we will need to compute the evaluation by scaling the errors back to the original resolutions:
```
$ python code/evaluate_sdd.py packed_prepro/sdd_resized.lst sdd_out.p
```

The evaluation result should be:
<table>
  <tr>
    <td>ADE</td>
    <td>FDE</td>
  </tr>
  <tr>
    <td>15.7041016113</td>
    <td>30.20249250</td>
  </tr>
</table>


The following is for Argoverse:

```
$ python code/test.py packed_prepro/argoverse_prepro/ packed_models/ best_simaug_model \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 --enc_hidden_size 256 \
--dec_hidden_size 256 --activation_func tanh --keep_prob 1.0 --num_epochs 30 \
--batch_size 12 --init_lr 0.3 --use_gnn --learning_rate_decay 0.95 --num_epoch_per_decay 5.0 \
--grid_loss_weight 1.0 --grid_reg_loss_weight 0.5 --save_period 3000 \
--scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,0 --val_grid_num 0 --gpuid 0 --load_best \
--save_output argoverse_out.p
```
The evaluation result should be:
<table>
  <tr>
    <td>Grid0_Accuracy</td>
    <td>ADE</td>
    <td>FDE</td>
  </tr>
  <tr>
    <td>0.3181818</td>
    <td>68.729341008</td>
    <td>178.886694</td>
  </tr>
</table>

### Visualization
To visualize the outputs on SDD or Argoverse, first follow [here](PREPRO.md) to get the video frames (`resized_videos_frames` and `val_frames_renamed`) for both datasets.

We save the prediction output in `sdd_out.p` and `argoverse_out.p` during inference. Now we need to make a file with path to the output file and its trajectory color (BGR, separated with `_`):
```
$ echo $PWD/sdd_out.p,0_0_255 > sdd_run.lst
$ echo $PWD/argoverse_out.p,0_0_255 > argoverse_run.lst
```

Visualize a fix amount of trajectory samples (Note the color will be ignored if you want heatmap visualization as the paper):
```
$ python code/visualize_output.py sdd_run.lst resized_videos_frames \
sdd_vis_heatmap/ --vis_num 100 --use_heatmap
$ python code/visualize_output.py argoverse_run.lst val_frames_renamed \
argoverse_vis_heatmap/ --vis_num 100 --use_heatmap
```
The code will use the video frame at the first observation timestep for visualization.


### Test ActEV-Finetuned Model
The following is for ActEv experiments.

```
$ python code/test.py packed_prepro/actev_prepro/ packed_models/ best_actev_finetuned \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 --enc_hidden_size 256 \
--dec_hidden_size 256 --activation_func tanh --keep_prob 1.0 --num_epochs 30 \
--batch_size 12 --init_lr 0.3 --use_gnn --learning_rate_decay 0.95 --num_epoch_per_decay 5.0 \
--grid_loss_weight 1.0 --grid_reg_loss_weight 0.5 --save_period 3000 \
--scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,0 --val_grid_num 0 --gpuid 0 --load_best
```
The evaluation result should be:
<table>
  <tr>
    <td>Grid0_Accuracy</td>
    <td>ADE</td>
    <td>FDE</td>
  </tr>
  <tr>
    <td>0.695828457</td>
    <td>17.96449857</td>
    <td>34.6829926720</td>
  </tr>
</table>

