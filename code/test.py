# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Test person prediction model.

See README for running instructions.
"""

import argparse
import os
import pred_models
import pred_utils
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# the following will still have colocation debug info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()

# inputs and outputs
parser.add_argument("prepropath", type=str)
parser.add_argument("outbasepath", type=str,
                    help="full path will be outbasepath/modelname/runId")
parser.add_argument("modelname", type=str)
parser.add_argument("--runId", type=int, default=0,
                    help="used for run the same model multiple times")

# ---- gpu stuff. Now only one gpu is used
parser.add_argument("--gpuid", default=0, type=int)

parser.add_argument("--load", action="store_true",
                    default=False, help="whether to load existing model")
parser.add_argument("--load_best", action="store_true",
                    default=False, help="whether to load the best model")
# use for pre-trained model
parser.add_argument("--load_from", type=str, default=None)

parser.add_argument("--save_output", default=None)

# ------------- experiment settings
parser.add_argument("--obs_len", type=int, default=8)
parser.add_argument("--pred_len", type=int, default=12)

parser.add_argument("--per_scene_eval", action="store_true")
parser.add_argument("--show_grid_acc_at_T", action="store_true")

# ------------------- basic model parameters
parser.add_argument("--emb_size", type=int, default=128)
parser.add_argument("--enc_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--dec_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--activation_func", type=str,
                    default="tanh", help="relu/lrelu/tanh")

parser.add_argument("--show_center_only", action="store_true")

# --------- scene features
parser.add_argument("--scene_conv_kernel", default=3, type=int)
parser.add_argument("--scene_h", default=36, type=int)
parser.add_argument("--scene_w", default=64, type=int)
parser.add_argument("--scene_class", default=11, type=int)
parser.add_argument("--scene_conv_dim", default=64, type=int)
parser.add_argument("--pool_scale_idx", default=0, type=int)
parser.add_argument("--convlstm_kernel", default=3, type=int)

parser.add_argument("--scene_grid_strides", default="4,8,16")
parser.add_argument("--use_grids", default="1,1,1")
parser.add_argument("--val_grid_num", type=int, default=1,
                    help="which grid to use for validation metric")

parser.add_argument("--use_beam_search", action="store_true")
parser.add_argument("--diverse_beam", action="store_true")
parser.add_argument("--diverse_gamma", type=float, default=1.0)
parser.add_argument("--fix_num_timestep", type=int, default=0)
parser.add_argument("--beam_size", type=int, default=5)

# 07/2019
parser.add_argument("--use_gn", action="store_true")


# ----multi future training
parser.add_argument("--use_teacher_forcing", action="store_true")

parser.add_argument("--use_soft_grid_class", action="store_true")
parser.add_argument("--soft_grid", default=1, type=int)

parser.add_argument("--mask_grid_regression", action="store_true")

parser.add_argument("--use_gnn", action="store_true")

parser.add_argument("--use_scene_enc", action="store_true")

parser.add_argument("--use_single_decoder", action="store_true")

# multi future testing
parser.add_argument("--use_gt_grid", action="store_true")


#  --------- loss weight
parser.add_argument("--loss_moving_avg_step", default=100, type=int)
parser.add_argument("--grid_loss_weight", default=1.0, type=float)
parser.add_argument("--grid_reg_loss_weight", default=1.0, type=float)

# ---------------------------- training hparam
parser.add_argument("--save_period", type=int, default=300,
                    help="num steps to save model and eval")
parser.add_argument("--batch_size", type=int, default=64)
# num_step will be num_example/batch_size * epoch
parser.add_argument("--num_epochs", type=int, default=100)
# drop out rate
parser.add_argument("--keep_prob", default=1.0, type=float,
                    help="1.0 - drop out rate")
# l2 weight decay rate
parser.add_argument("--wd", default=0.0001, type=float,
                    help="l2 weight decay loss")
parser.add_argument("--clip_gradient_norm", default=10, type=float,
                    help="gradient clipping")
parser.add_argument("--optimizer", default="adadelta",
                    help="momentum|adadelta|adam|rmsprop")
parser.add_argument("--use_cosine_lr", action="store_true",
                    help="use cosine learning rate decay")
parser.add_argument("--learning_rate_decay", default=0.95,
                    type=float, help="learning rate decay")
parser.add_argument("--num_epoch_per_decay", default=2.0,
                    type=float, help="how epoch after which lr decay")
parser.add_argument("--init_lr", default=0.2, type=float,
                    help="Start learning rate")
parser.add_argument("--emb_lr", type=float, default=1.0,
                    help="learning scaling factor for 'emb' variables")


def main(args):
  """Run testing."""
  test_data = pred_utils.read_data(args, "test")
  print("total test samples:%s" % test_data.num_examples)

  model = pred_models.get_model(args, gpuid=args.gpuid)
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i for i in [args.gpuid]]))

  with tf.Session(config=tfconfig) as sess:
    pred_utils.initialize(load=True, load_best=args.load_best,
                          args=args, sess=sess)

    # load the graph and variables
    tester = pred_models.Tester(model, args, sess)

    perf = pred_utils.evaluate(test_data, args, sess, tester)

  print("performance:")
  key_metrics = []
  for i in range(len(args.scene_grids)):
    if not args.use_grids[i]:
      continue
    key_metrics += ["grid%d_acc" % i, "grid%d_traj_ade" % i,
                    "grid%d_traj_fde" % i]
    if args.show_center_only:
      key_metrics += ["grid%d_centerOnly_traj_ade" % i,
                      "grid%d_centerOnly_traj_fde" % i]
    if args.show_grid_acc_at_T:
      # min, max length, then 2 second, 4 second
      show_T = [0, 4, 9, 11]
      key_metrics += ["grid%d_acc_@T=%d" % (i, t) for t in show_T]

  if args.per_scene_eval:
    scenes = ["0000", "0002", "0400", "0401", "0500"]
    key_metrics += [("%s_ade" % scene) for scene in scenes]
    key_metrics += [("%s_fde" % scene) for scene in scenes]
  numbers = []
  for k in sorted(perf.keys()):
    print("%s, %s" % (k, perf[k]))
    if k in key_metrics:
      numbers.append(("%s" % perf[k], k))
  print(" ".join([k for v, k in numbers]))
  print(" ".join([v for v, k in numbers]))


if __name__ == "__main__":
  arguments = parser.parse_args()
  arguments.is_train = False
  arguments.is_test = True
  arguments = pred_utils.process_args(arguments)

  main(arguments)
