# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Train person prediction model.

See README for running instructions.
"""

import argparse
import math
import os
import sys
import pred_models
import pred_utils
import tensorflow as tf
from tqdm import tqdm

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
parser.add_argument("--only_scene", default=None)


# ------------------- basic model parameters
parser.add_argument("--emb_size", type=int, default=128)
parser.add_argument("--enc_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--dec_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--activation_func", type=str,
                    default="tanh", help="relu/lrelu/tanh")



# --------- scene features
parser.add_argument("--scene_conv_kernel", default=3, type=int)
parser.add_argument("--scene_h", default=36, type=int)
parser.add_argument("--scene_w", default=64, type=int)
parser.add_argument("--scene_class", default=11, type=int)
parser.add_argument("--scene_conv_dim", default=64, type=int)
parser.add_argument("--convlstm_kernel", default=3, type=int)

parser.add_argument("--scene_grid_strides", default="2,4,8")
parser.add_argument("--use_grids", default="1,1,1")
parser.add_argument("--val_grid_num", type=int, default=1,
                    help="which grid to use for validation metric")

parser.add_argument("--use_beam_search", action="store_true")
parser.add_argument("--diverse_beam", action="store_true")
parser.add_argument("--beam_size", type=int, default=5)

# 07/2019
parser.add_argument("--use_gn", action="store_true")


# ----multi future training
parser.add_argument("--use_teacher_forcing", action="store_true")
parser.add_argument("--train_w_onehot", action="store_true")

parser.add_argument("--use_soft_grid_class", action="store_true")
parser.add_argument("--soft_grid", default=1, type=int)

parser.add_argument("--mask_grid_regression", action="store_true")  # bad

parser.add_argument("--use_gnn", action="store_true")

#parser.add_argument("--use_scene_enc", action="store_true")

parser.add_argument("--use_single_decoder", action="store_true")  # bad


# multi future testing
parser.add_argument("--use_gt_grid", action="store_true")

parser.add_argument("--check_model", action="store_true",
                    help="print the model variables and exit.")

# adversarial training
parser.add_argument("--adv_train", action="store_true")
# cannot use fp16 due to ConvLSTMCell has fp32
#parser.add_argument("--adv_use_fp16", action="store_true")
# Xie et. al. use epsilon 0.128, step_size 0.008, clean_prob 0.2
parser.add_argument("--adv_epsilon", type=float, default=0.1)
parser.add_argument("--adv_step_size", type=float, default=0.001)
parser.add_argument("--adv_num_iter", type=int, default=30)
parser.add_argument("--adv_start_from_clean_prob", default=0.0, type=float,
                    help="feature perturbation initialize from clean")
parser.add_argument("--adv_use_fgsm", action="store_true",
                    help="use FGSM instead of multi-step PGD")

parser.add_argument("--standard_aug", action="store_true")

# normalize the scene seg feature
parser.add_argument("--norm_feat", action="store_true")

# use mixup
parser.add_argument("--use_mixup", action="store_true")
parser.add_argument("--mixup_alpha", type=float, default=1.0)
parser.add_argument("--mixup_mix_adv", action="store_true",
                    help="mix two adv image instead of mixing with clean")

# multiview train
parser.add_argument("--multiview_train", action="store_true")
parser.add_argument("--norm_input", action="store_true",
                    help="normalize input feature to [-1, 1]")
parser.add_argument("--multiview_exp", default=1, type=int,
                    help="1 for top loss, 2 for random")
parser.add_argument("--multiview_random", action="store_true")
parser.add_argument("--multiview_max_weight_for_first", action="store_true")
#parser.add_argument("--multiview_use_adv_feat1", action="store_true")
parser.add_argument("--multiview_use_adv_for_loss", action="store_true")

parser.add_argument("--double_weighting", action="store_true")
parser.add_argument("--fl_gamma", default=1.0, type=float, help="1,2,3")


#  --------- loss weight
parser.add_argument("--loss_moving_avg_step", default=100, type=int)
parser.add_argument("--grid_loss_weight", default=1.0, type=float)
parser.add_argument("--grid_reg_loss_weight", default=0.1, type=float)

# ---------------------------- training hparam
parser.add_argument("--save_period", type=int, default=300,
                    help="num steps to save model and eval")
parser.add_argument("--batch_size", type=int, default=64)
# num_step will be num_example/batch_size * epoch
parser.add_argument("--num_epochs", type=int, default=100)
# drop out rate
parser.add_argument("--keep_prob", default=0.7, type=float,
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
                    help="learning scaling factor for emb variables")



def main(args):
  """Run training."""
  val_perf = []  # summary of validation performance, and the training loss

  train_data = pred_utils.read_data(args, "train")
  val_data = pred_utils.read_data(args, "val")

  args.train_num_examples = train_data.num_examples

  # construct model under gpu0
  model = pred_models.get_model(args, gpuid=args.gpuid)
  args.is_train = False
  val_model = pred_models.get_model(args, gpuid=args.gpuid)
  args.is_train = True

  if args.check_model:
    print("--------------- Model Weights -----------------")
    for var in tf.global_variables():
      not_show = False
      for c in ["Adam", "beta1_power", "beta2_power", "Adam_1", "Adadelta_1",
                "Adadelta", "Momentum", "global_step"]:
        if c in var.name:
          not_show = True
      if not_show:
        continue
      shape = var.get_shape()
      print("%s %s\n" % (var.name, shape))
    return

  trainer = pred_models.Trainer(model, args)
  tester = pred_models.Tester(val_model, args)
  saver = tf.train.Saver(max_to_keep=5)
  bestsaver = tf.train.Saver(max_to_keep=5)

  save_period = args.save_period  # also the eval period

  # start training!
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i for i in [args.gpuid]]))
  with tf.Session(config=tfconfig) as sess:

    pred_utils.initialize(
        load=args.load, load_best=args.load_best, args=args, sess=sess)

    # the total step (iteration) the model will run
    # total / batchSize  * epoch
    num_steps = int(math.ceil(train_data.num_examples /
                              float(args.batch_size)))*args.num_epochs
    # get_batches is a generator, run on the fly

    print(" batch_size:%s, epoch:%s, %s step every epoch, total step:%s,"
          " eval/save every %s steps" % (args.batch_size,
                                         args.num_epochs,
                                         math.ceil(train_data.num_examples/
                                                   float(args.batch_size)),
                                         num_steps,
                                         args.save_period))

    metric = "grid%d_traj_ade" % args.val_grid_num  # average displacement error # smaller better
    # remember the best eval acc during training
    best = {metric: 999999, "step": -1}

    finalperf = None
    is_start = True
    loss, wd_loss = [pred_utils.FIFO_ME(args.loss_moving_avg_step)
                     for i in range(2)]
    pred_grid_loss = [pred_utils.FIFO_ME(args.loss_moving_avg_step)
                      for _ in range(sum(args.use_grids))] * 2

    for batch in tqdm(train_data.get_batches(args.batch_size,
                                             num_steps=num_steps),
                      total=num_steps, ascii=True):

      global_step = sess.run(model.global_step) + 1  # start from 0

      # if load from existing model, save if first
      if (global_step % save_period == 0) or \
         (args.load_best and is_start) or \
         (args.load and is_start):

        tqdm.write("\tsaving model %s..." % global_step)
        saver.save(sess, args.save_dir_model, global_step=global_step)
        tqdm.write("\tdone")

        evalperf = pred_utils.evaluate(val_data, args, sess, tester)

        tqdm.write(("\tmoving average of %s steps: loss:%s, wd_loss:%s,"
                    " pred_grid_loss:%s,"
                    " eval on validation:%s,"
                    " (best %s:%s at step %s) ") % (
                        args.loss_moving_avg_step,
                        loss, wd_loss,
                        pred_grid_loss,
                        ["%s: %.4f" % (k, evalperf[k])
                         for k in sorted(evalperf.keys())], metric,
                        best[metric], best["step"]))

        # remember the best acc
        if evalperf[metric] < best[metric]:
          best[metric] = evalperf[metric]
          best["step"] = global_step
          # save the best model
          tqdm.write("\t saving best model...")
          bestsaver.save(sess, args.save_dir_best_model,
                         global_step=global_step)
          tqdm.write("\t done.")

          val_perf.append((loss, evalperf))

        finalperf = evalperf
        is_start = False

      this_loss, _, this_wd_loss, this_pred_grid_loss = \
          trainer.step(sess, batch)

      if math.isnan(this_loss):
        print("nan loss.")
        print(this_pred_grid_loss)
        sys.exit()

      # add to moving average
      loss.put(this_loss)
      wd_loss.put(this_wd_loss)
      for i in range(len(pred_grid_loss)):
        pred_grid_loss[i].put(this_pred_grid_loss[i])

    if global_step % save_period != 0:
      saver.save(sess, args.save_dir_model, global_step=global_step)

    print("best eval on val %s: %s at %s step, final step %s %s is %s" % (
        metric, best[metric], best["step"], global_step, metric,
        finalperf[metric]))


if __name__ == "__main__":
  arguments = parser.parse_args()
  arguments.is_train = True
  arguments.is_test = False
  arguments = pred_utils.process_args(arguments)

  main(arguments)
