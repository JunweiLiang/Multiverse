# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Utility functions and classes."""

import collections
import itertools
import math
import operator
import os
import random
import pickle
import sys
import numpy as np
import tensorflow as tf
import tqdm

__all__ = ["activity2id", "object2id",
           "initialize", "read_data"]

activity2id = {
    "BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
}

object2id = {
    "Person": 0,
    "Vehicle": 1,
    "Parking_Meter": 2,
    "Construction_Barrier": 3,
    "Door": 4,
    "Push_Pulled_Object": 5,
    "Construction_Vehicle": 6,
    "Prop": 7,
    "Bike": 8,
    "Dumpster": 9,
}


def process_args(args):
  """Process arguments.

  Model will be in outbasepath/modelname/runId/save

  Args:
    args: arguments.

  Returns:
    Edited arguments.
  """

  def mkdir(path):
    if not os.path.exists(path):
      os.makedirs(path)

  if args.activation_func == "relu":
    args.activation_func = tf.nn.relu
  elif args.activation_func == "tanh":
    args.activation_func = tf.nn.tanh
  elif args.activation_func == "lrelu":
    args.activation_func = tf.nn.leaky_relu
  else:
    print("unrecognied activation function, using relu...")
    args.activation_func = tf.nn.relu

  args.seq_len = args.obs_len + args.pred_len

  args.outpath = os.path.join(
      args.outbasepath, args.modelname, str(args.runId).zfill(2))
  mkdir(args.outpath)

  args.save_dir = os.path.join(args.outpath, "save")
  mkdir(args.save_dir)
  args.save_dir_model = os.path.join(args.save_dir, "save")
  args.save_dir_best = os.path.join(args.outpath, "best")
  mkdir(args.save_dir_best)
  args.save_dir_best_model = os.path.join(args.save_dir_best, "save-best")

  args.write_self_sum = True
  args.self_summary_path = os.path.join(args.outpath, "train_sum.txt")

  args.record_val_perf = True
  args.val_perf_path = os.path.join(args.outpath, "val_perf.p")

  args.object2id = object2id
  args.num_box_class = len(args.object2id)

  args.num_act = len(activity2id.keys())  # include the BG class

  # has to be 2,4 to match the scene CNN strides
  args.scene_grid_strides = [int(o) for o in args.scene_grid_strides.split(",")]
  args.use_grids = [bool(int(o)) for o in args.use_grids.split(",")]
  assert len(args.scene_grid_strides) == len(args.use_grids)
  assert sum(args.use_grids) <= 2, "Currently only supports at most two scale" \
                                   " training at a time"

  args.scene_grids = []
  for stride in args.scene_grid_strides:
    h, w = args.scene_h, args.scene_w
    this_h, this_w = round(h*1.0/stride), round(w*1.0/stride)
    this_h, this_w = int(this_h), int(this_w)
    args.scene_grids.append((this_h, this_w))

  if args.load_best:
    args.load = True
  if args.load_from is not None:
    args.load = True

  # if test, has to load
  if not args.is_train:
    args.load = True
    args.num_epochs = 1
    args.keep_prob = 1.0

  args.activity2id = activity2id
  return args


def initialize(load, load_best, args, sess):
  """Initialize graph with given model weights.

  Args:
    load: boolean, whether to load model weights
    load_best: whether to load from best model path
    args: arguments
    sess: tf.Session() instance

  Returns:
    None
  """

  tf.global_variables_initializer().run()

  if load:
    print("restoring model...")
    allvars = tf.global_variables()
    allvars = [var for var in allvars if "global_step" not in var.name]
    restore_vars = allvars
    opts = ["Adam", "beta1_power", "beta2_power",
            "Adam_1", "Adadelta_1", "Adadelta", "Momentum"]
    restore_vars = [var for var in restore_vars \
        if var.name.split(":")[0].split("/")[-1] not in opts]

    saver = tf.train.Saver(restore_vars, max_to_keep=5)

    load_from = None

    if args.load_from is not None:
      load_from = args.load_from
    else:
      if load_best:
        load_from = args.save_dir_best
      else:
        load_from = args.save_dir

    ckpt = tf.train.get_checkpoint_state(load_from)
    if ckpt and ckpt.model_checkpoint_path:
      loadpath = ckpt.model_checkpoint_path

      saver.restore(sess, loadpath)
      print("Model:")
      print("\tloaded %s" % loadpath)
      print("")
    else:
      if os.path.exists(load_from):
        if load_from.endswith(".ckpt"):
          # load_from should be a single .ckpt file
          saver.restore(sess, load_from)
        else:
          print("Not recognized model type:%s" % load_from)
          sys.exit()
      else:
        print("Model not exists")
        sys.exit()
    print("done.")


def read_data(args, data_type):
  """Read propocessed data into memory for experiments.

  Args:
    args: Arguments
    data_type: train/val/test

  Returns:
    Dataset instance
  """

  def get_traj_cat(cur_acts, traj_cats):
    """Get trajectory categories for virat/actev dataset experiments."""

    def is_in(l1, l2):
      """Check whether any of l1"s item is in l2."""
      for i in l1:
        if i in l2:
          return True
      return False

    # 1 is moving act, 0 is static
    act_cat = int(is_in(cur_acts, args.virat_mov_actids))
    i = -1
    for i, (_, actid) in enumerate(traj_cats):
      if actid == act_cat:
        return i
    # something is wrong
    assert i >= 0

  data_path = os.path.join(args.prepropath, "data_%s.npz" % data_type)

  data = dict(np.load(data_path, allow_pickle=True))

  # save some shared feature first

  shared = {}
  shares = ["scene_feat", "video_wh", "scene_grid_strides",
            "vid2name", "person_boxkey2id", "person_boxid2key"]

  excludes = [
      "seq_start_end",
      "obs_kp_rel", "obs_kp", "cur_activity", "obs_box", "future_activity",
      "pred_kp", "obs_other_box", "person_boxid2key"]

  if "video_wh" in data:
    args.box_img_w, args.box_img_h = data["video_wh"]
  else:
    args.box_img_w, args.box_img_h = 1920, 1080

  for i in range(len(args.scene_grid_strides)):
    shares.append("grid_center_%d" % i)

  for key in data:
    if key in shares:
      if not data[key].shape:
        shared[key] = data[key].item()
      else:
        shared[key] = data[key]

  num_examples = len(data["obs_traj"])  # (input,pred)

  newdata = {}
  for key in data:
    if key not in excludes+shares:
      if len(data[key]) != num_examples:
        print("warning, ignoring %s.." % key)
        continue
      newdata[key] = data[key]
  data = newdata

  # assert len(shared["scene_grid_strides"]) == len(args.scene_grid_strides)
  assert shared["scene_grid_strides"][0] == args.scene_grid_strides[0]



  #for key in data:
  #  assert len(data[key]) == num_examples, \
  #      (key, data[key].shape, num_examples)

  # category each trajectory for training
  #if shared.has_key("person_boxid2key"):
  if "person_boxid2key" in shared:
    data["traj_key"] = []
    boxid2key = shared["person_boxid2key"]
    for i in range(num_examples):
      # videoname_frameidx_personid
      key = boxid2key[data["obs_boxid"][i][0]]
      data["traj_key"].append(key)

  print("loaded %s data points for %s" % (num_examples, data_type))

  return Dataset(data, data_type, shared=shared, config=args)


def get_scene(videoname_):
  """Get the scene camera from the ActEV videoname."""
  s = videoname_.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]

# simple FIFO class for moving average computation
class FIFO_ME:
  def __init__(self, N):
    self.N = N
    self.lst = []
    assert N > 0
  def __str__(self):
    return "%.4f" % float(self.me())
  def __repr__(self):
    return "%.4f" % float(self.me())

  def put(self, val):
    if val is None:
      return None
    self.lst.append(val)
    if len(self.lst) > self.N:
      self.lst.pop(0)
    return 1

  def me(self):
    if len(self.lst) == 0:
      return -1
    return np.mean(self.lst)


def is_different_beam(beam_ids):
  # [beam_size, T]
  prev_list = beam_ids[0]
  is_different = False
  for i in range(1, len(beam_ids)):
    for t in range(len(prev_list)):
      if prev_list[t] != beam_ids[i, t]:
        is_different = True
        break
    if is_different:
      break
    prev_list = beam_ids[i]
  return is_different


def topk(np_list, k):
  indices = np.argsort(np_list)[::-1][:k]
  return indices, np_list[indices]


def evaluate(dataset, config, sess, tester):
  """Evaluate the dataset using the tester model.

  Args:
    dataset: the Dataset instance
    config: arguments
    sess: tensorflow session
    tester: the Tester instance

  Returns:
    Evaluation results.
  """

  # show the evaluation per trajectory class if actev experiment

  grid1 = []
  grid2 = []
  p = {}
  pred_len = config.pred_len

  if config.per_scene_eval:
    assert sum(config.use_grids) == 1, "per scene eval is for one grid only"
    scenes = ["0000", "0002", "0400", "0401", "0500"]
    l2dis_scenes = [[] for i in range(len(scenes))]

  if config.save_output is not None:
    out_data = {
        "obs_list": [],  # observable traj gt
        "pred_gt_list": [],
        "seq_ids": [],
    }
    for i in range(len(config.scene_grids)):
      out_data.update({("grid%s_class" % i): []})  # each should be [T] int
      out_data.update({("grid%s_gt_class" % i): []})
      out_data.update({("grid%s_pred_traj" % i): []})
      out_data.update(
          {("grid_center_%d" % i): dataset.shared["grid_center_%d" % i]})
    if config.use_beam_search:
      out_data.update({
          "beam_grid_ids": [],  # each is [beam_size, T]
          "beam_logprobs": []})  # [beam_size]

  # multi-future, predict grid at each time

  # trajectory from grid, displacements
  l2dis_grid = [[] for i in range(len(config.scene_grids))]
  l2dis_grid_centerOnly = [[] for i in range(len(config.scene_grids))]

  grid_class_pred = [[] for i in range(len(config.scene_grids))]
  grid_class_pred_at_T = [[[] for j in range(pred_len)]
                          for i in range(len(config.scene_grids))]

  num_batches_per_epoch = int(
      math.ceil(dataset.num_examples / float(config.batch_size)))

  for evalbatch in tqdm.tqdm(dataset.get_batches(config.batch_size, \
    full=True, shuffle=False), total=num_batches_per_epoch, ascii=True):
    #if count != 17:
    #  continue
    # [N,pred_len, 2]
    # here the output is relative output by default
    grid_pred_class, grid_pred_reg, beam_outputs = tester.step(sess, evalbatch)

    _, batch = evalbatch

    this_actual_batch_size = batch.data["original_batch_size"]

    N = this_actual_batch_size

    if config.use_beam_search:
      assert sum(config.use_grids) == 1
      beam_logits, beam_grid_ids, beam_logprobs = beam_outputs
      # logits [N, beam_size, T, H*W]
      # grid_ids [N, beam_size, T]  # each is 0-indexed id in H*W
      # logprobs [N, beam_size] # total log likelihood for each beam
      """
      for i in range(len(grid_ids)):
        if is_different_beam(grid_ids[i]):
          print(i)
          print(batch.data["pred_grid_class"][i][0])
          print(np.argmax(grid_pred_class[0][i], axis=1))
          print(grid_ids[i])
          print(logprobs[i])
          sys.exit()

      test_i = 0
      np.set_printoptions(threshold=sys.maxsize)
      print(batch.data["pred_grid_class"][test_i][0])
      print(logprobs[test_i])
      print(grid_ids[test_i])
      sys.exit()
      """

    for j, (H, W) in enumerate(config.scene_grids):
      if not config.use_grids[j]:
        continue
      # displacements in this minibatch
      grid_traj_d = []
      grid_centerOnly_traj_d = []

      # [N, T, H, W, 1]
      grid_class = grid_pred_class[j][:N]
      # [N, T, H*W]
      grid_class = grid_class.reshape([N, pred_len, H*W])
      # masking and leave only neighbors?
      # [N, T]
      grid_class_selected = np.argmax(grid_class, axis=2)
      # second-largest every time
      # grid_class_selected = np.argsort(grid_class, axis=2)[:, :, -2]
      if config.use_gt_grid:
        # [N, T]
        grid_class_selected = np.array(
            [batch.data["pred_grid_class"][i][j, :] for i in range(N)])

      # [N, T, H, W, 2]
      grid_reg = grid_pred_reg[j][:N]
      # [N, T, H*W, 2]
      grid_reg = grid_reg.reshape([N, pred_len, H*W, 2])

      for i in range(N):
        # [T]
        gt_grid_pred_class = batch.data["pred_grid_class"][i][j, :]
        grid_class_pred[j].extend(
            gt_grid_pred_class == grid_class_selected[i, :])

        grid_traj = []
        grid_traj_centerOnly = []
        for t in range(pred_len):
          grid_class_pred_at_T[j][t].append(
              gt_grid_pred_class[t] == grid_class_selected[i, t])

          # get the trajectory based on classification and regression
          this_grid_class = grid_class_selected[i, t]
          center_coors = batch.shared["grid_center_%s" % j]  # [H, W, 2]
          center_coors = center_coors.reshape([-1, 2])  # [H*W, 2]
          this_center = center_coors[this_grid_class]
          # [2]
          pred_point = this_center + grid_reg[i, t, this_grid_class, :]
          grid_traj.append(pred_point)
          grid_traj_centerOnly.append(this_center)

        # compute displacement
        grid_traj = np.array(grid_traj)
        gt_traj = batch.data["pred_traj"][i]  # [pred_len, 2]
        # [pred_len, 2]
        diff = gt_traj - grid_traj
        diff = diff**2
        diff = np.sqrt(np.sum(diff, axis=1))  # [pred_len]
        traj_diff = diff

        grid_traj_d.append(diff)

        grid_traj_centerOnly = np.array(grid_traj_centerOnly)
        # [pred_len, 2]
        diff = gt_traj - grid_traj_centerOnly
        diff = diff**2
        diff = np.sqrt(np.sum(diff, axis=1))  # [pred_len]

        grid_centerOnly_traj_d.append(diff)

        if config.per_scene_eval:
          traj_key = batch.data["traj_key"][i]  # videoname_frameidx_personid
          scene = get_scene(traj_key)  # 0000/0002, etc.
          l2dis_scenes[scenes.index(scene)].append(traj_diff)

        if config.save_output is not None:
          # videoname_frameidx_personid
          if j == 0:
            out_data["seq_ids"].append(batch.data["traj_key"][i])
            out_data["obs_list"].append(batch.data["obs_traj"][i])  # [obs_len, 2]
            out_data["pred_gt_list"].append(batch.data["pred_traj"][i])
          out_data["grid%s_pred_traj" % j].append(grid_traj)
          out_data["grid%s_gt_class" % j].append(gt_grid_pred_class)  # [T]
          out_data["grid%s_class" % j].append(grid_class[i])  # [T, H*W]

          if config.use_beam_search:
            out_data["beam_grid_ids"].append(beam_grid_ids[i])
            out_data["beam_logprobs"].append(beam_logprobs[i])

      l2dis_grid[j] += grid_traj_d
      l2dis_grid_centerOnly[j] += grid_centerOnly_traj_d

  # 1. the grid classfication accuracy
  for j in range(len(config.scene_grids)):
    if not config.use_grids[j]:
      continue
    grid_acc = np.mean(grid_class_pred[j])
    p.update({
        ("grid%d_acc" % j): grid_acc,
    })
    for t in range(pred_len):
      grid_acc_at_T = np.mean(grid_class_pred_at_T[j][t])
      p.update({
          ("grid%d_acc_@T=%d" % (j, t)): grid_acc_at_T,
      })

    # 2. the grid traj
    # average displacement
    ade = [t for o in l2dis_grid[j] for t in o]
    # final displacement
    fde = [o[-1] for o in l2dis_grid[j]]
    p.update({
        ("grid%d_traj_ade" % j): np.mean(ade),
        ("grid%d_traj_fde" % j): np.mean(fde),
    })
    # center point only traj, so we can see the regression's useful
    ade = [t for o in l2dis_grid_centerOnly[j] for t in o]
    # final displacement
    fde = [o[-1] for o in l2dis_grid_centerOnly[j]]
    p.update({
        ("grid%d_traj_centerOnly_ade" % j): np.mean(ade),
        ("grid%d_traj_centerOnly_fde" % j): np.mean(fde),
    })

  # per-scene eval
  if config.per_scene_eval:
    for scene_id, scene in enumerate(scenes):
      diffs = l2dis_scenes[scene_id]
      ade = [t for l in diffs for t in l]
      fde = [l[-1] for l in diffs]
      p.update({
          ("%s_ade" % scene): np.mean(ade) if ade else 0.0,
          ("%s_fde" % scene): np.mean(fde) if fde else 0.0,
      })

  if config.save_output is not None:
    num_seq = len(out_data["seq_ids"])
    # for k in out_data:
    #  print(k, len(out_data[k]))
    with open(config.save_output, "wb") as f:
      pickle.dump(out_data, f)
    print("saved output at %s." % config.save_output)
  return p


class Dataset(object):
  """Class for batching during training and testing."""

  def __init__(self, data, data_type, config=None, shared=None):
    self.data = data
    self.data_type = data_type
    self.valid_idxs = range(self.get_data_size())
    self.num_examples = len(self.valid_idxs)
    self.shared = shared
    self.config = config

  def get_data_size(self):
    return len(self.data["obs_traj"])

  def get_by_idxs(self, idxs):
    out = collections.defaultdict(list)
    for key, val in self.data.items():
      out[key].extend(val[idx] for idx in idxs)
    return out

  def get_batches(self, batch_size, \
      num_steps=0, shuffle=True, cap=False, full=False):
    """Iterator to get batches.

    should return num_steps -> batches
    step is total/batchSize * epoch
    cap means limits max number of generated batches to 1 epoch

    Args:
      batch_size: batch size.
      num_steps: total steps.
      shuffle: whether shuffling the data
      cap: cap at one epoch
      full: use full one epoch

    Yields:
      Dataset object.
    """

    num_batches_per_epoch = int(
        math.ceil(self.num_examples / float(batch_size)))
    if full:
      num_steps = num_batches_per_epoch

    if cap and (num_steps > num_batches_per_epoch):
      num_steps = num_batches_per_epoch
    # this may be zero
    num_epochs = int(math.ceil(num_steps/float(num_batches_per_epoch)))
    # shuflle
    if shuffle:
      # All epoch has the same order.
      random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs))
      # all batch idxs for one epoch

      def random_grouped():
        return list(grouper(random_idxs, batch_size))
      # grouper
      # given a list and n(batch_size), devide list into n sized chunks
      # last one will fill None
      grouped = random_grouped
    else:
      def raw_grouped():
        return list(grouper(self.valid_idxs, batch_size))
      grouped = raw_grouped

    # all batches idxs from multiple epochs
    batch_idxs_iter = itertools.chain.from_iterable(
        grouped() for _ in range(num_epochs))
    for _ in range(num_steps):  # num_step should be batch_idxs length
      # so in the end batch, the None will not included
      batch_idxs = tuple(i for i in next(batch_idxs_iter)
                         if i is not None)  # each batch idxs

      # so batch_idxs might not be size batch_size
      # pad with the last item
      original_batch_size = len(batch_idxs)
      if len(batch_idxs) < batch_size:
        pad = batch_idxs[-1]
        batch_idxs = tuple(
            list(batch_idxs) + [pad for i in
                                range(batch_size - len(batch_idxs))])

      # get the actual data based on idx
      batch_data = self.get_by_idxs(batch_idxs)

      batch_data.update({
          "original_batch_size": original_batch_size,
      })

      config = self.config

      # assemble a scene feat from the full scene feat matrix for this batch
      oldid2newid = {}
      new_obs_scene = np.zeros((config.batch_size, config.obs_len, 1),
                               dtype="int32")

      for i in range(len(batch_data["obs_scene"])):
        for j in range(len(batch_data["obs_scene"][i])):
          oldid = batch_data["obs_scene"][i][j][0]
          if oldid not in oldid2newid:
            oldid2newid[oldid] = len(oldid2newid.keys())
          newid = oldid2newid[oldid]
          new_obs_scene[i, j, 0] = newid
      # get all the feature used by this mini-batch
      scene_feat = np.zeros((len(oldid2newid), config.scene_h,
                             config.scene_w, config.scene_class),
                            dtype="float32")
      for oldid in oldid2newid:
        newid = oldid2newid[oldid]
        scene_feat[newid, :, :, :] = \
            self.shared["scene_feat"][oldid, :, :, :]

      batch_data.update({
          "batch_obs_scene": new_obs_scene,
          "batch_scene_feat": scene_feat,
      })

      yield batch_idxs, Dataset(batch_data, self.data_type, shared=self.shared)


def grouper(lst, num):
  args = [iter(lst)]*num
  if sys.version_info > (3, 0):
    out = itertools.zip_longest(*args, fillvalue=None)
  else:
    out = itertools.izip_longest(*args, fillvalue=None)
  out = list(out)
  return out


def compute_ap(lists):
  """Compute Average Precision."""
  lists.sort(key=operator.itemgetter("score"), reverse=True)
  rels = 0
  rank = 0
  score = 0.0
  for one in lists:
    rank += 1
    if one["label"] == 1:
      rels += 1
      score += rels/float(rank)
  if rels != 0:
    score /= float(rels)
  return score


def relative_to_abs(rel_traj, start_pos):
  """Relative x,y to absolute x,y coordinates.

  Args:
    rel_traj: numpy array [T,2]
    start_pos: [2]
  Returns:
    abs_traj: [T,2]
  """

  # batch, seq_len, 2
  # the relative xy cumulated across time first
  displacement = np.cumsum(rel_traj, axis=0)
  abs_traj = displacement + np.array([start_pos])  # [1,2]
  return abs_traj
