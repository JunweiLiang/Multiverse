# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Model graph definitions and other functions for training and testing."""

import math
import operator
import os
import random
import re
import sys
import numpy as np
import tensorflow as tf
from scipy import ndimage
from functools import reduce

def get_model(config, gpuid):
  """Make model instance and pin to one gpu.

  Args:
    config: arguments.
    gpuid: gpu id to use
  Returns:
    Model instance.
  """
  with tf.name_scope(config.modelname), tf.device("/gpu:%d" % gpuid):
    model = Model(config, "%s" % config.modelname)
  return model

class Model(object):
  """Model graph definitions.
  """

  def __init__(self, config, scope):
    self.scope = scope
    self.config = config

    self.global_step = tf.get_variable("global_step", shape=[],
                                       dtype="int32",
                                       initializer=tf.constant_initializer(0),
                                       trainable=False)

    # get all the dimension here
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N = config.batch_size

    SH = self.SH = config.scene_h
    SW = self.SW = config.scene_w
    SC = self.SC = config.scene_class

    self.beam_size = config.beam_size

    # all the inputs

    # the trajactory sequence,
    # in training, it is the obs+pred combined,
    # in testing, only obs is fed and the rest is zeros
    # [N,T1,2] # T1 is the obs_len

    self.obs_length = tf.placeholder(
        "int32", [N], name="obs_length")
    self.pred_length = tf.placeholder(
        "int32", [N], name="obs_length")

    # used for drop out switch
    self.is_train = tf.placeholder("bool", [], name="is_train")

    # scene semantic segmentation features
    # the index to the feature
    self.obs_scene = tf.placeholder("int32", [N, None], name="obs_scene")
    self.obs_scene_mask = tf.placeholder(
        "bool", [N, None], name="obs_scene_mask")
    # the actual feature
    self.scene_feat = tf.placeholder(
        "float32", [None, SH, SW, SC], name="scene_feat")


    # grid loss
    self.grid_pred_labels = []
    self.grid_pred_targets = []
    self.grid_obs_labels = []
    self.grid_obs_targets = []


    self.grid_obs_regress = []  # [N, T, H, W, 2]
    self.grid_pred_labels_T = []
    self.grid_pred_regress = []

    for h, w in config.scene_grids:
      # [N, seq_len]
      # currently only the destination
      self.grid_pred_labels.append(
          tf.placeholder("int32", [N]))  # grid class
      self.grid_pred_targets.append(tf.placeholder("float32", [N, 2]))

      self.grid_obs_labels.append(
          tf.placeholder("int32", [N, None]))  # grid class
      self.grid_obs_targets.append(
          tf.placeholder("float32", [N, None, 2]))


      self.grid_obs_regress.append(
          tf.placeholder("float32", [N, None, h, w, 2],
                         name="grid_obs_regress"))
      if config.use_soft_grid_class:
        self.grid_pred_labels_T.append(
            tf.placeholder("float32", [N, None, h, w, 1], name="grid_pred_classes"))
      else:
        self.grid_pred_labels_T.append(
            tf.placeholder("float32", [N, None], name="grid_pred_classes"))
      self.grid_pred_regress.append(
          tf.placeholder("float32", [N, None, h, w, 2],
                         name="grid_pred_regress"))

    self.beam_outputs = None
    self.loss = None
    self.build_forward()
    if config.is_train:
      self.build_loss()

  def build_forward(self):
    """Build the forward model graph."""
    config = self.config
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N

    # add dropout
    keep_prob = tf.cond(self.is_train,
                        lambda: tf.constant(config.keep_prob),
                        lambda: tf.constant(1.0))

    # ------------------------- multi-future encoder decoder
    obs_length = self.obs_length
    pred_length = self.pred_length

    # top_scope is used for variable inside
    # encode and decode if want to share variable across
    with tf.variable_scope("person_pred") as top_scope:

      self.grid_memory = []
      self.grid_pred_decoded = []
      self.grid_pred_reg_decoded = []

      if config.use_scene_enc:
        # [N, T, SH, SW, SC]
        obs_scene = tf.nn.embedding_lookup(
            self.scene_feat, self.obs_scene)
        # [N*T, SH, SW, SC]
        obs_scene = tf.reshape(
            obs_scene, [-1, config.scene_h, config.scene_w, config.scene_class])
        scene_conv = obs_scene
        self.scene_convs = []
        for i, stride in enumerate(config.scene_grid_strides):
          # [N*T, SH/2, SW/2, dim]
          scene_conv = conv2d(scene_conv, out_channel=config.scene_conv_dim,
                              kernel=config.scene_conv_kernel,
                              stride=2, activation=config.activation_func,
                              add_bias=True, scope="scene_conv%d" % (i+1))
          scene_feat = tf.reshape(
              scene_conv,
              [N, -1, config.scene_h // stride, config.scene_w // stride,
               config.scene_conv_dim])
          self.scene_convs.append(scene_feat)


      for i, (h, w) in enumerate(config.scene_grids):
        if not config.use_grids[i]:
          self.grid_pred_decoded.append([])
          self.grid_pred_reg_decoded.append([])
          continue
        # [N, T, h, w, 1]
        obs_grid_class = tf.one_hot(self.grid_obs_labels[i], h*w)
        obs_grid_class = tf.reshape(obs_grid_class, [N, -1, h, w, 1])

        # [N, h, w, 1]  # all the grid class that the agent has traveled
        self.grid_memory.append(tf.reduce_sum(obs_grid_class, 1))

        # [N, T, h, w, 2]
        obs_grid_reg = self.grid_obs_regress[i]

        grid_rnn_input_shape = [h, w, 1]
        if config.use_scene_enc:
          grid_rnn_input_shape = [h, w, config.scene_conv_dim]
        else:
          grid_rnn_input_shape = [h, w, config.emb_size]

        enc_cell_obs_grid = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2, input_shape=grid_rnn_input_shape,
            output_channels=config.enc_hidden_size,
            kernel_shape=[config.convlstm_kernel, config.convlstm_kernel],
            name="enc_grid_%d" % i)
        enc_cell_obs_grid = tf.nn.rnn_cell.DropoutWrapper(
            enc_cell_obs_grid, keep_prob)
        enc_cell_obs_grid_reg = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2, input_shape=[h, w, 2],
            output_channels=config.enc_hidden_size,
            kernel_shape=[config.convlstm_kernel, config.convlstm_kernel],
            name="enc_grid_regress_%d" % i)
        enc_cell_obs_grid_reg = tf.nn.rnn_cell.DropoutWrapper(
            enc_cell_obs_grid_reg, keep_prob)

        # [N, T, h, w, 1]  ->  [N, T, h, w, h_dim]
        T = tf.shape(obs_grid_class)[1]
        self.scene_enc_states = []
        if config.use_scene_enc:
          # use the obs spatial grid to mask the convolution feature
          # [N, T, h, w, conv_dim]
          obs_grid_class_conv = tf.multiply(self.scene_convs[i], obs_grid_class)

          obs_grid_enc_h, obs_grid_enc_last_state = tf.nn.dynamic_rnn(
              enc_cell_obs_grid, obs_grid_class_conv,
              sequence_length=obs_length,
              dtype="float", scope="encoder_grid_class_%d" % i)
          # [N, H, W, h_dim]
          self.scene_enc_states.append(obs_grid_enc_last_state)
        else:
          # use the same grid2emb as in the decoder
          # [N, T, H, W, P] -> [N, T, H, W, dim]
          grid_emb = self.grid_emb(tf.reshape(obs_grid_class, [-1, h, w, 1]),
                                   output_size=config.emb_size,
                                   activation=config.activation_func,
                                   add_bias=True, is_conv=True,
                                   scope="grid_emb")  # also used in enc
          grid_emb = tf.reshape(grid_emb, [N, T, h, w, config.emb_size])
          obs_grid_enc_h, obs_grid_enc_last_state = tf.nn.dynamic_rnn(
              enc_cell_obs_grid, grid_emb, sequence_length=obs_length,
              dtype="float", scope="encoder_grid_class_%d" % i)

        # [N, T, h, w, 2]  ->  [N, T, h, w, h_dim]
        obs_grid_reg_enc_h, obs_grid_reg_enc_last_state = tf.nn.dynamic_rnn(
            enc_cell_obs_grid_reg, obs_grid_reg, sequence_length=obs_length,
            dtype="float", scope="encoder_grid_reg_%d" % i)

        dec_cell_grid = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2, input_shape=[h, w, config.enc_hidden_size],
            output_channels=config.dec_hidden_size,
            kernel_shape=[config.convlstm_kernel, config.convlstm_kernel],
            name="dec_grid_%d" % i)
        dec_cell_grid = tf.nn.rnn_cell.DropoutWrapper(
            dec_cell_grid, keep_prob)
        dec_cell_grid_reg = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2, input_shape=[h, w, config.enc_hidden_size],
            output_channels=config.dec_hidden_size,
            kernel_shape=[config.convlstm_kernel, config.convlstm_kernel],
            name="dec_grid_reg_%d" % i)
        dec_cell_grid_reg = tf.nn.rnn_cell.DropoutWrapper(
            dec_cell_grid_reg, keep_prob)

        # decoder [N, T_pred, h, w, 1/2]
        if config.use_soft_grid_class:
          grid_pred_labels_one_hot = self.grid_pred_labels_T[i]
        else:
          grid_pred_labels_one_hot = tf.one_hot(
              tf.cast(self.grid_pred_labels_T[i], "int32"), h*w)
          grid_pred_labels_one_hot = tf.reshape(
              grid_pred_labels_one_hot, [N, -1, h, w, 1])

        if config.use_beam_search:
          assert not config.is_train
          assert sum(config.use_grids) == 1, "only one scale test at a time"
          # logits: [N, beam_size, T, H*W]
          # ids: [N, beam_size, T]
          # logrpobs [N, beam_size]
          # best_beam: [N, T, H, W, 1]
          best_beam, outputs_logits, outputs_ids, logprobs, outputs_states = \
              self.grid_decoder_beam_search(
                  obs_grid_class[:, -1], obs_grid_enc_last_state,
                  pred_length, dec_cell_grid, grid_pred_labels_one_hot,
                  scale_idx=i,
                  top_scope=top_scope, scope="decoder_grid_class_%d" % i,
                  use_gnn=config.use_gnn,
                  save_output_states=config.use_single_decoder)
          grid_pred_decoded = best_beam
          self.beam_outputs = [outputs_logits, outputs_ids, logprobs]
        else:
          grid_pred_decoded, grid_pred_decoder_states = self.grid_decoder(
              obs_grid_class[:, -1], obs_grid_enc_last_state,
              pred_length, dec_cell_grid, grid_pred_labels_one_hot,
              scale_idx=i,
              top_scope=top_scope, scope="decoder_grid_class_%d" % i,
              teacher_forcing=config.use_teacher_forcing,
              use_gnn=config.use_gnn,
              input_onehot=not config.is_train or config.train_w_onehot)

        if config.use_single_decoder:
          # decode from classification hidden state
          # [N, T, h, w, h_dim] -> [N, T, h, w, 2] (conv,)
          if config.use_beam_search:
            # [N*beam_size, T, H, W, d]
            grid_pred_decoder_states = tf.reshape(
                outputs_states,
                [N*config.beam_size, -1, h, w, config.dec_hidden_size])
          grid_pred_reg_decoded = self.hidden2grid(
              grid_pred_decoder_states, is_conv=True, dim=2, scope="decode_reg")
        else:
          grid_pred_reg_decoded, _ = self.grid_decoder(
              obs_grid_reg[:, -1], obs_grid_reg_enc_last_state,
              pred_length, dec_cell_grid_reg, self.grid_pred_regress[i],
              top_scope=top_scope,
              scale_idx=i,
              scope="decoder_grid_reg_%d" % i,
              teacher_forcing=config.use_teacher_forcing, use_gnn=False,
              input_onehot=False)

        self.grid_pred_decoded.append(grid_pred_decoded)
        self.grid_pred_reg_decoded.append(grid_pred_reg_decoded)


  def grid_decoder(self, first_input, enc_last_state, pred_length, rnn_cell,
                   pred_gt, top_scope, scope, is_conv=False, scale_idx=0,
                   teacher_forcing=False, use_gnn=False, input_onehot=False):
    """Decoder definition."""

    # first_input -> [N, [H, W], P]
    config = self.config
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N
    P = first_input.get_shape().as_list()[-1]  # 1 or 2
    H, W = first_input.get_shape().as_list()[-3:-1]
    input_size = [N, H, W, P]
    perm = [1, 0, 2, 3, 4]
    if input_onehot:
      assert P == 1

    with tf.variable_scope(scope):
      # this is only used for training
      with tf.name_scope("prepare_pred_gt_training"):
        # these input only used during training
        time_1st_pred_gt = tf.transpose(
            pred_gt, perm=perm)  # [N, T2, H, W, P] -> [T2, N, H, W, P]
        T2 = tf.shape(time_1st_pred_gt)[0]  # T2
        pred_gt = tf.TensorArray(size=T2, dtype="float")
        pred_gt = pred_gt.unstack(
            time_1st_pred_gt)  # [T2] , [N, H, W, P]
        # need this for Python3 + tf 1.15,
        # otherwise ugly warning during inferencing since it is not used
        pred_gt.mark_used()

      # all None for first call
      with tf.name_scope("decoder_rnn"):
        def decoder_loop_fn(time, cell_output, cell_state, loop_state):
          """RNN loop function for the decoder."""
          emit_output = cell_output  # == None for time==0

          elements_finished = time >= pred_length
          finished = tf.reduce_all(elements_finished)

          # h_{t-1}  # [H, W, h_dim]
          with tf.name_scope("prepare_next_cell_state"):
            # LSTMStateTuple, will have c, h, c is the cell memory state and h
            # is c* output_gate and get h, h will be the next state input
            if cell_output is None:
              next_cell_state = enc_last_state
            else:
              next_cell_state = cell_state

            if use_gnn:

              # 1. compute edge weight,
              # [N, H, W, H, W]
              edge_weights = self.gnn_edge(next_cell_state.h,
                                           scale_idx,
                                           scope=top_scope,
                                           additional_scope="gnn_%s" % scope)
              # 2. mask non-neighbor nodes' edge weight
              #  non-neighbor location will have very negative number
              edge_weights = self.gnn_mask_edge(edge_weights, use_exp_mask=True)

              # 3. compute each nodes
              # [N, H, W, h_dim]
              node_states = self.gnn_node(next_cell_state.h, edge_weights,
                                          scope=top_scope,
                                          additional_scope="gnn_%s" % scope)

              # [H, W, h_dim] -> [H, W, h_dim]
              next_cell_state_h = next_cell_state.h + node_states

              # yikes
              next_cell_state = tf.nn.rnn_cell.LSTMStateTuple(
                  c=next_cell_state.c, h=next_cell_state_h)
          # x_t
          with tf.name_scope("prepare_next_input"):
            # cell_output is [N, [H, W], h_dim]
            if cell_output is None:  # first time
              next_input_grid = first_input  # the last observed [N, [H, W], P]
            else:
              # for testing, construct from this output to be next input
              if teacher_forcing:
                next_input_grid = tf.cond(
                    # first check the sequence finished or not
                    finished,
                    lambda: tf.zeros(input_size, dtype="float"),
                    # pylint: disable=g-long-lambda
                    lambda: tf.cond(
                        self.is_train,
                        # teacher forcing
                        lambda: pred_gt.read(time),
                        # hidden vector from last step to coordinates
                        lambda: self.hidden2grid(
                            cell_output, is_conv=True, dim=P,
                            scope=top_scope,
                            additional_scope="hidden2grid_%s" % scope))
                )
              else:
                if input_onehot:  # true for classification
                  # input the one-hot from last timestep prediction instead of
                  # the probability
                  # [N, H, W, 1]
                  def argmax_onehot(tensor):
                    tensor = tf.reshape(tensor, [-1, H*W])
                    tensor_argmax = tf.argmax(tensor, axis=1)
                    tensor_argmax_onehot = tf.one_hot(tensor_argmax, H*W)
                    return tf.reshape(tensor_argmax_onehot, [-1, H, W, 1])

                  next_input_grid = tf.cond(
                      # first check the sequence finished or not
                      finished,
                      lambda: tf.zeros(input_size, dtype="float"),
                      # pylint: disable=g-long-lambda
                      lambda: argmax_onehot(self.hidden2grid(
                          cell_output, is_conv=True, dim=P,
                          scope=top_scope,
                          additional_scope="hidden2grid_%s" % scope)))
                else:
                  next_input_grid = tf.cond(
                      # first check the sequence finished or not
                      finished,
                      lambda: tf.zeros(input_size, dtype="float"),
                      # pylint: disable=g-long-lambda
                      lambda: self.hidden2grid(
                          cell_output, is_conv=True, dim=P,
                          scope=top_scope,
                          additional_scope="hidden2grid_%s" % scope))



            # spatial embedding

            # [N, [H, W], P] -> [N, [H, W], dim]
            grid_emb = self.grid_emb(next_input_grid,
                                     output_size=config.emb_size,
                                     activation=config.activation_func,
                                     add_bias=True, is_conv=True,
                                     scope="grid_emb")  # also used in enc

            # graph attention?

            next_input = grid_emb

          return elements_finished, next_input, next_cell_state, \
              emit_output, None  # next_loop_state

        decoder_out_ta, _, _ = tf.nn.raw_rnn(
            rnn_cell, decoder_loop_fn, scope="decoder_rnn")

      with tf.name_scope("reconstruct_output"):
        decoder_out_h = decoder_out_ta.stack()  # [T2, N, [H, W], h_dim]
        # [N, T2, [H, W], h_dim]
        decoder_out_h = tf.transpose(decoder_out_h, perm=perm)

      # recompute the output;
      # if use loop_state to save the output, will 10x slower

      # use the same hidden2grid for different decoder
      decoder_out = self.hidden2grid(
          decoder_out_h, is_conv=True, dim=P, scope=top_scope,
          additional_scope="hidden2grid_%s" % scope)

    return decoder_out, decoder_out_h

  #
  def grid_decoder_beam_search(self, first_input, enc_last_state, pred_length, rnn_cell,
                   pred_gt, top_scope, scope, is_conv=False, scale_idx=0,
                   use_gnn=False, save_output_states=False):
    """decode grid with beam search. Adapted from
    https://github.com/guillaumegenthial/im2latex
    """

    # first_input -> [N, H, W, P]
    config = self.config
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N
    beam_size = self.beam_size
    P = first_input.get_shape().as_list()[-1]  # 1
    assert P == 1
    H, W = first_input.get_shape().as_list()[-3:-1]
    h_dim = enc_last_state.h.get_shape().as_list()[-1]
    pred_length_tile_beam = tf.tile(tf.expand_dims(
        pred_length, axis=1), [1, beam_size])
    pred_length_tile_beam = tf.reshape(pred_length_tile_beam, [-1])

    with tf.variable_scope(scope):
      with tf.name_scope("prepare_data"):
        # tile input & initial_state to beam
        initial_input = tf.tile(tf.expand_dims(first_input, axis=1),
                                [1, beam_size, 1, 1, 1])
        initial_cell_state = tf.nest.map_structure(
            lambda t: tf.tile(
                tf.expand_dims(t, axis=1), [1, beam_size, 1, 1, 1]),
            enc_last_state)
        # output to be saved during RNN decode loop
        # each timestep's new top_beam_size grid classid
        output_grid_ids = tf.TensorArray(
            dtype=tf.int32, size=0, dynamic_size=True)
        # selected beam idx to trace back to get the full sequence
        output_parent_idxs = tf.TensorArray(
            dtype=tf.int32, size=0, dynamic_size=True)
        # each timestep's logits [batch_size, beam_size, V]
        output_logits = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True)

        initial_logprob = tf.zeros([N, beam_size], dtype=tf.float32)


      # loop will run from time=0 to time=pred_length # note the extra 1
      def decoder_loop_fn(time, cell_output, cell_state, loop_state):
        """RNN loop function for the decoder."""
        elements_finished = time >= pred_length_tile_beam  # [N*beam_size]
        finished = tf.reduce_all(elements_finished)
        emit_output = cell_output
        if cell_output is None:
          # time==0
          # before first step
          # merge beam
          next_cell_state = tf.nest.map_structure(
              lambda t: tf.reshape(t, [N * beam_size, H, W, h_dim]),
              initial_cell_state)
          # obs_last, one-hot
          next_input_grid = tf.reshape(initial_input,
                                       [N * beam_size, H, W, 1])


          # put the output stuff into loop_state
          # also need a variable to remember each beam's total log prob
          next_loop_state = (output_grid_ids,
                             output_parent_idxs,
                             output_logits,
                             initial_logprob)

        else:
          (output_grid_ids_inloop,
           output_parent_idxs_inloop,
           output_logits_inloop,
           prev_logprob) = loop_state
          # compute the classification logits first
          # cell_output [N*beam_size, H, W, h_dim]
          # -> [N*beam_size, H, W, 1]
          this_output_logits = self.hidden2grid(
              cell_output, is_conv=True, dim=P,
              scope=top_scope,
              additional_scope="hidden2grid_%s" % scope)
          this_output_logits = tf.reshape(
              this_output_logits, [N, beam_size, H*W])
          #this_logprobs = this_output_logits
          this_logprobs = tf.nn.log_softmax(this_output_logits)

          # [N, beam_size] -> [N, beam_size, H*W]
          logprobs = tf.expand_dims(prev_logprob, axis=-1) + this_logprobs
          if config.diverse_beam:
            # encourage diverse parent beam
            # by adding negative penalties to dominent top rank logits
            # gamma should > 1.0
            logprobs = add_div_penalty(logprobs,
                                       config.diverse_gamma,  # div_gamma=
                                       N, beam_size, H*W)

          logprobs_flat = tf.reshape(logprobs, [N, beam_size * H*W])
          # if time = 0, consider only one beam since all logits the same
          # [N, H*W] or [N, beam_size*H*W]
          logprobs_flat = tf.cond(time > 1, lambda: logprobs_flat,
                                  lambda: logprobs[:, 0])

          # new_logprobs: [N, beam_size], top beam_size logprobs
          # indices: [N, beam_size] indices along the beam_size*H*W dimension
          # [N, 0] will always be the best beam
          new_logprobs, indices = tf.nn.top_k(logprobs_flat, beam_size,
                                              sorted=True)
          # keep the first few steps logits to be small
          new_logprobs = tf.cond(
              time > config.fix_num_timestep,
              lambda: new_logprobs,
              lambda: tf.zeros([N, beam_size], dtype=tf.float32))

          # [N, beam_size]
          # each new beam's best grid_id
          new_grid_ids = indices % int(H*W)
          # some beam from last may be used in multiple new beam
          # like [0,2,1,1,3] for beam=5
          new_parent_idxs = indices // int(H*W)

          # save the stuff
          output_grid_ids_inloop = output_grid_ids_inloop.write(
              time-1, new_grid_ids)
          output_parent_idxs_inloop = output_parent_idxs_inloop.write(
              time-1, new_parent_idxs)
          output_logits_inloop = output_logits_inloop.write(
              time-1, this_output_logits)

          # get new input and cell states for next round
          def onehot(tensor):
            tensor = tf.reshape(tensor, [-1])
            tensor_onehot = tf.one_hot(tensor, H*W)
            return tf.reshape(tensor_onehot, [-1, H, W, 1])
          next_input_grid = onehot(new_grid_ids)

          # new cell state
          # split the memory state and hidden state to
          # [N, beam_size, h, w, h_dim]
          next_cell_state = tf.nest.map_structure(
              lambda t: tf.reshape(t, [N, beam_size, H, W, h_dim]),
              cell_state)

          # gather the cell state according to original beam idx
          next_cell_state = tf.nest.map_structure(
              lambda t: gather_helper(t, new_parent_idxs, N, beam_size),
              next_cell_state)

          # merge beam back
          next_cell_state = tf.nest.map_structure(
              lambda t: tf.reshape(t, [N*beam_size, H, W, h_dim]),
              next_cell_state)


          next_loop_state = (output_grid_ids_inloop,
                             output_parent_idxs_inloop,
                             output_logits_inloop,
                             new_logprobs)

        if use_gnn:

          # 1. compute edge weight,
          # [N, H, W, H, W]
          edge_weights = self.gnn_edge(next_cell_state.h,
                                       scale_idx,
                                       tile_to_beam=True,
                                       scope=top_scope,
                                       additional_scope="gnn_%s" % scope)
          # 2. mask non-neighbor nodes' edge weight
          #  non-neighbor location will have very negative number
          edge_weights = self.gnn_mask_edge(edge_weights, use_exp_mask=True)

          # 3. compute each nodes
          # [N, H, W, h_dim]
          node_states = self.gnn_node(next_cell_state.h, edge_weights,
                                      scope=top_scope,
                                      additional_scope="gnn_%s" % scope)

          # [H, W, h_dim] -> [H, W, h_dim]
          next_cell_state_h = next_cell_state.h + node_states

          next_cell_state = tf.nn.rnn_cell.LSTMStateTuple(
              c=next_cell_state.c, h=next_cell_state_h)

          # todo(junweil): modify cell memory, too?


        # spatial embedding

        # [N, [H, W], P] -> [N, [H, W], dim]
        grid_emb = self.grid_emb(next_input_grid,
                                 output_size=config.emb_size,
                                 activation=config.activation_func,
                                 add_bias=True, is_conv=True,
                                 scope="grid_emb")  # also used in enc


        next_input = grid_emb

        return elements_finished, next_input, next_cell_state, \
            emit_output, next_loop_state

      output_states_ta, _, (
          output_grid_ids_ta,
          output_parent_idxs_ta,
          output_logits_ta,
          final_logprobs) = tf.nn.raw_rnn(rnn_cell,
                                          decoder_loop_fn,
                                          scope="decoder_rnn")

      # reconstruct the output
      # [Time, N, beam_size, ..]
      output_grid_ids = output_grid_ids_ta.stack()
      output_parent_idxs = output_parent_idxs_ta.stack()
      output_logits = output_logits_ta.stack()


      # trace back to reconstruct the actual sequence
      max_timestep = tf.shape(output_grid_ids)[0]
      # reverse the time dimension
      output_grid_ids_rt = tf.reverse(output_grid_ids, axis=[0])
      output_parent_idxs_rt = tf.reverse(output_parent_idxs, axis=[0])
      output_logits_rt = tf.reverse(output_logits, axis=[0])


      initial_time = tf.constant(0, dtype=tf.int32)
      initial_outputs_logits_ta = tf.TensorArray(dtype=tf.float32,
                                                 size=max_timestep)
      initial_outputs_ids_ta = tf.TensorArray(dtype=tf.int32,
                                              size=max_timestep)
      if save_output_states:
        # [T, N*beam_size, h, w, h_dim]
        output_states = output_states_ta.stack()
        output_states = tf.reshape(
            output_states, [-1, N, beam_size, H, W, h_dim])

        output_states_rt = tf.reverse(output_states, axis=[0])

      initial_outputs_states_ta = tf.TensorArray(dtype=tf.float32,
                                                 size=max_timestep)


      initial_parents = tf.tile(
          tf.expand_dims(tf.range(beam_size), axis=0),
          [N, 1])

      # final_logprobs is [N, beam_size]
      # first beam will always be the best beam
      final_logprobs = final_logprobs

      def condition(time, outputs_logits_ta, outputs_ids_ta, parents,
                    outputs_states_ta):
        return tf.less(time, max_timestep)

      # beam search decoding cell
      def body(time, outputs_logits_ta, outputs_ids_ta, parents,
               outputs_states_ta):
        # get ids, logits and parents predicted at time step by decoder
        input_grid_ids_t = output_grid_ids_rt[time]
        input_parent_idxs_t = output_parent_idxs_rt[time]
        input_logits_t = output_logits_rt[time]

        # extract the entries corresponding to parents
        new_grid_ids = gather_helper(input_grid_ids_t, parents, N, beam_size)
        new_parent_idxs = gather_helper(input_parent_idxs_t, parents, N,
                                        beam_size)
        new_logits = gather_helper(input_logits_t, parents, N, beam_size)

        # write beam ids
        outputs_logits_ta = outputs_logits_ta.write(time, new_logits)
        outputs_ids_ta = outputs_ids_ta.write(time, new_grid_ids)
        if save_output_states:
          input_states_t = output_states_rt[time]
          new_states = gather_helper(input_states_t, parents, N, beam_size)
          outputs_states_ta = outputs_states_ta.write(time, new_states)

        # continue to next
        parents = new_parent_idxs

        # https://github.com/guillaumegenthial/im2latex/issues/2?
        return (time + 1), outputs_logits_ta, outputs_ids_ta, \
            parents, outputs_states_ta

      res = tf.while_loop(
          condition,
          body,
          loop_vars=[
              initial_time,
              initial_outputs_logits_ta,
              initial_outputs_ids_ta,
              initial_parents,
              initial_outputs_states_ta],
          back_prop=False)
      final_outputs_logits_ta, final_outputs_ids_ta = res[1], res[2]
      final_outputs_logits_rt = final_outputs_logits_ta.stack()
      final_outputs_ids_rt = final_outputs_ids_ta.stack()
      # [T, N, beam_size, H*W]
      final_outputs_logits = tf.reverse(final_outputs_logits_rt, axis=[0])
      # [T, N, beam_size]
      final_outputs_ids = tf.reverse(final_outputs_ids_rt, axis=[0])

      # [N, beam_size, T, H*W]
      final_outputs_logits = tf.transpose(final_outputs_logits,
                                          perm=[1, 2, 0, 3])
      final_outputs_ids = tf.transpose(final_outputs_ids,
                                       perm=[1, 2, 0])
      if save_output_states:
        final_outputs_states_ta = res[4]
        final_outputs_states_rt = final_outputs_states_ta.stack()
        # [T, N, beam_size, H, W, d]
        final_outputs_states = tf.reverse(final_outputs_states_rt, axis=[0])
        # [N, beam_size, T, H, W, d]
        final_outputs_states = tf.transpose(final_outputs_states,
                                       perm=[1, 2, 0, 3, 4, 5])
        print(final_outputs_states.get_shape())
      else:
        final_outputs_states = None
      # select the best beam
      # no need, first beam is the best beam since use of top_k
      # [N]
      #best_beam_idxs = tf.argmax(final_logprobs, axis=1, output_type=tf.int32)
      #best_beam_idxs = tf.stack([tf.range(N, dtype=tf.int32), best_beam_idxs],
      #                          axis=1)

      # [N, T, H*W]
      #final_outputs_logits_best_beam = tf.gather_nd(final_outputs_logits,
      #                                              best_beam_idxs)
      final_outputs_logits_best_beam = final_outputs_logits[:, 0]

      # [N, T, H, W, P], same with greedy decoder output
      final_outputs_logits_best_beam = tf.reshape(
          final_outputs_logits_best_beam, [N, -1, H, W, P])

    return final_outputs_logits_best_beam, final_outputs_logits, \
        final_outputs_ids, final_logprobs, final_outputs_states

  def gnn_edge(self, input_states, scale_idx=None, additional_scope=None,
               tile_to_beam=False,
               scope="gnn_edge"):
    """Get edge weights."""
    config = self.config
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as this_scope:
      if additional_scope is not None:
        return self.gnn_edge(input_states, scale_idx=scale_idx,
                             tile_to_beam=tile_to_beam,
                             scope=additional_scope, additional_scope=None)
      # input_states [N, H, W, h_dim]
      #N = self.N
      N, H, W, h_dim = input_states.get_shape().as_list()

      input_states = tf.reshape(input_states, [N, H*W, h_dim])
      node_features = input_states
      if config.use_scene_enc:
        # scene_features [N, T, H, W, conv_dim]
        scene_features = self.scene_convs[scale_idx]
        # [N, H, W, conv_dim]
        scene_features = tf.reduce_mean(scene_features, axis=1)
        conv_dim = scene_features.get_shape().as_list()[-1]

        if tile_to_beam:
          beam_size = self.beam_size
          scene_features = tf.tile(tf.expand_dims(
              scene_features, axis=1), [1, beam_size, 1, 1, 1])

        scene_features = tf.reshape(scene_features, [N, H*W, conv_dim])
        # [N, K, node_feat_dim]
        node_features = tf.concat([input_states, scene_features], axis=-1)

      # self attention or other
      K = H * W
      node_features = tf.nn.l2_normalize(node_features, -1)
      """
      node_features_o1 = tf.tile(
          tf.expand_dims(node_features, axis=1), [1, K, 1, 1])
      node_features_o2 = tf.tile(
          tf.expand_dims(node_features, axis=2), [1, 1, K, 1])

      # [N, H*W, H*W, d] -> [N, H*W, H*W]
      edge_weights = tf.reduce_sum(
          tf.multiply(node_features_o1, node_features_o2), -1)
      """
      # [N, K, node_feat_dim] . [N, node_feat_dim, K] -> [N, K, K]
      node_features_o1 = tf.transpose(node_features, [0, 2, 1])
      edge_weights = tf.matmul(node_features, node_features_o1)
      edge_weights = tf.reshape(edge_weights, [N, H, W, H, W])

      return edge_weights

  def gnn_node(self, input_states, edge_weights, additional_scope=None,
               scope="gnn_edge"):
    """Just apply softmax."""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as this_scope:
      if additional_scope is not None:
        return self.gnn_node(input_states, edge_weights,
                             scope=additional_scope, additional_scope=None)

      # input_states [N, H, W, h_dim]
      # edge weights [N, H, W, H, W]
      N, H, W, h_dim = input_states.get_shape().as_list()
      input_states = tf.reshape(input_states, [N, H*W, h_dim])
      # [N, h_dim, H*W]
      edge_weights = tf.reshape(edge_weights, [N, H*W, H*W])

      edge_weights = softmax(edge_weights)

      # [N, H*W, H*W], [N, H*W, h_dim] -> [N, H*W, h_dim]
      summed_states = tf.matmul(edge_weights, input_states)

      summed_states = tf.reshape(summed_states, [N, H, W, h_dim])

      return summed_states


  def gnn_mask_edge(self, edge_weights, use_exp_mask=False):
    """Mask the weights according to spatial neighbors."""
    # [N, H, W, H, W]
    N, H, W, _, _ = edge_weights.get_shape().as_list()
    # a list from 0 to H*W
    mask = tf.range(start=0, limit=H*W, delta=1, dtype="int32")
    # [H*W, H*W], each H*W node has a spatial grid of H*W, where itself is 1
    mask = tf.one_hot(mask, H*W, dtype="float32")
    mask = tf.reshape(mask, [-1, H, W, 1])
    neighbors = tf.constant(1.0, shape=[3, 3], dtype="float32")
    neighbors = tf.reshape(neighbors, [3, 3, 1, 1])
    # [H*W, H, W, 1] -> [H*W, H, W, 1]
    nn_mask = tf.nn.conv2d(mask, filter=neighbors, strides=[1, 1, 1, 1],
                           padding="SAME",
                           data_format="NHWC")
    # [H*W, H*W], each H*W node has a spatial grid of H*W, where itself and the
    # neighbors are 1 while other location is zero
    nn_mask = tf.reshape(nn_mask, [H, W, H, W])
    nn_mask = tf.tile(tf.expand_dims(nn_mask, axis=0), [N, 1, 1, 1, 1])

    if use_exp_mask:
      edge_weights = exp_mask(edge_weights, nn_mask)
    else:
      edge_weights = tf.multiply(edge_weights, nn_mask)
    return edge_weights


  def grid_emb(self, input_grid, output_size, activation, add_bias, is_conv,
               scope):
    """Given grid input, get gird embedding."""
    if is_conv:
      # [N, H, W, 1/2] -> [N, H, W, output_size]
      return conv2d(input_grid, out_channel=output_size,
                    kernel=3, stride=1, activation=activation,
                    add_bias=add_bias, scope=scope)
    else:
      # [N, P] -> [N, output_size]
      return linear(input_grid, output_size=output_size, activation=activation,
                    add_bias=add_bias, scope=scope)

  def hidden2grid(self, lstm_h, return_scope=False, scope="hidden2grid",
                  additional_scope=None, is_conv=False, dim=1):
    """Hiddent states to grid output."""
    # Tensor dimensions, so pylint: disable=g-bad-name
    P = dim
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as this_scope:
      if additional_scope is not None:
        return self.hidden2grid(lstm_h, return_scope=return_scope,
                                is_conv=is_conv, dim=P,
                                scope=additional_scope, additional_scope=None)

      if is_conv:
        flatten_t = False
        if len(lstm_h.get_shape()) == 5:
          N, _, H, W, h_dim = lstm_h.get_shape()
          lstm_h = tf.reshape(lstm_h, [-1, H, W, h_dim])
          flatten_t = True

        elif len(lstm_h.get_shape()) > 5:
          raise ValueException(
              "hidden2grid input shape: %s" % (lstm_h.get_shape()))

        # [N, H, W, h_dim] -> [N, H, W, P]
        out = conv2d(lstm_h, out_channel=P,
                     kernel=3, stride=1, activation=tf.identity,
                     add_bias=False, scope="out_dec_grid")
        if flatten_t:
          out = tf.reshape(out, [N, -1, H, W, P])
      else:
        out = linear(lstm_h, output_size=P, activation=tf.identity,
                     add_bias=False, scope="out_dec_grid")

      if return_scope:
        return out, this_scope
      return out

  def build_loss(self):
    """Model loss."""
    config = self.config
    losses = []

    self.pred_grid_loss = []
    for i, (h, w) in enumerate(config.scene_grids):
      if not config.use_grids[i]:
        continue
      # gt
      # # [N, T] / [N, T, h, w, 1]
      grid_pred_class_labels = self.grid_pred_labels_T[i]
      # [N*T]
      if config.use_soft_grid_class:
        grid_pred_class_labels = tf.reshape(grid_pred_class_labels, [-1, h*w])
      else:
        grid_pred_class_labels = tf.reshape(grid_pred_class_labels, [-1])
      grid_pred_regress_targets = self.grid_pred_regress[i]  # [N, T, H, W, 2]

      # model output
      # [N, T, H, W, 1]
      grid_pred_decoded = self.grid_pred_decoded[i]
      # [N*T, H*W]
      grid_pred_decoded = tf.reshape(grid_pred_decoded, [-1, h*w])
      # one-hot label
      if config.use_soft_grid_class:
        # processed labels with maybe more than one entry > 0 (spatial filter)
        classification_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=grid_pred_class_labels, logits=grid_pred_decoded)
      else:
        classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(grid_pred_class_labels, "int32"),
            logits=grid_pred_decoded)

      classification_loss = tf.reduce_mean(classification_loss)

      # [N, T, H, W, 2]
      grid_pred_reg_decoded = self.grid_pred_reg_decoded[i]
      if config.mask_grid_regression:
        # masking out other prediction that is not the correct grid
        # get [K, 2]
        if not config.use_soft_grid_class:
          grid_pred_class_labels = tf.one_hot(
              tf.cast(grid_pred_class_labels, "int32"), h*w)
        # [N*T*h*w]
        grid_pred_class_labels = tf.reshape(grid_pred_class_labels, [-1])
        # [N*T*h*w, 2]
        grid_pred_reg_decoded = tf.reshape(grid_pred_reg_decoded, [-1, 2])
        grid_pred_regress_targets = tf.reshape(
            grid_pred_regress_targets, [-1, 2])

        fg_inds = tf.where(grid_pred_class_labels > 0)[:, 0]  # [K]
        fg_targets = tf.gather(grid_pred_regress_targets, fg_inds)
        fg_decoded = tf.gather(grid_pred_reg_decoded, fg_inds)

        regression_loss = tf.losses.huber_loss(
            labels=fg_targets, predictions=fg_decoded,
            reduction=tf.losses.Reduction.MEAN)
      else:
        regression_loss = tf.losses.huber_loss(
            labels=grid_pred_regress_targets, predictions=grid_pred_reg_decoded,
            reduction=tf.losses.Reduction.MEAN)

      classification_loss = classification_loss * \
        tf.constant(config.grid_loss_weight, dtype="float")
      regression_loss = regression_loss * \
        tf.constant(config.grid_reg_loss_weight, dtype="float")

      self.pred_grid_loss.extend([classification_loss, regression_loss])

      losses.extend([classification_loss, regression_loss])

    wd = wd_cost(".*/W", config.wd, scope="wd_cost")
    if wd:
      wd = tf.add_n(wd)
      losses.append(wd)
    self.wd_loss = wd

    # there might be l2 weight loss in some layer
    self.loss = tf.add_n(losses, name="total_losses")

  def get_feed_dict(self, batch, is_train=False):
    """Givng a batch of data, construct the feed dict."""
    # get the cap for each kind of step first
    config = self.config
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N
    P = 2

    T_in = config.obs_len
    T_pred = config.pred_len

    feed_dict = {}

    # initial all the placeholder

    obs_length = np.zeros((N), dtype="int32")
    pred_length = np.zeros((N), dtype="int32")
    feed_dict[self.obs_length] = obs_length
    feed_dict[self.pred_length] = pred_length
    for i in range(N):
      obs_length[i] = T_in
      pred_length[i] = T_pred

    feed_dict[self.is_train] = is_train

    data = batch.data
    # encoder features

    for j, (h, w) in enumerate(config.scene_grids):
      if not config.use_grids[j]:
        continue
      grid_obs_labels = np.zeros([N, T_in], dtype="int")
      grid_obs_reg_targets = np.zeros([N, T_in, h, w, 2], dtype="float")

      grid_pred_labels = np.zeros([N, T_pred], dtype="float")
      if config.use_soft_grid_class:
        grid_pred_labels = np.zeros([N, T_pred, h, w, 1], dtype="float")

      grid_pred_reg_targets = np.zeros([N, T_pred, h, w, 2], dtype="float")

      for i in range(len(data["obs_grid_class"])):

        grid_obs_labels[i, :] = data["obs_grid_class"][i][j, :]
        if config.use_soft_grid_class:
          # use a spatial convolution to turn one-hot to a neighborhood of
          # non-zero labels
          if config.soft_grid == 1:
            k = np.array([
                [0.1, 0.1, 0.1],
                [0.1, 1.0, 0.1],
                [0.1, 0.1, 0.1]], dtype="float")
          elif config.soft_grid == 2:
            k = np.array([
                [0.01, 0.01, 0.01],
                [0.01, 1.0, 0.01],
                [0.01, 0.01, 0.01]], dtype="float")
          elif config.soft_grid == 3:
            k = np.array([
                [0.05, 0.05, 0.05],
                [0.05, 1.0, 0.05],
                [0.05, 0.05, 0.05]], dtype="float")
          elif config.soft_grid == 4:  # best
            k = np.array([
                [0.0125, 0.0125, 0.0125],
                [0.0125, 0.9, 0.0125],
                [0.0125, 0.0125, 0.0125]], dtype="float")
          elif config.soft_grid == 5:
            k = np.array([
                [0.05, 0.05, 0.05],
                [0.05, 0.6, 0.05],
                [0.05, 0.05, 0.05]], dtype="float")
          elif config.soft_grid == 6:
            k = np.array([
                [0.1, 0.1, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1]], dtype="float")
          elif config.soft_grid == 7:  # best
            k = np.array([
                [0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
                [0.0625, 0.0125, 0.0125, 0.0125, 0.0625],
                [0.0625, 0.0125, 0.8, 0.0125, 0.0625],
                [0.0625, 0.0125, 0.0125, 0.0125, 0.0625],
                [0.0625, 0.0625, 0.0625, 0.0625, 0.0625]], dtype="float")
          # assert np.sum(k) == 1.0, k

          for t in range(T_pred):
            this_pred_label = np.zeros((h*w), dtype="float")
            this_pred_label[data["pred_grid_class"][i][j, t]] = 1.0
            this_pred_label = this_pred_label.reshape([h, w])
            this_pred_label = ndimage.convolve(this_pred_label, k,
                                               mode='constant', cval=0.0)
            this_pred_label = this_pred_label.reshape([h, w, 1])

            # [N, T, h, w, 1]
            grid_pred_labels[i, t] = this_pred_label
        else:
          grid_pred_labels[i, :] = data["pred_grid_class"][i][j, :]

        grid_obs_reg_targets[i, :, :, :, :] = \
          data["obs_grid_target_all_%d" % j][i][:, :, :, :]
        grid_pred_reg_targets[i, :, :, :, :] = \
          data["pred_grid_target_all_%d" % j][i][:, :, :, :]

      feed_dict[self.grid_obs_labels[j]] = grid_obs_labels

      feed_dict[self.grid_obs_regress[j]] = grid_obs_reg_targets

      if is_train or config.use_gt_grid:
        feed_dict[self.grid_pred_regress[j]] = grid_pred_reg_targets
        # note we use [N] pred_labels for activity location prediction
        feed_dict[self.grid_pred_labels_T[j]] = grid_pred_labels
      else:
        feed_dict[self.grid_pred_regress[j]] = np.zeros(
            [N, T_pred, h, w, 2], dtype="float")
        # note we use [N] pred_labels for activity location prediction
        if config.use_soft_grid_class:
          feed_dict[self.grid_pred_labels_T[j]] = np.zeros(
              [N, T_pred, h, w, 1], dtype="int")
        else:
          feed_dict[self.grid_pred_labels_T[j]] = np.zeros(
              [N, T_pred], dtype="int")


    # ---------------------------------------

    # scene input
    # the feature index
    obs_scene = np.zeros((N, T_in), dtype="int32")
    obs_scene_mask = np.zeros((N, T_in), dtype="bool")

    feed_dict[self.obs_scene] = obs_scene
    feed_dict[self.obs_scene_mask] = obs_scene_mask
    feed_dict[self.scene_feat] = data["batch_scene_feat"]

    # 02/14/2020, remove the following
    # and obs_mask, and other label, target that are not used
    # each bacth
    for i in range(len(data["batch_obs_scene"])):
      for j in range(len(data["batch_obs_scene"][i])):
        # it was (1) shaped
        obs_scene[i, j] = data["batch_obs_scene"][i][j][0]
        obs_scene_mask[i, j] = True

    # [N,num_scale, T] # each is int to num_grid_class
    for j, _ in enumerate(config.scene_grids):
      this_grid_label = np.zeros([N, T_in], dtype="int32")
      for i in range(len(data["obs_grid_class"])):
        this_grid_label[i, :] = data["obs_grid_class"][i][j, :]

      feed_dict[self.grid_obs_labels[j]] = this_grid_label

    # ----------------------------training
    return feed_dict


def add_div_penalty(log_probs, div_gamma, batch_size, beam_size,
      vocab_size):
  """Adds penalty to beam hypothesis following this paper by Li et al. 2016
  "A Simple, Fast Diverse Decoding Algorithm for Neural Generation"

  Args:
      log_probs: (tensor of floats)
          shape = (batch_size, beam_size, vocab_size)
      div_gamma: (float) diversity parameter
      div_prob: (float) adds penalty with proba div_prob

  """

  # 1. get indices that would sort the array
  top_probs, top_inds = tf.nn.top_k(log_probs, k=vocab_size, sorted=True)
  # 2. inverse permutation to get rank of each entry
  top_inds = tf.reshape(top_inds, [-1, vocab_size])
  # [0, 1, .... vocab_size]'s rank
  index_rank = tf.map_fn(tf.invert_permutation, top_inds, back_prop=False)
  index_rank = tf.reshape(index_rank, shape=[batch_size, beam_size,
          vocab_size])
  # 3. compute penalty
  # higher rank should have more positive values to add
  penalties = tf.log(div_gamma) * tf.cast(index_rank, log_probs.dtype)
  # 4. only apply penalty with some probability

  return log_probs + penalties

def gather_helper(t, indices, batch_size, beam_size):
  """
  Args:
      t: tensor of shape = [batch_size, beam_size, d]
      indices: tensor of shape = [batch_size, beam_size]

  Returns:
      new_t: tensor w shape as t but new_t[:, i] = t[:, new_parents[:, i]]

  """
  range_ = tf.expand_dims(tf.range(batch_size) * beam_size, axis=1)
  indices = tf.reshape(indices + range_, [-1])
  output = tf.gather(
      tf.reshape(t, [batch_size*beam_size, -1]),
      indices)

  if t.shape.ndims == 2:
    return tf.reshape(output, [batch_size, beam_size])

  elif t.shape.ndims == 3:
    d = t.shape[-1].value
    return tf.reshape(output, [batch_size, beam_size, d])
  elif t.shape.ndims == 5:
    d = t.shape[-1].value
    w = t.shape[-2].value
    h = t.shape[-3].value
    return tf.reshape(output, [batch_size, beam_size, h, w, d])

def wd_cost(regex, wd, scope):
  """Given regex to get the parameter to do regularization.

  Args:
    regex: regular expression
    wd: weight decay factor
    scope: variable scope
  Returns:
    Tensor
  """
  params = tf.trainable_variables()
  with tf.name_scope(scope):
    costs = []
    for p in params:
      para_name = p.op.name
      if re.search(regex, para_name):
        regloss = tf.multiply(tf.nn.l2_loss(p), wd, name="%s/wd" % p.op.name)
        assert regloss.dtype.is_floating, regloss
        if regloss.dtype != tf.float32:
          regloss = tf.cast(regloss, tf.float32)
        costs.append(regloss)

    return costs


def reconstruct(tensor, ref, keep):
  """Reverse the flatten function.

  Args:
    tensor: the tensor to operate on
    ref: reference tensor to get original shape
    keep: index of dim to keep

  Returns:
    Reconstructed tensor
  """
  ref_shape = ref.get_shape().as_list()
  tensor_shape = tensor.get_shape().as_list()
  ref_stop = len(ref_shape) - keep
  tensor_start = len(tensor_shape) - keep
  pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
  keep_shape = [tensor_shape[i] or tf.shape(tensor)[i]
                for i in range(tensor_start, len(tensor_shape))]
  # keep_shape = tensor.get_shape().as_list()[-keep:]
  target_shape = pre_shape + keep_shape
  out = tf.reshape(tensor, target_shape)
  return out


def flatten(tensor, keep):
  """Flatten a tensor.

  keep how many dimension in the end, so final rank is keep + 1
  [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]

  Args:
    tensor: the tensor to operate on
    keep: index of dim to keep

  Returns:
    Flattened tensor
  """
  # get the shape
  fixed_shape = tensor.get_shape().as_list()  # [N, JQ, di] # [N, M, JX, di]
  # len([N, JQ, di]) - 2 = 1 # len([N, M, JX, di] ) - 2 = 2
  start = len(fixed_shape) - keep
  # each num in the [] will a*b*c*d...
  # so [0] -> just N here for left
  # for [N, M, JX, di] , left is N*M
  left = reduce(operator.mul, [fixed_shape[i] or tf.shape(tensor)[i]
                               for i in range(start)])
  # [N, JQ,di]
  # [N*M, JX, di]
  out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i]
                        for i in range(start, len(fixed_shape))]
  # reshape
  flat = tf.reshape(tensor, out_shape)
  return flat


def conv2d(x, out_channel, kernel, padding="SAME", stride=1,
           activation=tf.identity, add_bias=True, data_format="NHWC",
           dilations=1,
           w_init=None, scope="conv"):
  """Convolutional layer."""
  # auto reuse needed for rnn loop
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    in_shape = x.get_shape().as_list()

    channel_axis = 3 if data_format == "NHWC" else 1
    in_channel = in_shape[channel_axis]

    assert in_channel is not None

    kernel_shape = [kernel, kernel]

    filter_shape = kernel_shape + [in_channel, out_channel]

    if data_format == "NHWC":
      stride = [1, stride, stride, 1]
      dilations = [1, dilations, dilations, 1]
    else:
      stride = [1, 1, stride, stride]
      dilations = [1, 1, dilations, dilations]

    if w_init is None:
      w_init = tf.variance_scaling_initializer(scale=2.0)
    # common weight tensor, so pylint: disable=g-bad-name
    W = tf.get_variable("W", filter_shape, initializer=w_init)

    conv = tf.nn.conv2d(x, W, stride, padding, dilations=dilations,
                        data_format=data_format)

    if add_bias:
      b_init = tf.constant_initializer()
      b = tf.get_variable("b", [out_channel], initializer=b_init)
      conv = tf.nn.bias_add(conv, b, data_format=data_format)

    ret = activation(conv, name="output")

  return ret


def softmax(logits, scope=None):
  """a flatten and reconstruct version of softmax."""
  with tf.name_scope(scope or "softmax"):
    flat_logits = flatten(logits, 1)
    flat_out = tf.nn.softmax(flat_logits)
    out = reconstruct(flat_out, logits, 1)
    return out


def softsel(target, logits, use_sigmoid=False, scope=None):
  """Apply attention weights."""

  with tf.variable_scope(scope or "softsel"):  # no new variable tho
    if use_sigmoid:
      a = tf.nn.sigmoid(logits)
    else:
      a = softmax(logits)  # shape is the same
    target_rank = len(target.get_shape().as_list())
    # [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
    # second last dim
    return tf.reduce_sum(tf.expand_dims(a, -1)*target, target_rank-2)


def exp_mask(val, mask):
  """Apply exponetial mask operation."""
  return tf.add(val, (1 - tf.cast(mask, "float")) * -1e30, name="exp_mask")


def linear(x, output_size, scope, add_bias=False, wd=None, return_scope=False,
           reuse=None, activation=tf.identity, keep=1, additional_scope=None):
  """Fully-connected layer."""
  with tf.variable_scope(scope or "xy_emb", reuse=tf.AUTO_REUSE) as this_scope:
    if additional_scope is not None:
      return linear(x, output_size, scope=additional_scope, add_bias=add_bias,
                    wd=wd, return_scope=return_scope, reuse=reuse,
                    activation=activation, keep=keep, additional_scope=None)
    # since the input here is not two rank,
    # we flat the input while keeping the last dims
    # keeping the last one dim # [N,M,JX,JQ,2d] => [N*M*JX*JQ,2d]
    flat_x = flatten(x, keep)
    # print flat_x.get_shape() # (?, 200) # wd+cwd
    bias_start = 0.0
    # need to be get_shape()[k].value
    if not isinstance(output_size, int):
      output_size = output_size.value

    def init(shape, dtype, partition_info):
      dtype = dtype
      partition_info = partition_info
      return tf.truncated_normal(shape, stddev=0.1)
    # Common weight tensor name, so pylint: disable=g-bad-name
    W = tf.get_variable("W", dtype="float", initializer=init,
                        shape=[flat_x.get_shape()[-1].value, output_size])
    flat_out = tf.matmul(flat_x, W)
    if add_bias:
      # disable=unused-argument
      def init_b(shape, dtype, partition_info):
        dtype = dtype
        partition_info = partition_info
        return tf.constant(bias_start, shape=shape)

      bias = tf.get_variable(
          "b", dtype="float", initializer=init_b, shape=[output_size])
      flat_out += bias

    flat_out = activation(flat_out)

    out = reconstruct(flat_out, x, keep)
    if return_scope:
      return out, this_scope
    else:
      return out



def focal_attention(query, context, parametric=False, use_sigmoid=False,
                    scope=None):
  """Focal attention layer.

  Args:
    query : [N, dim1]
    context: [N, num_channel, T, dim2],
    parametric: whether not to use simple cosine similarity function
    use_sigmoid: use sigmoid instead of softmax
    scope: variable scope

  Returns:
    Tensor
  """
  with tf.variable_scope(scope or "attention", reuse=tf.AUTO_REUSE):
    # Tensor dimensions, so pylint: disable=g-bad-name
    _, d = query.get_shape().as_list()
    _, K, _, d2 = context.get_shape().as_list()
    assert d == d2

    T = tf.shape(context)[2]

    # [N,d] -> [N,K,T,d]
    query_aug = tf.tile(tf.expand_dims(
        tf.expand_dims(query, 1), 1), [1, K, T, 1])

    if parametric:
      # [N, K, T, 1]
      a_logits = linear(tf.concat([(query_aug-context)*(query_aug-context),
                                   query_aug*context], 3),
                        output_size=1, activation=tf.nn.tanh,
                        scope="att_logits")
      a_logits = tf.squeeze(a_logits, 3)
    else:
      # cosine simi
      query_aug_norm = tf.nn.l2_normalize(query_aug, -1)
      context_norm = tf.nn.l2_normalize(context, -1)
      # [N, K, T]
      a_logits = tf.reduce_sum(tf.multiply(query_aug_norm, context_norm), 3)

    a_logits_maxed = tf.reduce_max(a_logits, 2)  # [N,K]

    attended_context = softsel(softsel(context, a_logits,
                                       use_sigmoid=use_sigmoid), a_logits_maxed,
                               use_sigmoid=use_sigmoid)

    return attended_context


def concat_states(state_tuples, axis):
  """Concat LSTM states."""
  return tf.nn.rnn_cell.LSTMStateTuple(c=tf.concat([s.c for s in state_tuples],
                                                   axis=axis),
                                       h=tf.concat([s.h for s in state_tuples],
                                                   axis=axis))


# ----------------07/2019
# added resnet
# code modified from https://github.com/JunweiLiang/Object_Detection_Tracking
def resnet_group(l, num_channel, num_block, stride, dilations=1,
                 modified_block_num=3, use_gn=False, scope="resnet_group"):
  """ResNet Group."""

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    for i in range(0, num_block):
      with tf.variable_scope("block{}".format(i)):
        dilations_ = 1
        # so there are num_block, only block==last modified_block_num will be
        # modified to dilated CNN or deformable CNN
        if i in range(num_block)[-modified_block_num:]:
          dilations_ = dilations

        l = resnet_bottleneck(
            l, num_channel, stride if i == 0 else 1, dilations=dilations_,
            use_gn=use_gn)

        l = tf.nn.relu(l)
  return l


# 3 conv layer and a conv residual connection
# stride is put at the second conv layer
def resnet_bottleneck(l, ch_out, stride, dilations=1, use_gn=False):
  """Resnet Block Function."""

  shortcut = l
  if use_gn:
    normalization_func = group_norm_relu
  else:
    normalization_func = just_relu

  # ------------------- conv1, 1x1
  l = conv2d(l, ch_out, 1, activation=normalization_func, scope="conv1",
             add_bias=False, data_format="NHWC")

  # ------------------- conv2, 3x3

  l = conv2d(l, ch_out, 3, dilations=dilations, stride=stride,
             activation=normalization_func, scope="conv2", add_bias=False,
             data_format="NHWC")

  # ------------------- conv3, 1x1
  l = conv2d(l, ch_out * 4, 1, activation=normalization_func, scope="conv3",
             add_bias=False, data_format="NHWC")

  return l + resnet_shortcut(shortcut, ch_out * 4, stride,
                             activation=normalization_func, data_format="NHWC")


def just_relu(x, name=None):
  x = tf.nn.relu(x, name=name)
  return x


def group_norm_relu(x, name=None):
  """Group norm then Relu."""

  x = group_norm(x, scope="gn")
  x = tf.nn.relu(x, name=name)
  return x


# from github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN
# assuming NHWC
def group_norm(x, group=32, gamma_init=tf.constant_initializer(1.), scope="gn"):
  """Group normalization by Yuxin Wu."""
  with tf.variable_scope(scope):

    x = tf.transpose(x, [0, 3, 1, 2])  # [NHWC] -> [NCHW]

    shape = x.get_shape().as_list()
    ndims = len(shape)

    assert ndims == 4, shape

    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable("beta", [chan],
                           initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable("gamma", [chan], initializer=gamma_init)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5,
                                    name="output")

    out = tf.reshape(out, orig_shape, name="output")

    # back to NHWC
    out = tf.transpose(out, [0, 2, 3, 1])
    return out


def resnet_shortcut(l, n_out, stride, activation=tf.identity,
                    data_format="NCHW"):
  """Residual link."""

  n_in = l.get_shape().as_list()[1 if data_format == "NCHW" else 3]
  if n_in != n_out:
    if stride == 2:
      l = l[:, :, :-1, :-1]
      return conv2d(l, n_out, 1, stride=stride, padding="VALID",
                    activation=activation, add_bias=False,
                    data_format=data_format, scope="convshortcut")
    else:
      return conv2d(l, n_out, 1, stride=stride, activation=activation,
                    add_bias=False, data_format=data_format,
                    scope="convshortcut")
  else:
    return l


class Trainer(object):
  """Trainer class for model."""

  def __init__(self, model, config):
    self.config = config
    self.model = model  # this is an model instance

    self.global_step = model.global_step

    learning_rate = config.init_lr

    if config.use_cosine_lr:
      max_steps = int(config.train_num_examples /
                      config.batch_size * config.num_epochs)
      learning_rate = tf.train.cosine_decay(
        config.init_lr,
        self.global_step,
        max_steps,
        alpha=0.0
      )
    elif config.learning_rate_decay is not None:
      decay_steps = int(config.train_num_examples /
                        config.batch_size * config.num_epoch_per_decay)

      learning_rate = tf.train.exponential_decay(
          config.init_lr,
          self.global_step,
          decay_steps,  # decay every k samples used in training
          config.learning_rate_decay,
          staircase=True)

    if config.optimizer == "momentum":
      opt = tf.train.MomentumOptimizer(
          learning_rate*config.emb_lr, momentum=0.9)
      #opt_rest = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif config.optimizer == "adadelta":
      opt = tf.train.AdadeltaOptimizer(learning_rate*config.emb_lr)
      #opt_rest = tf.train.AdadeltaOptimizer(learning_rate)
    elif config.optimizer == "adam":
      opt = tf.train.AdamOptimizer(learning_rate*config.emb_lr)
      #opt_rest = tf.train.AdamOptimizer(learning_rate)
    elif config.optimizer == "rmsprop":
      opt = tf.train.RMSPropOptimizer(learning_rate*config.emb_lr)
      #opt_rest = tf.train.RMSPropOptimizer(learning_rate)
    else:
      raise Exception("Optimizer not implemented")

    # losses
    self.loss = model.loss  # get the loss funcion
    self.wd_loss = model.wd_loss

    # valist for embding layer
    """
    var_emb = [var for var in tf.trainable_variables()
               if "emb" in var.name]
    var_rest = [var for var in tf.trainable_variables()
                if "emb" not in var.name]
    """
    var = tf.trainable_variables()

    # for training, we get the gradients first, then apply them
    # self.grads = tf.gradients(self.loss, var_emb+var_rest)
    self.grads = tf.gradients(self.loss, var)

    if config.clip_gradient_norm is not None:
      # pylint: disable=g-long-ternary
      self.grads = [grad if grad is None else
                    tf.clip_by_value(grad, -1*config.clip_gradient_norm,
                                     config.clip_gradient_norm)
                    for grad in self.grads]

    """
    grads_emb = self.grads[:len(var_emb)]
    grads_rest = self.grads[len(var_emb):]

    train_emb = opt_emb.apply_gradients(zip(grads_emb, var_emb))
    train_rest = opt_rest.apply_gradients(
        zip(grads_rest, var_rest), global_step=self.global_step)
    self.train_op = tf.group(train_emb, train_rest)
    """
    self.train_op = opt.apply_gradients(zip(self.grads, var),
                                        global_step=self.global_step)

  def step(self, sess, batch):
    """One training step."""
    config = self.config
    # idxs is a tuple (23,123,33..) index for sample
    _, batch_data = batch
    feed_dict = self.model.get_feed_dict(batch_data, is_train=True)
    inputs = [self.loss, self.train_op, self.wd_loss]
    num_out = 3

    inputs += [self.model.pred_grid_loss]
    num_out += 1


    outputs = sess.run(inputs, feed_dict=feed_dict)

    offset = 3
    loss, train_op, wd_loss = outputs[:offset]

    pred_grid_loss = outputs[offset]
    offset += 1


    return loss, train_op, wd_loss,  \
           pred_grid_loss


class Tester(object):
  """Tester for model."""

  def __init__(self, model, config, sess=None):
    self.config = config
    self.model = model

    self.sess = sess

    #assert len(config.scene_grids) == 2
    # [num_grids][N, T_pred, h, w, 1/2]
    self.grid_pred_decoded = self.model.grid_pred_decoded
    self.grid_pred_reg_decoded = self.model.grid_pred_reg_decoded

    self.beam_outputs = self.model.beam_outputs

  def step(self, sess, batch):
    """One inferencing step."""
    config = self.config
    # give one batch of Dataset, use model to get the result,
    _, batch_data = batch
    feed_dict = self.model.get_feed_dict(batch_data, is_train=False)

    beam_outputs = None

    inputs = []

    num_out = 0

    inputs += self.grid_pred_decoded + self.grid_pred_reg_decoded
    num_out += int(2 * len(config.scene_grids))

    if config.use_beam_search:
      inputs.append(self.beam_outputs)
    outputs = sess.run(inputs, feed_dict=feed_dict)

    offset = 0

    grid_pred_class = outputs[offset:offset+len(config.scene_grids)]
    grid_pred_reg = outputs[
        offset+len(config.scene_grids):offset+int(2*len(config.scene_grids))]

    if config.use_beam_search:
      beam_outputs = outputs[-1]

    return grid_pred_class, grid_pred_reg, beam_outputs
