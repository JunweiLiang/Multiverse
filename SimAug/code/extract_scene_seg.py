# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Given a list of images, run scene semantic segmentation using deeplab."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
import os
import argparse
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("imglst")
parser.add_argument("model_path", help="path to the model .pb file")
parser.add_argument("out_path")
parser.add_argument("--save_two_level", action="store_true")
parser.add_argument("--every", type=int, default=1,
                    help="scene semantic segmentation doesn't have to be"
                         " run on every frame")
parser.add_argument("--down_rate", default=8.0, type=float,
                    help="down-size how many times")
parser.add_argument("--keep_full", action="store_true",
                    help="get 512x288 feature")

# ---- gpu stuff. Now only one gpu is used
parser.add_argument("--gpuid", default=0, type=int)

# For running parallel jobs, set --job 4 --curJob k, where k=1/2/3/4
parser.add_argument("--job", type=int, default=1, help="total job")
parser.add_argument("--curJob", type=int, default=1,
                    help="this script run job Num")

# ade20k -> 150 + 1 (bg) classes
# city -> 18 + 1


def resize_seg_map(seg, down_rate, keep_full=False):
  img_ = Image.fromarray(seg.astype(dtype=np.uint8))
  w_, h_ = img_.size
  neww, newh = int(w_ / down_rate), int(h_ / down_rate)
  if keep_full:
    neww, newh = 512, 288

  newimg = img_.resize((neww, newh))  # neareast neighbor

  newdata = np.array(newimg)
  return newdata


if __name__ == "__main__":
  args = parser.parse_args()

  input_size = 513  # the model's input size, has to be this

  # load the model graph
  print("loading model...")
  graph = tf.Graph()
  with graph.as_default():
    gd = tf.GraphDef()
    with tf.gfile.GFile(args.model_path, "rb") as f:
      sg = f.read()
      gd.ParseFromString(sg)
      tf.import_graph_def(gd, name="")

    input_tensor = graph.get_tensor_by_name("ImageTensor:0")
    output_tensor = graph.get_tensor_by_name("SemanticPredictions:0")

  print("loaded.")

  if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

  imgs = [one.strip()
          for one in open(args.imglst, "r").readlines()][::args.every]

  tfconfig = tf.ConfigProto()
  tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i for i in [args.gpuid]]))

  skipped = 0
  with graph.as_default():
    with tf.Session(graph=graph, config=tfconfig) as sess:
      count = 0
      for img in tqdm(imgs):
        count += 1
        if (count % args.job) != (args.curJob - 1):
          continue
        imgname = os.path.splitext(os.path.basename(img))[0]

        target_path = args.out_path
        if args.save_two_level:
          # assuming the first part is the video name
          target_path = os.path.join(args.out_path, imgname.split("_F_")[0])

        if not os.path.exists(target_path):
          os.makedirs(target_path)

        targetfile = os.path.join(target_path, "%s.npy" % imgname)
        if os.path.exists(targetfile):
          skipped += 1
          continue

        ori_img = Image.open(img)

        w, h = ori_img.size
        resize_r = 1.0 * input_size / max(w, h)
        target_size = (int(resize_r * w), int(resize_r * h))
        resize_img = ori_img.convert("RGB").resize(target_size, Image.ANTIALIAS)

        seg_map, = sess.run([output_tensor],
                            feed_dict={input_tensor: [np.asarray(resize_img)]})
        seg_map = seg_map[0]  # single image input test

        # print seg_map.shape
        # print seg_map
        """
        (288, 513)
        [[ 8  8  8 ...  8  8  8]
         [ 8  8  8 ...  8  8  8]
         [ 8  8  8 ...  8  8  8]
         ...
         [11 11 11 ... 11 11 11]
         [11 11 11 ... 11 11 11]
         [11 11 11 ... 11 11 11]]

        """

        seg_map = resize_seg_map(seg_map, args.down_rate, args.keep_full)
        np.save(targetfile, seg_map)
  print("skipped %s" % skipped)







