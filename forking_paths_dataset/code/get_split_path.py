# coding=utf-8
"""Given the anchor video dataset/multi future dataset, getdata split."""
import argparse
import os
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("video_path")
parser.add_argument("split_path")
parser.add_argument("--is_anchor", action="store_true")
parser.add_argument("--ori_split_path", default=None)


def write_lst(file_path, lst):
  with open(file_path, "w") as f:
    for one in lst:
      f.writelines("%s\n" % one)

if __name__ == "__main__":
  args = parser.parse_args()

  all_data_videonames = glob(os.path.join(args.video_path, "*.mp4"))
  all_data_videonames = [os.path.splitext(os.path.basename(l))[0]
                         for l in all_data_videonames]

  if not os.path.exists(args.split_path):
    os.makedirs(args.split_path)

  train_lst = []
  val_lst = []
  test_lst = []
  if not args.is_anchor:
    test_lst = all_data_videonames
  else:
    # load the original splits
    filelst = {
        "train": [os.path.splitext(os.path.basename(line.strip()))[0]
                  for line in open(os.path.join(args.ori_split_path,
                                                "train.lst"), "r").readlines()],
        "val": [os.path.splitext(os.path.basename(line.strip()))[0]
                for line in open(os.path.join(args.ori_split_path,
                                              "val.lst"), "r").readlines()],
        "test": [os.path.splitext(os.path.basename(line.strip()))[0]
                 for line in open(os.path.join(args.ori_split_path,
                                               "test.lst"), "r").readlines()],
    }
    # check each video
    for videoname in all_data_videonames:
      virat_videoname, _ = videoname.split("_F_")
      if virat_videoname in filelst["train"]:
        train_lst.append(videoname)
      elif virat_videoname in filelst["val"]:
        val_lst.append(videoname)
      elif virat_videoname in filelst["test"]:
        test_lst.append(videoname)
      else:
        print("%s not in all lst" % videoname)
  print("original %s videos, split into train %s, val %s, test %s" % (
      len(all_data_videonames), len(train_lst), len(val_lst), len(test_lst)))

  write_lst(os.path.join(args.split_path, "train.lst"), train_lst)
  write_lst(os.path.join(args.split_path, "val.lst"), val_lst)
  write_lst(os.path.join(args.split_path, "test.lst"), test_lst)

