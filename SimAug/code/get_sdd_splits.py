# coding=utf-8
# get n fold cross validation splits for Sdd
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("videolst")
parser.add_argument("splitpath")
parser.add_argument("--n_fold", default=5, type=int)

if __name__ == "__main__":
  args = parser.parse_args()
  videos = [os.path.basename(line.strip())
            for line in open(args.videolst).readlines()]
  random.shuffle(videos)
  random.shuffle(videos)

  folds = [videos[i::args.n_fold] for i in range(args.n_fold)]

  def write_list(filepath, fold):
    with open(filepath, "w") as f:
      for video in fold:
        f.writelines("%s\n" % video)

  for i, test_fold in enumerate(folds):
    target_path = os.path.join(args.splitpath, "fold_%d" % (i+1))
    if not os.path.exists(target_path):
      os.makedirs(target_path)

    test_list = os.path.join(target_path, "test.lst")

    write_list(test_list, test_fold)
    val_fold = []
    train_fold = []
    for j in range(args.n_fold):
      if j != i:
        if not val_fold:
          val_fold = folds[j]
        else:
          train_fold += folds[j]

    val_list = os.path.join(target_path, "val.lst")
    write_list(val_list, val_fold)
    train_list = os.path.join(target_path, "train.lst")
    write_list(train_list, train_fold)

