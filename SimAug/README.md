# SimAug

This repository contains the code and models for the following ECCV'20 paper:

**[SimAug: Learning Robust Representations from Simulation for Trajectory Prediction](https://arxiv.org/abs/2004.02022)** \
[Junwei Liang](https://www.cs.cmu.edu/~junweil/),
[Lu Jiang](http://www.lujiang.info/),
[Alexander Hauptmann](https://www.cs.cmu.edu/~alex/)

You can find more information at our [Project Page](https://next.cs.cmu.edu/simaug/).

If you find this code useful in your research then please cite

```
@inproceedings{liang2020simaug,
  title={SimAug: Learning Robust Representations from Simulation for Trajectory Prediction},
  author={Liang, Junwei and Jiang, Lu and Hauptmann, Alexander},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  month = {August},
  year={2020}
}
```

# Introduction
<div align="center">
  <div style="">
      <img src="../images/prob_simaug.gif" height="300px" />
  </div>
  <p style="font-weight:bold;font-size:1.2em;">
    SimAug - Simulation as Augmentation for Trajectory Prediction
  </p>
</div>

This paper studies the problem of predicting future trajectories of people in unseen cameras of novel scenarios and views. We approach this problem through the real-data-free setting in which the model is trained only on 3D simulation data and applied out-of-the-box to a wide variety of real cameras. We propose a novel approach to learn robust representation through augmenting the simulation training data such that the representation can better generalize to unseen real-world test data. The key idea is to mix the feature of the hardest camera view with the adversarial feature of the original view. We refer to our method as **SimAug**. We show that SimAug achieves promising results on three real-world benchmarks using zero real training data, and state-of-the-art performance in the Stanford Drone and the VIRAT/ActEV dataset when using in-domain training data. Checkout our ECCV'20 presentation [here](https://www.youtube.com/watch?v=m6Jd99qUazc).

# Dataset

Here we provide the link to the multi-view trajectory dataset for download.

+ Download links: [Google Drive](https://drive.google.com/file/d/1AgMXXI7VKcB9sqvuWnkW7RlWJOtxlq-Y/view?usp=sharing)
[Baidu Pan](https://pan.baidu.com/s/1nuc726hX8bUBXmMRj6UBJw) (提取码: tpd7)

+ The dataset includes 5628 1920x1080 videos (1407 reconstructed trajectory samples in 4 camera views) with bounding boxes and scene semantic segmentation ground truth.

<div align="center">
  <div style="">
      <img src="../images/multi_view_anchor.gif" height="255px" />
      <img src="../images/eccv2020_data.png" height="255px" />
  </div>
  <p style="font-weight:bold;font-size:1.2em;">
    Multi-view trajectories reconstructed from the VIRAT dataset.
  </p>
</div>

# The SimAug Algorithm

<div align="center">
  <div style="">
      <img src="../images/eccv2020_model.png" height="300px" />
  </div>
  <br/>
</div>
