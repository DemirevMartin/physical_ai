# Multi-sensor pedestrian detection on KITTI

Detecting pedestrians and cars by fusing the left colour camera with the
Velodyne LiDAR, and comparing two fusion designs under an identical protocol.

## Notebooks

| notebook | contents |
|---|---|
| `exploration.ipynb` | the dataset: classes, distances, occlusion, LiDAR overlay |
| `assignment.ipynb` | the pipeline: geometry, split, targets, loss, both models, results |
| `verification.ipynb` | three independent checks that the pipeline is correct |

## Results

Validation split, interpolated AP at IoU >= 0.5:

| model | Car | Pedestrian |
|---|---|---|
| early fusion (depth as a 4th input channel) | 0.310 | 0.059 |
| cross-attention (separate encoders, learned fusion) | 0.182 | 0.019 |

Validation is by held-out drive, not by random frames: KITTI frames come from
continuous 10 Hz sequences, so a random split would put 592 validation frames
within 0.1 s of a training frame. Splitting by drive puts zero.

Zeroing the depth channel at inference collapses early fusion
(Car 0.310 -> 0.027) but barely moves cross-attention (0.182 -> 0.156).
Measuring the fusion weights directly points the same way: their normalised
entropy is 0.987, so each image cell adds roughly the same average of LiDAR
values. The attention does respond to depth content, but too weakly to affect
the output.


## Data

**Committed here:** all of the KITTI text data, namely the labels
(`training/label_2/`) and the calibration (`data_object_calib/`, both the
training and testing splits), together with the KITTI devkit, the trained
checkpoints and the figures. Roughly 60 MB in total, and every result in the
notebooks depends on it.

**Not committed:** the two large binary modalities, ~26 GB together. GitHub
caps a repository at 5 GB, and the point clouds are float32 data that barely
compresses. Download them from
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>
(registration required) and extract into the project root:

```
data_object_image_2.zip   ->  data_object_image_2/training/image_2/   (12 GB)
data_object_velodyne.zip  ->  training/velodyne/                      (13.6 GB)
```

They are needed to re-run anything that reads images or LiDAR. The committed
checkpoints, logs and figures mean the results themselves can be inspected
without downloading them.

`processed/` is committed except for `processed/depth/`: the checkpoints,
training logs and the overfit-test result are in, while the ~7 GB depth cache
is excluded because the notebook rebuilds it from the two archives above,
please check the section Running.

**For the label and calibration formats, please read `devkit_object/readme.txt`**
provided by the official KITTI documentation. It defines the 15 label columns, the
`DontCare` regions, the difficulty criteria, and the projection chain
`P2 * R0_rect * Tr_velo_to_cam` that this project implements and verifies.

## Running

Developed on Python 3.13 with a CUDA GPU.

```
pip install numpy pandas matplotlib pillow pyarrow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The second line installs a CUDA 12.8 build; a plain `pip install torch` gives a
CPU-only build, which runs but will not use a GPU. Match the index URL to your
driver's CUDA version.

In `assignment.ipynb`, set `RUN_PRECOMPUTE = True` and run that cell once to
build the depth cache (~7 GB, ~15 min), then set it back to `False`. The
trained checkpoints are committed, so everything else reproduces without
retraining.

Both `RUN_PRECOMPUTE` and `RUN_TRAINING` default to `False`, so running all
cells never retrains or overwrites the committed checkpoints. Set
`RUN_TRAINING = True` only to retrain from scratch.

## Data licence and attribution

The KITTI files redistributed here (labels, calibration, devkit) are from the
KITTI Vision Benchmark Suite and are licensed CC BY-NC-SA 3.0. Non-commercial
use only.

Geiger, Lenz and Urtasun, *Are We Ready for Autonomous Driving? The KITTI
Vision Benchmark Suite*, CVPR 2012.
