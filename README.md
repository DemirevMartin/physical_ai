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

Zeroing the depth channel at inference collapses early fusion
(Car 0.310 -> 0.027) but barely moves cross-attention (0.182 -> 0.156): the
cross-attention model learned to ignore its LiDAR branch. Measuring its fusion
weights directly confirms this, as their normalised entropy is 0.9871, i.e.
effectively uniform.

## Data

**Committed here:** the annotations and everything derived from them:
labels (`training/label_2/`), calibration (`data_object_calib/training/calib/`),
the KITTI devkit, the trained checkpoints, and the figures. These are small
text and binary files, ~50 MB in total, and every result depends on them.

**Not committed:** the two large sensor modalities, ~26 GB together. Download
them from
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>
(registration required) and extract into the project root:

```
data_object_image_2.zip   ->  data_object_image_2/training/image_2/   (12 GB)
data_object_velodyne.zip  ->  training/velodyne/                      (14 GB)
```

The depth cache (`processed/depth/`, ~7 GB) is also excluded because it is
rebuilt from those two by the notebook, please check the section Running.

**For the label and calibration formats, please read `devkit_object/readme.txt`**
provided by the official KITTI documentation. It defines the 15 label columns, the
`DontCare` regions, the difficulty criteria, and the projection chain
`P2 * R0_rect * Tr_velo_to_cam` that this project implements and verifies.

## Running

```
pip install numpy pandas matplotlib pillow torch torchvision pyarrow
```

In `assignment.ipynb`, set `RUN_PRECOMPUTE = True` and run that cell once to
build the depth cache (~7 GB, ~15 min), then set it back to `False`. The
trained checkpoints are committed, so everything else reproduces without
retraining.

Both `RUN_PRECOMPUTE` and `RUN_TRAINING` default to `False`, so running all
cells never retrains or overwrites the committed checkpoints. Set
`RUN_TRAINING = True` only to retrain from scratch.
