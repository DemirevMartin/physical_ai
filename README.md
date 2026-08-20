# Multi-sensor pedestrian detection on KITTI (camera + LiDAR)

Student project: detecting pedestrians and cars by fusing the left colour
camera with the Velodyne LiDAR of the KITTI object-detection benchmark, and
comparing two fusion designs under an identical protocol.

## Notebooks

| notebook | contents |
|---|---|
| `exploration.ipynb` | the dataset: classes, distances, occlusion, LiDAR overlay |
| `assignment.ipynb` | the pipeline: geometry, split, targets, loss, both models, results |
| `verification.ipynb` | three independent checks that the pipeline is correct |

## Results (validation split, interpolated AP at IoU >= 0.5)

| model | Car AP | Pedestrian AP |
|---|---|---|
| early fusion (RGB + depth as a 4th channel) | 0.310 | 0.059 |
| cross-attention (separate encoders, learned fusion) | 0.182 | 0.019 |

Zeroing the depth channel at inference collapses early fusion
(Car 0.310 -> 0.027) but barely moves cross-attention (0.182 -> 0.156): the
cross-attention model learned to ignore its LiDAR branch. Measuring the fusion
attention weights directly confirms it -- their normalised entropy is 0.9871,
i.e. effectively uniform.

## Getting the data

The dataset is not in this repository (~32 GB). Download from
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>
(registration required) and extract into the project root:

```
data_object_image_2.zip   ->  data_object_image_2/training/image_2/
data_object_velodyne.zip  ->  training/velodyne/
data_object_calib.zip     ->  data_object_calib/training/calib/
data_object_label_2.zip   ->  training/label_2/
```

The devkit (`devkit_object/`) is included because the train/validation split
depends on its `mapping/` files.

## Reproducing

1. `pip install numpy pandas matplotlib pillow torch torchvision pyarrow`
2. In `assignment.ipynb`, set `RUN_PRECOMPUTE = True` and run that cell once to
   build `processed/depth/` (~7 GB, ~15 min). Set it back to `False`.
3. Run the notebook. The trained checkpoints are committed, so the results and
   figures reproduce without retraining.

To retrain from scratch instead, set `RUN_TRAINING = True` (~45 min per model
on a 4 GB GPU). Both flags default to `False` so a plain "Run All" never
retrains or overwrites the committed checkpoints.
