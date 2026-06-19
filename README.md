# VML ProtoRes Inference

This repository includes the code needed to run ProtoRes inference with `infer.py`. (repository link : https://github.com/yeonso22/VML_ProtoRes)

Given partial joint effector data, the script predicts a full 64-joint pose.


## Included Files

```text
VML_ProtoRes/
  infer.py
  README.md
  requirements.txt
  LICENSE.md
  protores/
```

Large files are not included in git:

```text
protores/logs/
```

## Required External Files

Before running inference, place the missing checkpoint files like this:

```text
  protores/logs/
    hparams.yaml
    checkpoints/
      epoch=1999.ckpt
```

Default paths are set in `infer.py`:


## Environment

Python 3.8 is recommended.

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run from the repository root.

Example 1 : 

```bash
python infer.py \
  --input ./datasets/deeppose_paper2021_minimixamo/clipgroup/input_ex_minimixamo_test_6PointsEffectors.csv --output ./infer_output/output_test_6points.csv
```

Example 2 : 

```bash
python infer.py \
  --input ./datasets/deeppose_paper2021_minimixamo/clipgroup/input_ex.npy --output ./infer_output/output_pose.csv
```

The script saves both:

```text
infer_output/~~.csv
infer_output/~~.npy
```


## Input Format

### Input 1: CSV

CSV file, where row is batch, and column is concatenated many effector information. (specified below)

Example Columns:


#### Global Position (required)

```text
Shape  : (batch, N*3)
Columns: BonePositions_{joint1}_X
         BonePositions_{joint1}_Y
         BonePositions_{joint1}_Z
         ...
         BonePositions_{jointN}_X
         BonePositions_{jointN}_Y
         BonePositions_{jointN}_Z
```
           

#### Global Rotation (optional)

```text
Shape  : (batch, N*4)  
Columns: BoneRotations_{joint1}_X   (quaternion x)
         BoneRotations_{joint1}_Y   (quaternion y)
         BoneRotations_{joint1}_Z   (quaternion z)
         BoneRotations_{joint1}_W   (quaternion w)
         ...
         BoneRotations_{jointN}_X   (quaternion x)
         BoneRotations_{jointN}_Y   (quaternion y)
         BoneRotations_{jointN}_Z   (quaternion z)
         BoneRotations_{jointN}_W   (quaternion w)
```

### Input 2: Numpy

Numpy file, where the size is [Batch Size, Number of Every Joints(64), 4, 4]

#### Orientation & Position

- [ :, :, 0:3, 0:3] correspond to orientation of each joint.
- [ :, :, 0:3, 3] corresponds to position of each joint.

#### Joints

- [ :, 0, :, : ] corresponds to joint "Hips". Its position is world position, and its orientation is world orientation.
- [ :, 1:64, :, : ] corresponds to joints other than "Hips". Its position is parent-relative position, and its orientation is parent-relative orientation. 

Among the 64 joints, you can select which joints to use as effector inputs to ProtoRes.
To choose which joints are used as inputs, modify the 'EFFECTOR_JOINTS_NUMPY' list in infer.py. 



## Output Format

### Output 1: CSV

CSV file, where row is batch, and column is listed all 64 joints' position and rotation information.

#### Global Position

```text
Shape  : (frame, 64*3)
Columns: BonePositions_{joint1}_X
         BonePositions_{joint1}_Y
         BonePositions_{joint1}_Z
         ...
         BonePositions_{joint64}_X
         BonePositions_{joint64}_Y
         BonePositions_{joint64}_Z
```
 
#### Relative Rotation

```text
Shape  : (frame, 64*4) 
Columns: BoneRotations_{joint1}_X   (quaternion x)
         BoneRotations_{joint1}_Y   (quaternion y)
         BoneRotations_{joint1}_Z   (quaternion z)
         BoneRotations_{joint1}_W   (quaternion w)
         ...
         BoneRotations_{joint64}_X   (quaternion x)
         BoneRotations_{joint64}_Y   (quaternion y)
         BoneRotations_{joint64}_Z   (quaternion z)
         BoneRotations_{joint64}_W   (quaternion w)
```

### Output 2: Numpy

Numpy file, where the size is [Batch Size, Number of Every Joints(64), 4, 4]

#### Orientation & Position

- [ :, :, 0:3, 0:3] correspond to orientation of each joint.
- [ :, :, 0:3, 3] corresponds to position of each joint.

#### Joints

- [ :, 0, :, : ] corresponds to joint "Hips". Its position is world position, and its orientation is world orientation.
- [ :, 1:64, :, : ] corresponds to joints other than "Hips". Its position is parent-relative position, and its orientation is parent-relative orientation.  


## Skeleton

- Total joints: 64
- Each joint has:
    - Position : 1 x Vector3 (x, y, z)
    - Rotation : 1 x Quaternion (x, y, z, w)
- Defined in:
  datasets/deeppose_paper2021_minimixamo/dataset_settings.json
- Hips, Spine0, Spine1, Chest, Neck, Head, ClavicleLeft, ClavicleRight, BicepLeft, ForearmLeft, HandLeft,
  Index0Left, Index1Left, Index2Left, Index2LeftEnd, Middle0Left, Middle1Left, Middle2Left, Middle2LeftEnd,
  Ring0Left, Ring1Left, Ring2Left, Ring2LeftEnd, Pinky0Left, Pinky1Left, Pinky2Left, Pinky2LeftEnd, Thumb0Left,
  Thumb1Left, Thumb2Left, Thumb2LeftEnd, BicepRight, ForarmRight, HandRight, Index0Right, Index1Right,
  Index2Right, Index2RightEnd, Middle0Right, Middle1Right, Middle2Right, Middle2RightEnd, Ring0Right, Ring1Right,
  Ring2Right, Ring2RightEnd, Pinky0Right, Pinky1Right, Pinky2Right, Pinky2RightEnd, Thumb0Right, Thumb1Right,
  Thumb2Right, Thumb2RightEnd, ThighLeft, CalfLeft, FootLeft, ToeLeft, ToeLeftEnd, ThighRight, CalfRight,
  FootRight, ToeRight, ToeRightEnd


## Citation

```bibtex
@inproceedings{oreshkin2022protores:,
  title={ProtoRes: Proto-Residual Network for Pose Authoring via Learned Inverse Kinematics},
  author={Boris N. Oreshkin and Florent Bocquelet and F{\'{e}}lix G. Harvey and Bay Raitt and Dominic Laflamme},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
