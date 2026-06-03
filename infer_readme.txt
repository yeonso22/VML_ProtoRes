===========================================================
ProtoRes Inference (infer.py)
===========================================================

-----------------------------------------------------
0. OVERVIEW
-----------------------------------------------------
infer.py runs inference using a trained ProtoRes model.
Given a set of effector joint positions/rotation, it predicts the
full body pose (positions + rotations) for all 64 joints.


-----------------------------------------------------
1. SKELETON
-----------------------------------------------------
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


-----------------------------------------------------
2. INPUT FORMAT
-----------------------------------------------------
  2-1. Input 1 - CSV 
-----------------------------------------------------
CSV file, where row is frame, and column is concatenated many effector information. (specified below)

Example Columns : BonePositions_Hips_X, BonePositions_Hips_Y, BonePositions_Hips_Z, ...
                    BonePositions_FootRight_X, BonePositions_FootRight_Y, BonePositions_FootRight_Z,
                    BoneRotations_Hips_X, BoneRotations_Hips_Y, BoneRotations_Hips_Z, BoneRotations_Hips_W,

Example File - datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_6PointsEffectors.csv

[Global Position] (required)
  Shape  : (frame, N*3)
  Columns: BonePositions_{joint1}_X
           BonePositions_{joint1}_Y
           BonePositions_{joint1}_Z
                    ...
           BonePositions_{jointN}_X
           BonePositions_{jointN}_Y
           BonePositions_{jointN}_Z
           

[Parent-Relative Rotation] (optional)
  Shape  : (frame, N*4)  
  Columns: BoneRotations_{joint1}_X   (quaternion x)
           BoneRotations_{joint1}_Y   (quaternion y)
           BoneRotations_{joint1}_Z   (quaternion z)
           BoneRotations_{joint1}_W   (quaternion w)
                    ...
           BoneRotations_{jointN}_X   (quaternion x)
           BoneRotations_{jointN}_Y   (quaternion y)
           BoneRotations_{jointN}_Z   (quaternion z)
           BoneRotations_{jointN}_W   (quaternion w)

-----------------------------------------------------
  2-2. Input 2 - Numpy  
-----------------------------------------------------
Numpy file, where the size is [Batch Size, Number of Every Joints(64), 4, 4]

[Orientation & Position]
  - [ :, :, 0:3, 0:3] correspond to orientation of each joint.
  - [ :, :, 0:3, 3] corresponds to position of each joint.

[Joints]
  - [ :, 0, :, : ] corresponds to joint "Hips". Its position is world position, and its orientation is world orientation.
  - [ :, 1:64, :, : ] corresponds to joints other than "Hips". Its position is parent-relative position, and its orientation is parent-relative orientation. 

Among the 64 joints, you can select which joints to use as effector inputs to ProtoRes.
To choose which joints are used as inputs, modify the 'EFFECTOR_JOINTS_NUMPY' list in infer.py. 



-----------------------------------------------------
3. OUTPUT FORMAT
-----------------------------------------------------
  3-1. OUTPUT 1 - CSV
-----------------------------------------------------
CSV file, where row is frame, and column is listed all 64 joints' position and rotation information.

Example Columns : BonePositions_Hips_X, BonePositions_Hips_Y, BonePositions_Hips_Z,
                    BonePositions_Spine0_X, BonePositions_Spine0_Y, BonePositions_Spine0_Z,
                                                  ...
                    BonePositions_ToeRightEnd_X, BonePoswitions_ToeRightEnd_Y, BonePositions_ToeRightEnd_Z,
                    BoneRotations_Hips_X, BoneRotations_Hips_Y, BoneRotations_Hips_Z, BoneRotations_Hips_W,
                                                  ...
                    BoneRotations_ToeRightEnd_X, BoneRotations_ToeRightEnd_Y, BoneRotations_ToeRightEnd_Z, BoneRotations_ToeRightEnd_W

Example File - datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_fullPose.csv
 
[Global Position]
  Shape  : (frame, 64*3)
  Columns: BonePositions_{joint1}_X
           BonePositions_{joint1}_Y
           BonePositions_{joint1}_Z
                    ...
           BonePositions_{joint64}_X
           BonePositions_{joint64}_Y
           BonePositions_{joint64}_Z
 
[Parent-Relative Rotation]
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

-----------------------------------------------------
  3-2. OUTPUT 2 - Numpy
-----------------------------------------------------
Numpy file, where the size is [Batch Size, Number of Every Joints(64), 4, 4]

[Orientation & Position]
  - [ :, :, 0:3, 0:3] correspond to orientation of each joint.
  - [ :, :, 0:3, 3] corresponds to position of each joint.

[Joints]
  - [ :, 0, :, : ] corresponds to joint "Hips". Its position is world position, and its orientation is world orientation.
  - [ :, 1:64, :, : ] corresponds to joints other than "Hips". Its position is parent-relative position, and its orientation is parent-relative orientation.  



-----------------------------------------------------
4. USAGE
-----------------------------------------------------
python infer.py --input <input_csv> [options]

[Required]
  --input <path>               Input CSV path
                               Must contain BonePositions_{joint}_{X/Y/Z} columns

[Optional]
  --output <path>              Output CSV path (default: ./infer_output/output_pose.csv)
  --gt <path>                  Ground truth CSV path for evaluation (default: None)
  --position_tolerance <float>  Tolerance for position effectors (default: 0.0)
  --rotation_tolerance <float>  Tolerance for rotation effectors (default: 0.0)



-----------------------------------------------------
5. EVALUATION METRICS (when --gt is provided)
-----------------------------------------------------
  [Position]
  - MSE        : Mean Squared Error of joint positions
  - Mean L2    : Mean L2 distance of joint positions
  - Per-joint  : Top 5 joints with largest L2 error

  [Rotation]
  - Geodesic   : Mean geodesic distance between predicted and GT rotations (radians)
  - Per-joint  : Top 5 joints with largest geodesic error


-----------------------------------------------------
6. EXAMPLES
-----------------------------------------------------
# Run in the directory : Labs/Projects/ProtoRes/code/
DATA_DIR=./datasets/deeppose_paper2021_minimixamo/clipgroup

# [Without GT]
# 1. python infer.py \
  --input  $DATA_DIR/minimixamo_test_6PointsEffectors.csv \
  --output ./infer_output/output_test_6points.csv

# 2. python infer.py \
  --input  $DATA_DIR/minimixamo_validation_6PointsEffectors.csv \
  --output ./infer_output/output_validation_6points.csv


# [With GT]
# 3. python infer.py \
  --input  $DATA_DIR/minimixamo_test_6PointsEffectors.csv \
  --output ./infer_output/output_test_6points.csv \
  --gt     $DATA_DIR/minimixamo_test_fullPose.csv

# 4. python infer.py \
  --input  $DATA_DIR/minimixamo_validation_6PointsEffectors.csv \
  --output ./infer_output/output_validation_6points.csv \
  --gt     $DATA_DIR/minimixamo_validation_fullPose.csv



