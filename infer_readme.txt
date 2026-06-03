===========================================================
ProtoRes Inference (infer.py)
===========================================================

-----------------------------------------------------
0. OVERVIEW
-----------------------------------------------------
infer.py runs inference using a trained ProtoRes model.
Given a set of effector joint positions/rotation/lookat, it predicts the
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
- Hips, Spine0, Spine1, Chest, Neck, Head
  ClavicleLeft, BicepLeft, ForarmLeft, HandLeft
  ClavicleRight, BicepRight, ForarmRight, HandRight
  ThighLeft, CalfLeft, FootLeft, ToeLeft, ToeLeftEnd
  ThighRight, CalfRight, FootRight, ToeRight, ToeRightEnd
  (+ finger joints for both hands)


-----------------------------------------------------
2. INPUT FORMAT
-----------------------------------------------------
CSV file, where row is frame, and column is concatenated many effector information. (specified below)

Example Columns : BonePositions_Hips_X, BonePositions_Hips_Y, BonePositions_Hips_Z, ...
                    BonePositions_FootRight_X, BonePositions_FootRight_Y, BonePositions_FootRight_Z,
                    BoneRotations_Hips_X, BoneRotations_Hips_Y, BoneRotations_Hips_Z, BoneRotations_Hips_W,
                    LookAtTarget_Head_X, LookAtTarget_Head_Y, LookAtTarget_Head_Z,
                    LookAtDir_Head_X, LookAtDir_Head_Y, LookAtDir_Head_Z

Example File - datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_6PointsEffectors.csv

[Position effectors] (required)
  Shape  : (frame, N*3)
  Columns: BonePositions_{joint1}_X
           BonePositions_{joint1}_Y
           BonePositions_{joint1}_Z
                    ...
           BonePositions_{jointN}_X
           BonePositions_{jointN}_Y
           BonePositions_{jointN}_Z
           

[Rotation effectors] (optional)
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
3. OUTPUT FORMAT
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
 
[Position]
  Shape  : (frame, 64*3)
  Columns: BonePositions_{joint1}_X
           BonePositions_{joint1}_Y
           BonePositions_{joint1}_Z
                    ...
           BonePositions_{joint64}_X
           BonePositions_{joint64}_Y
           BonePositions_{joint64}_Z
 
[Rotation]
  Shape  : (frame, 64*4)  ← quaternion
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
4. USAGE
-----------------------------------------------------
python infer.py --input <input_csv> [options]

[Required]
  --input <path>               Input CSV path
                               Must contain BonePositions_{joint}_{X/Y/Z} columns

[Optional]
  --output <path>              Output CSV path (default: ./infer_output/output_pose.csv)
  --gt <path>                  Ground truth CSV path for evaluation (default: None)
  --joints <j1> <j2> ...       Position effector joints
                               (default: Hips Neck HandLeft HandRight FootLeft FootRight)
  --rotation_joints <j1> ...   Rotation effector joints
                               Input CSV must have BoneRotations_{joint}_{X/Y/Z/W} columns
  --lookat_joints <j1> ...     Lookat effector joints
                               Input CSV must have LookAtTarget_{joint}_{X/Y/Z} columns
                               LookAtDir_{joint}_{X/Y/Z} optional (default direction: (0,0,1))
  --position_tolerance <float>  Tolerance for position effectors (default: 0.0)
  --rotation_tolerance <float>  Tolerance for rotation effectors (default: 0.0)
  --lookat_tolerance <float>    Tolerance for lookat effectors (default: 0.0)



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

# 1. test set, 6 effectors (default joints)
python infer.py \
  --input  $DATA_DIR/minimixamo_test_6PointsEffectors.csv \
  --output ./infer_output/output_test_6points.csv

# 2. validation set, 6 effectors (default joints)
python infer.py \
  --input  $DATA_DIR/minimixamo_validation_6PointsEffectors.csv \
  --output ./infer_output/output_validation_6points.csv

# 3. test set, 5 effectors (custom joints)
python infer.py \
  --input  $DATA_DIR/minimixamo_test_5PointsEffectors.csv \
  --output ./infer_output/output_test_5points.csv \
  --joints Chest HandLeft HandRight FootLeft FootRight


# [With GT]

# 4. test set, 6 effectors + GT evaluation
python infer.py \
  --input  $DATA_DIR/minimixamo_test_6PointsEffectors.csv \
  --output ./infer_output/output_test_6points.csv \
  --gt     $DATA_DIR/minimixamo_test_fullPose.csv

# 5. validation set, 6 effectors + GT evaluation
python infer.py \
  --input  $DATA_DIR/minimixamo_validation_6PointsEffectors.csv \
  --output ./infer_output/output_validation_6points.csv \
  --gt     $DATA_DIR/minimixamo_validation_fullPose.csv

# 6. test set, 5 effectors (custom joints) + GT evaluation
python infer.py \
  --input  $DATA_DIR/minimixamo_test_5PointsEffectors.csv \
  --output ./infer_output/output_test_5points.csv \
  --gt     $DATA_DIR/minimixamo_test_fullPose.csv \
  --joints Chest HandLeft HandRight FootLeft FootRight


