import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import json
import argparse
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


sys.path.insert(0, '.')

DATASET_SETTINGS = './datasets/deeppose_paper2021_minimixamo/dataset_settings.json'
MODEL_FOLDER     = './logs'
CHECKPOINT_NAME  = 'epoch=1999.ckpt'
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
EFFECTOR_JOINTS  = ['Hips', 'Neck', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight']


def setup_model(model_folder, checkpoint_name, skeleton, device="cuda:0"):
    """
    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_utils.py
          setup_model()
    hparams.yaml로 모델 구조 생성 후 checkpoint로 weight 주입
    """
    from protores.utils.model_factory import ModelFactory
    import protores.models.optional_lookat_model
    checkpoint_path = os.path.join(model_folder, 'checkpoints', checkpoint_name)
    hparams_path    = os.path.join(model_folder, 'hparams.yaml')
    cfg = OmegaConf.load(hparams_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)
    model = ModelFactory.instantiate(cfg, skeleton=skeleton)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    return model


def save_predictions(skeleton, predictions, out_path):
    """
    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_utils.py
          save_predictions()
    6D rotation → rotation matrix → quaternion 변환 후 CSV 저장
    """
    from protores.geometry.rotations import compute_rotation_matrix_from_ortho6d
    from protores.geometry.quaternions import compute_quaternions_from_rotation_matrices

    # 모델 output에서 rotation(6D)과 position 가져오기
    # rotation: (batch, nb_joints, 6) - 모델은 6D rotation으로 출력
    # position: (batch, nb_joints, 3) - GPU 텐서를 numpy로 변환
    predicted_joint_rotations = predictions["joint_rotations"]
    predicted_joint_positions = predictions["joint_positions"].cpu().detach().numpy()

    # 6D rotation → 3x3 rotation matrix → quaternion(xyzw) 변환
    # CSV 저장 형식이 quaternion이라 변환 필요
    predicted_joint_rotations = compute_rotation_matrix_from_ortho6d(predicted_joint_rotations.view(-1, 6))
    predicted_joint_rotations = compute_quaternions_from_rotation_matrices(predicted_joint_rotations)
    # (batch * nb_joints, 4) → (batch, nb_joints, 4) 로 reshape 후 numpy 변환
    predicted_joint_rotations = predicted_joint_rotations.view(-1, skeleton.nb_joints, 4).cpu().detach().numpy()

    # skeleton에서 joint 이름 순서대로 가져오기
    # index_bones: {0: 'Hips', 1: 'Spine0', ...} 형태
    all_joints = [skeleton.index_bones[i] for i in range(skeleton.nb_joints)]

    # 컬럼을 dict로 먼저 모아서 한 번에 DataFrame 생성 (PerformanceWarning 방지)
    data = {}

    # position 컬럼 추가: BonePositions_{joint}_{X/Y/Z}
    for joint in all_joints:
        idx = skeleton.bone_indexes[joint]  # joint 이름 → index 번호
        data[f"BonePositions_{joint}_X"] = predicted_joint_positions[:, idx, 0]
        data[f"BonePositions_{joint}_Y"] = predicted_joint_positions[:, idx, 1]
        data[f"BonePositions_{joint}_Z"] = predicted_joint_positions[:, idx, 2]

    # rotation 컬럼 추가: BoneRotations_{joint}_{X/Y/Z/W}
    # quaternion 순서: protores 내부는 wxyz, CSV 저장은 xyzw
    # 그래서 idx 1,2,3,0 순서로 저장 (w가 마지막)
    for joint in all_joints:
        idx = skeleton.bone_indexes[joint]
        data[f"BoneRotations_{joint}_X"] = predicted_joint_rotations[:, idx, 1]  # x
        data[f"BoneRotations_{joint}_Y"] = predicted_joint_rotations[:, idx, 2]  # y
        data[f"BoneRotations_{joint}_Z"] = predicted_joint_rotations[:, idx, 3]  # z
        data[f"BoneRotations_{joint}_W"] = predicted_joint_rotations[:, idx, 0]  # w

    # DataFrame 생성 후 Frame을 index로 설정해서 CSV 저장
    predicted_df = pd.DataFrame(data)
    predicted_df['Frame'] = list(range(len(predicted_df)))
    predicted_df.set_index('Frame', inplace=True)
    predicted_df.to_csv(out_path, index=True)
    return out_path


if __name__ == "__main__":

    # ============================================================
    # 0. 인자 파싱
    # ============================================================

    # 인자 없이 실행하면 usage 안내 출력
    if len(sys.argv) == 1:
        print("""
Usage: python infer.py --input <input_csv> [options]

[Required]
  --input <path>             Input CSV path (must contain BonePositions_{joint}_{X/Y/Z} columns)

[Optional]
  --output <path>            Output CSV path (default: ./output_pose.csv)
  --gt <path>                Ground truth CSV path for evaluation (default: None)
  --joints <j1> <j2> ...     Position effector joints
                             (default: Hips Neck HandLeft HandRight FootLeft FootRight)
  --rotation_joints <j1> ... Rotation effector joints
                             (input CSV must have BoneRotations_{joint}_{X/Y/Z/W} columns)
  --lookat_joints <j1> ...   Lookat effector joints
                             (input CSV must have LookAtTarget_{joint}_{X/Y/Z} columns,
                              LookAtDir_{joint}_{X/Y/Z} optional, default direction: (0,0,1))
  --position_tolerance <float>  Tolerance for position effectors (default: 0.0)
  --rotation_tolerance <float>  Tolerance for rotation effectors (default: 0.0)
  --lookat_tolerance <float>    Tolerance for lookat effectors (default: 0.0)

Examples (run from Labs/Projects/ProtoRes/code/):
  python infer.py --input ./datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_6PointsEffectors.csv
  python infer.py --input ./datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_6PointsEffectors.csv --output ./infer_output/result.csv --gt ./datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_fullPose.csv
  python infer.py --input ./datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_5PointsEffectors.csv --output ./infer_output/result.csv --joints Chest HandLeft HandRight FootLeft FootRight
  python infer.py --input ./datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_5PointsEffectors.csv --output ./infer_output/result.csv --joints Chest HandLeft HandRight FootLeft FootRight --gt ./datasets/deeppose_paper2021_minimixamo/clipgroup/minimixamo_test_fullPose.csv

Available joints (64):
  Hips, Spine0, Spine1, Chest, Neck, Head,
  ClavicleLeft, BicepLeft, ForarmLeft, HandLeft,
  ClavicleRight, BicepRight, ForarmRight, HandRight,
  ThighLeft, CalfLeft, FootLeft, ToeLeft,
  ThighRight, CalfRight, FootRight, ToeRight,
  (+ finger joints)
""")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="ProtoRes Inference")
    parser.add_argument('--input',           type=str, required=True,
                        help='input CSV path (effector positions)')
    parser.add_argument('--output',          type=str, default='./infer_output/output_pose.csv',
                        help='output CSV path (default: ./output_pose.csv)')
    parser.add_argument('--gt',              type=str, default=None,
                        help='ground truth CSV path (optional, for evaluation)')
    parser.add_argument('--joints',          type=str, nargs='+', default=EFFECTOR_JOINTS,
                        help='position effector joint names (default: 6 joints)')
    parser.add_argument('--rotation_joints', type=str, nargs='+', default=None,
                        help='rotation effector joint names (requires BoneRotations columns in input CSV)')
    parser.add_argument('--lookat_joints',   type=str, nargs='+', default=None,
                        help='lookat effector joint names (requires LookAtTarget columns in input CSV)')
    parser.add_argument('--position_tolerance', type=float, default=0.0,
                    help='tolerance for position effectors (default: 0.0)')
    parser.add_argument('--rotation_tolerance', type=float, default=0.0,
                        help='tolerance for rotation effectors (default: 0.0)')
    parser.add_argument('--lookat_tolerance',   type=float, default=0.0,
                        help='tolerance for lookat effectors (default: 0.0)')
    args = parser.parse_args()

    # ============================================================
    # 1. Skeleton 로드
    #    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_utils.py
    #          build_dataset()
    #      → skeleton = Skeleton(config['skeleton'])
    # ============================================================
    from protores.geometry.skeleton import Skeleton
    with open(DATASET_SETTINGS, 'r') as f:
        config = json.load(f)
    skeleton = Skeleton(config['skeleton'])
    print(f"Skeleton loaded: {skeleton.nb_joints} joints")

    # ============================================================
    # 2. 모델 로드
    #    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_utils.py
    #          setup_model()
    # ============================================================
    model = setup_model(MODEL_FOLDER, CHECKPOINT_NAME, skeleton, DEVICE)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {CHECKPOINT_NAME}")

    # ============================================================
    # 3. Input 준비
    #    CSV 컬럼 형식:
    #      position → BonePositions_{joint}_{X/Y/Z}
    #      rotation → BoneRotations_{joint}_{X/Y/Z/W} (quaternion)
    #      lookat   → LookAtTarget_{joint}_{X/Y/Z}
    #                 LookAtDir_{joint}_{X/Y/Z} (없으면 기본값 (0,0,1) 사용)
    #    각 행 = 1 frame
    # ============================================================
    from protores.geometry.rotations import (compute_rotation_matrix_from_quaternion,
                                              compute_ortho6d_from_rotation_matrix)

    df = pd.read_csv(args.input)
    batch_size    = len(df)
    num_effectors = len(args.joints)
    print(f"Input loaded: {batch_size} frames from {args.input}")

    # ---------- 컬럼 검증 ----------
    missing_cols = []

    # position 컬럼 확인
    for j in args.joints:
        for axis in ["X", "Y", "Z"]:
            col = f"BonePositions_{j}_{axis}"
            if col not in df.columns:
                missing_cols.append(col)

    # rotation 컬럼 확인
    if args.rotation_joints:
        for j in args.rotation_joints:
            for axis in ["X", "Y", "Z", "W"]:
                col = f"BoneRotations_{j}_{axis}"
                if col not in df.columns:
                    missing_cols.append(col)

    # lookat 컬럼 확인
    if args.lookat_joints:
        for j in args.lookat_joints:
            for axis in ["X", "Y", "Z"]:
                col = f"LookAtTarget_{j}_{axis}"
                if col not in df.columns:
                    missing_cols.append(col)

    if missing_cols:
        print("\n[ERROR] The following columns are missing in the input CSV:")
        for col in missing_cols:
            print(f"  - {col}")
        print("\nAvailable columns in the input CSV:")
        for col in df.columns.tolist():
            print(f"  - {col}")
        sys.exit(1)

    # ---------- position effectors ----------
    # 참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_utils.py
    #       fixed_points_effector_batch() with_bone_rotation=False 브랜치
    position_data = np.stack([
        df[[f"BonePositions_{j}_X", f"BonePositions_{j}_Y", f"BonePositions_{j}_Z"]].values
        for j in args.joints
    ], axis=1).astype(np.float32)  # (batch_size, num_effectors, 3)

    position_ids       = [skeleton.bone_indexes[name] for name in args.joints]
    position_id        = torch.tensor(position_ids, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
    position_weight    = torch.ones((batch_size, num_effectors), dtype=torch.float32)
    position_tolerance = torch.full((batch_size, num_effectors), args.position_tolerance, dtype=torch.float32)

    # ---------- rotation effectors ----------
    # 참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_utils.py
    #       fixed_points_effector_batch() with_bone_rotation=True 브랜치
    #       quaternion(xyzw) → rotation matrix → 6D rotation 변환
    if args.rotation_joints:
        num_rot = len(args.rotation_joints)
        rot_quat = np.stack([
            df[[f"BoneRotations_{j}_X", f"BoneRotations_{j}_Y",
                f"BoneRotations_{j}_Z", f"BoneRotations_{j}_W"]].values
            for j in args.rotation_joints
        ], axis=1).astype(np.float32)  # (batch_size, num_rot, 4)

        # quaternion xyzw → wxyz (protores 내부 포맷)
        rot_quat_wxyz = np.concatenate([rot_quat[..., 3:4], rot_quat[..., :3]], axis=-1)
        rot_tensor = torch.tensor(rot_quat_wxyz).view(-1, 4)
        rot_mat    = compute_rotation_matrix_from_quaternion(rot_tensor)
        rot_6d     = compute_ortho6d_from_rotation_matrix(rot_mat)
        rot_6d     = rot_6d.view(batch_size, num_rot, 6)

        rotation_ids       = [skeleton.bone_indexes[name] for name in args.rotation_joints]
        rotation_id        = torch.tensor(rotation_ids, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
        rotation_weight    = torch.ones((batch_size, num_rot), dtype=torch.float32)
        rotation_tolerance = torch.full((batch_size, num_rot), args.rotation_tolerance, dtype=torch.float32)
    else:
        rot_6d             = torch.zeros((batch_size, 0, 6), dtype=torch.float32)
        rotation_id        = torch.zeros((batch_size, 0), dtype=torch.int64)
        rotation_weight    = torch.zeros((batch_size, 0), dtype=torch.float32)
        rotation_tolerance = torch.zeros((batch_size, 0), dtype=torch.float32)

    # ---------- lookat effectors ----------
    # 참고: Labs/Projects/ProtoRes/code/protores/models/optional_lookat_model.py
    #       get_data_from_batch() lookat effector 섹션
    #       lookat_effectors_in = torch.cat([lookat_positions, local_lookat_directions], dim=2)
    if args.lookat_joints:
        num_lookat = len(args.lookat_joints)
        lookat_targets = np.stack([
            df[[f"LookAtTarget_{j}_X", f"LookAtTarget_{j}_Y", f"LookAtTarget_{j}_Z"]].values
            for j in args.lookat_joints
        ], axis=1).astype(np.float32)  # (batch_size, num_lookat, 3)

        # local direction: CSV에 있으면 읽고, 없으면 기본값 (0, 0, 1) 사용
        if f"LookAtDir_{args.lookat_joints[0]}_X" in df.columns:
            lookat_dirs = np.stack([
                df[[f"LookAtDir_{j}_X", f"LookAtDir_{j}_Y", f"LookAtDir_{j}_Z"]].values
                for j in args.lookat_joints
            ], axis=1).astype(np.float32)
        else:
            lookat_dirs = np.zeros((batch_size, num_lookat, 3), dtype=np.float32)
            lookat_dirs[:, :, 2] = 1.0
            print("LookAtDir column not found → using default direction (0, 0, 1)")

        lookat_data = np.concatenate([lookat_targets, lookat_dirs], axis=-1)  # (batch_size, num_lookat, 6)

        lookat_ids       = [skeleton.bone_indexes[name] for name in args.lookat_joints]
        lookat_id        = torch.tensor(lookat_ids, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
        lookat_weight    = torch.ones((batch_size, num_lookat), dtype=torch.float32)
        lookat_tolerance = torch.full((batch_size, num_lookat), args.lookat_tolerance, dtype=torch.float32)
        lookat_data      = torch.tensor(lookat_data)
    else:
        lookat_data      = torch.zeros((batch_size, 0, 6), dtype=torch.float32)
        lookat_id        = torch.zeros((batch_size, 0), dtype=torch.int64)
        lookat_weight    = torch.zeros((batch_size, 0), dtype=torch.float32)
        lookat_tolerance = torch.zeros((batch_size, 0), dtype=torch.float32)

    # input_data dict 구조
    # 참고: Labs/Projects/ProtoRes/code/protores/models/optional_lookat_model.py
    #       get_dummy_input()
    input_data = {
        "position_data":      torch.tensor(position_data).to(DEVICE),
        "position_weight":    position_weight.to(DEVICE),
        "position_tolerance": position_tolerance.to(DEVICE),
        "position_id":        position_id.to(DEVICE),
        "rotation_data":      rot_6d.to(DEVICE),
        "rotation_weight":    rotation_weight.to(DEVICE),
        "rotation_tolerance": rotation_tolerance.to(DEVICE),
        "rotation_id":        rotation_id.to(DEVICE),
        "lookat_data":        lookat_data.to(DEVICE),
        "lookat_weight":      lookat_weight.to(DEVICE),
        "lookat_tolerance":   lookat_tolerance.to(DEVICE),
        "lookat_id":          lookat_id.to(DEVICE),
    }

    # ============================================================
    # 4. Inference
    #    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_model.py
    #          fixed_points_benchmark()
    #      → predictions = model(data)
    # ============================================================
    with torch.no_grad():
        predictions = model(input_data)

    print(f"\n=== Output ===")
    print(f"joint_positions : {predictions['joint_positions'].shape}")
    print(f"joint_rotations : {predictions['joint_rotations'].shape}")
    print(f"root_joint_pos  : {predictions['root_joint_position'].shape}")

    # ============================================================
    # 5. 결과 저장
    #    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_model.py
    #          fixed_points_benchmark()
    #      → save_predictions(validation_set.skeleton, predictions, out_file)
    # ============================================================
    save_predictions(skeleton, predictions, args.output)
    print(f"\nSaved to {args.output}")

    # ============================================================
    # 6. GT와 비교 (옵션, --gt 인자가 있을 때만 실행)
    #    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_utils.py
    #          get_confidence_interval_margins()
    #          update_test_metrics() in optional_lookat_model.py
    #    실제 test에서는 GT가 없으므로 생략 가능
    # ============================================================
    if args.gt is not None:
        from protores.geometry.rotations import (compute_rotation_matrix_from_ortho6d,
                                                  compute_rotation_matrix_from_quaternion,
                                                  compute_geodesic_distance_from_two_matrices)

        all_joints = [skeleton.index_bones[i] for i in range(skeleton.nb_joints)]
        gt_df = pd.read_csv(args.gt)

        # ---------- position 비교 ----------
        # GT position 로드: (batch_size, nb_joints, 3)
        gt_positions = np.stack([
            gt_df[[f"BonePositions_{j}_X", f"BonePositions_{j}_Y", f"BonePositions_{j}_Z"]].values
            for j in all_joints
        ], axis=1).astype(np.float32)

        pred_positions = predictions["joint_positions"].cpu().detach().numpy()

        # MSE: 전체 joint position의 평균 제곱 오차
        mse = np.mean((pred_positions - gt_positions) ** 2)

        # L2: joint별 예측-GT 거리 (axis=2는 xyz 방향)
        # l2_per_joint shape: (batch_size, nb_joints)
        l2_per_joint      = np.linalg.norm(pred_positions - gt_positions, axis=2)
        mean_l2           = np.mean(l2_per_joint)
        mean_l2_per_joint = np.mean(l2_per_joint, axis=0)  # frame 방향 평균 → (nb_joints,)

        # ---------- rotation 비교 ----------
        # 참고: eval_utils.py의 get_confidence_interval_margins()
        #       geodesic distance로 rotation 오차 측정
        #       (두 rotation matrix 사이의 각도 차이, 단위: radian)

        # GT rotation: quaternion(xyzw) → wxyz 변환 → rotation matrix
        gt_quat = np.stack([
            gt_df[[f"BoneRotations_{j}_X", f"BoneRotations_{j}_Y",
                   f"BoneRotations_{j}_Z", f"BoneRotations_{j}_W"]].values
            for j in all_joints
        ], axis=1).astype(np.float32)  # (batch_size, nb_joints, 4) xyzw

        # xyzw → wxyz (protores 내부 포맷)
        gt_quat_wxyz = np.concatenate([gt_quat[..., 3:4], gt_quat[..., :3]], axis=-1)
        gt_rot_mat = compute_rotation_matrix_from_quaternion(
            torch.tensor(gt_quat_wxyz).view(-1, 4)
        ).view(-1, skeleton.nb_joints, 3, 3).to(DEVICE)

        # predicted rotation: 6D rotation → rotation matrix
        pred_rot_mat = compute_rotation_matrix_from_ortho6d(
            predictions["joint_rotations"].view(-1, 6)
        ).view(-1, skeleton.nb_joints, 3, 3)

        # geodesic distance: 두 rotation matrix 사이의 각도 차이 (radian)
        # shape: (batch_size * nb_joints,) → (batch_size, nb_joints)로 reshape
        geodesic = compute_geodesic_distance_from_two_matrices(
            pred_rot_mat.view(-1, 3, 3),
            gt_rot_mat.view(-1, 3, 3)
        ).view(-1, skeleton.nb_joints)

        mean_geodesic           = geodesic.mean().item()
        mean_geodesic_per_joint = geodesic.mean(dim=0).cpu().numpy()  # (nb_joints,)

        # ---------- 결과 출력 ----------
        print(f"\n=== GT Evaluation ===")

        print(f"\n[Position]")
        print(f"MSE            : {mse:.6f}")
        print(f"Mean L2        : {mean_l2:.6f}")
        print(f"\nTop 5 joints with largest L2 error:")
        top5_pos = np.argsort(mean_l2_per_joint)[::-1][:5]
        for idx in top5_pos:
            print(f"  {all_joints[idx]:20s}: {mean_l2_per_joint[idx]:.6f}")

        print(f"\n[Rotation]")
        print(f"Mean Geodesic  : {mean_geodesic:.6f} rad")
        print(f"\nTop 5 joints with largest Geodesic error:")
        top5_rot = np.argsort(mean_geodesic_per_joint)[::-1][:5]
        for idx in top5_rot:
            print(f"  {all_joints[idx]:20s}: {mean_geodesic_per_joint[idx]:.6f} rad")