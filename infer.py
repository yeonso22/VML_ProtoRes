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
from protores.geometry.quaternions import compute_quaternions_from_rotation_matrices
from protores.geometry.rotations import (compute_rotation_matrix_from_ortho6d,
                                                  compute_rotation_matrix_from_quaternion,
                                                  compute_ortho6d_from_rotation_matrix)
from protores.geometry.skeleton import Skeleton

######## YOU CAN SELECT THE INPUT JOINTS HERE ########
## (1) All List of Character's joints
ALL_JOINTS = ['Hips', 'Spine0', 'Spine1', 'Chest', 'Neck', 'Head', 'ClavicleLeft', 'ClavicleRight', 'BicepLeft', 'ForearmLeft', 'HandLeft',
  'Index0Left', 'Index1Left', 'Index2Left', 'Index2LeftEnd', 'Middle0Left', 'Middle1Left', 'Middle2Left', 'Middle2LeftEnd',
  'Ring0Left', 'Ring1Left', 'Ring2Left', 'Ring2LeftEnd', 'Pinky0Left', 'Pinky1Left', 'Pinky2Left', 'Pinky2LeftEnd', 'Thumb0Left',
  'Thumb1Left', 'Thumb2Left', 'Thumb2LeftEnd', 'BicepRight', 'ForearmRight', 'HandRight', 'Index0Right', 'Index1Right',
  'Index2Right', 'Index2RightEnd', 'Middle0Right', 'Middle1Right', 'Middle2Right', 'Middle2RightEnd', 'Ring0Right', 'Ring1Right',
  'Ring2Right', 'Ring2RightEnd', 'Pinky0Right', 'Pinky1Right', 'Pinky2Right', 'Pinky2RightEnd', 'Thumb0Right', 'Thumb1Right',
  'Thumb2Right', 'Thumb2RightEnd', 'ThighLeft', 'CalfLeft', 'FootLeft', 'ToeLeft', 'ToeLeftEnd', 'ThighRight', 'CalfRight',
  'FootRight', 'ToeRight', 'ToeRightEnd']

## (2) Selected Character's joints for Protores Input (You can modify this list to select different joints.)
EFFECTOR_JOINTS_NUMPY = ['Hips', 'Neck', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight']
##############################################



sys.path.insert(0, '.')

DATASET_SETTINGS = './datasets/deeppose_paper2021_minimixamo/dataset_settings.json'
MODEL_FOLDER     = './logs'
CHECKPOINT_NAME  = 'epoch=1999.ckpt'
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'


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

def prepare_input_from_csv(df, skeleton, args, device):
    batch_size    = len(df)

    # (1) CSV에서 joint 종류 파싱
    position_joints = [
        col.replace("BonePositions_", "").replace("_X", "")
        for col in df.columns if col.startswith("BonePositions_") and col.endswith("_X")
    ]
    rotation_joints = [
        col.replace("BoneRotations_", "").replace("_X", "")
        for col in df.columns if col.startswith("BoneRotations_") and col.endswith("_X")
    ]

    # (2) position 데이터 준비하기
    num_position_effectors = len(position_joints)
    position_data = np.stack([
        df[[f"BonePositions_{j}_X", f"BonePositions_{j}_Y", f"BonePositions_{j}_Z"]].values
        for j in position_joints
    ], axis=1).astype(np.float32)

    position_ids       = [skeleton.bone_indexes[name] for name in position_joints]
    position_id        = torch.tensor(position_ids, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1) ## (batch_size, num_effectors)
    position_weight    = torch.ones((batch_size, num_position_effectors), dtype=torch.float32) ## (batch_size, num_effectors)
    position_tolerance = torch.full((batch_size, num_position_effectors), args.position_tolerance, dtype=torch.float32) ## 아직

    # (3) rotation 데이터 준비하기
    num_rotation_effectors = len(rotation_joints)
    if rotation_joints:
        num_rot = len(rotation_joints)
        rotation_data_xyzw = np.stack([
            df[[f"BoneRotations_{j}_X", f"BoneRotations_{j}_Y",
                f"BoneRotations_{j}_Z", f"BoneRotations_{j}_W"]].values
            for j in rotation_joints
        ], axis=1).astype(np.float32)  

        # quaternion xyzw -> wxyz
        rotation_data_wxyz = np.concatenate([rotation_data_xyzw[..., 3:4], rotation_data_xyzw[..., :3]], axis=-1)
        rotation_data_wxyz_tensor = torch.tensor(rotation_data_wxyz).view(-1, 4)

        # quaternion wxyz -> rotation matrix
        rotation_data_mat    = compute_rotation_matrix_from_quaternion(rotation_data_wxyz_tensor)

        # quaternion wxyz -> 6D rotation
        rotation_data_6d     = compute_ortho6d_from_rotation_matrix(rotation_data_mat)
        rotation_data     = rotation_data_6d.view(batch_size, num_rotation_effectors, 6)

        rotation_ids       = [skeleton.bone_indexes[name] for name in rotation_joints]
        rotation_id        = torch.tensor(rotation_ids, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
        rotation_weight    = torch.ones((batch_size, num_rotation_effectors), dtype=torch.float32)
        rotation_tolerance = torch.full((batch_size, num_rotation_effectors), args.rotation_tolerance, dtype=torch.float32)
    else:
        rotation_data      = torch.zeros((batch_size, 0, 6), dtype=torch.float32)
        rotation_id        = torch.zeros((batch_size, 0), dtype=torch.int64)
        rotation_weight    = torch.zeros((batch_size, 0), dtype=torch.float32)
        rotation_tolerance = torch.zeros((batch_size, 0), dtype=torch.float32)


    lookat_data      = torch.zeros((batch_size, 0, 6), dtype=torch.float32)
    lookat_id        = torch.zeros((batch_size, 0), dtype=torch.int64)
    lookat_weight    = torch.zeros((batch_size, 0), dtype=torch.float32)
    lookat_tolerance = torch.zeros((batch_size, 0), dtype=torch.float32)

    # (5) 마지막으로 input_data dict로 묶기
    input_data = {
        "position_data":      torch.tensor(position_data).to(device),
        "position_weight":    position_weight.to(device),
        "position_tolerance": position_tolerance.to(device),
        "position_id":        position_id.to(device),
        "rotation_data":      rotation_data.to(device),
        "rotation_weight":    rotation_weight.to(device),
        "rotation_tolerance": rotation_tolerance.to(device),
        "rotation_id":        rotation_id.to(device),
        "lookat_data":        lookat_data.to(device),
        "lookat_weight":      lookat_weight.to(device),
        "lookat_tolerance":   lookat_tolerance.to(device),
        "lookat_id":          lookat_id.to(device),
    }
    return input_data, batch_size

def prepare_input_from_numpy(npy_path, skeleton, args, device):
    # (1) 로드하기
    input_numpy = np.load(npy_path) 
    batch_size = input_numpy.shape[0]
    effector_joints = EFFECTOR_JOINTS_NUMPY
    num_effector_joints = len(effector_joints)

    # (2) position 데이터 준비 (Hip을 제외하고는 현재 local 상태. global로 바꿔야 함)
    # (2-1) 우선 root의 global position
    hip_position = torch.tensor(input_numpy[:, skeleton.bone_indexes['Hips'], 0:3, 3], dtype=torch.float32) ## (batch_size, 3)

    # (2-2) local rotation 준비
    joints_local_rotations = torch.tensor(input_numpy[:, :,0:3, 0:3], dtype=torch.float32)

    # (2-3) fk를 통한 모든 joints의 global position
    position_data_all, _ = skeleton.forward(joints_local_rotations, hip_position)
    position_data = position_data_all[:, [skeleton.bone_indexes[name] for name in effector_joints], :]

    position_ids       = [skeleton.bone_indexes[name] for name in effector_joints]
    position_id        = torch.tensor(position_ids, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1) ## (batch_size, num_effectors)
    position_weight    = torch.ones((batch_size, num_effector_joints), dtype=torch.float32) ## (batch_size, num_effectors)
    position_tolerance = torch.full((batch_size, num_effector_joints), args.position_tolerance, dtype=torch.float32) ## 아직


    # (3) rotation 데이터 준비
    rotation_data_mat = np.stack([
        input_numpy[:, skeleton.bone_indexes[j], 0:3, 0:3] for j in effector_joints], axis=1).astype(np.float32) ## (batch_size, num_effector_joints, 3, 3)

    rotation_data = compute_ortho6d_from_rotation_matrix(torch.tensor(rotation_data_mat)).reshape(batch_size, num_effector_joints, 6)
    rotation_ids       = [skeleton.bone_indexes[name] for name in effector_joints]
    rotation_id        = torch.tensor(rotation_ids, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
    rotation_weight    = torch.ones((batch_size, num_effector_joints), dtype=torch.float32)
    rotation_tolerance = torch.full((batch_size, num_effector_joints), args.rotation_tolerance, dtype=torch.float32)

    # (4) lookat 없음
    lookat_data      = torch.zeros((batch_size, 0, 6), dtype=torch.float32)
    lookat_id        = torch.zeros((batch_size, 0), dtype=torch.int64)
    lookat_weight    = torch.zeros((batch_size, 0), dtype=torch.float32)
    lookat_tolerance = torch.zeros((batch_size, 0), dtype=torch.float32)

    # (5) 마지막으로 input_data dict로 묶기
    input_data = {
        "position_data":      position_data.to(device),
        "position_weight":    position_weight.to(device),
        "position_tolerance": position_tolerance.to(device),
        "position_id":        position_id.to(device),
        "rotation_data":      rotation_data.to(device),
        "rotation_weight":    rotation_weight.to(device),
        "rotation_tolerance": rotation_tolerance.to(device),
        "rotation_id":        rotation_id.to(device),
        "lookat_data":        lookat_data.to(device),
        "lookat_weight":      lookat_weight.to(device),
        "lookat_tolerance":   lookat_tolerance.to(device),
        "lookat_id":          lookat_id.to(device),
    }
    return input_data, batch_size


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
  --position_tolerance <float>  Tolerance for position effectors (default: 0.0)
  --rotation_tolerance <float>  Tolerance for rotation effectors (default: 0.0)
""")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="ProtoRes Inference")
    parser.add_argument('--input',           type=str, required=True,
                        help='input path')
    parser.add_argument('--output',          type=str, default='./infer_output/output_pose.csv',
                        help='output CSV path (default: ./output_pose.csv)')
    parser.add_argument('--position_tolerance', type=float, default=0.0,
                    help='tolerance for position effectors (default: 0.0)')
    parser.add_argument('--rotation_tolerance', type=float, default=0.0,
                        help='tolerance for rotation effectors (default: 0.0)')
    # parser.add_argument('--lookat_tolerance',   type=float, default=0.0,
    #                     help='tolerance for lookat effectors (default: 0.0)')
    args = parser.parse_args()

    # ============================================================
    # 1. Skeleton 로드
    # ============================================================
    with open(DATASET_SETTINGS, 'r') as f:
        config = json.load(f)
    skeleton = Skeleton(config['skeleton'])

    # ============================================================
    # 2. 모델 로드
    #    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_utils.py
    #          setup_model()
    # ============================================================
    model = setup_model(MODEL_FOLDER, CHECKPOINT_NAME, skeleton, DEVICE)
    model = model.to(DEVICE)
    model.eval()

    # ============================================================
    # 3. Input 준비
    # ============================================================
    if args.input.endswith('.npy'):
        input_data, batch_size = prepare_input_from_numpy(args.input, skeleton, args, DEVICE)
    else:
        df = pd.read_csv(args.input)
        input_data, batch_size = prepare_input_from_csv(df, skeleton, args, DEVICE)

    # ============================================================
    # 4. Inference
    #    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_model.py
    #          fixed_points_benchmark()
    #      → predictions = model(data)
    # ============================================================
    with torch.no_grad():
        predictions = model(input_data)

    # ============================================================
    # 5-1. CSV 결과 저장
    #    참고: Labs/Projects/ProtoRes/code/protores/evaluation/eval_model.py
    #          fixed_points_benchmark()
    # ============================================================
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_predictions(skeleton, predictions, args.output)

    # ============================================================
    # 5-2. Numpy 결과 저장
    # ============================================================
    np_output_path = args.output.replace('.csv', '.npy')   

    output_numpy = np.zeros((batch_size, skeleton.nb_joints, 4, 4))

    # (1) rotation matrix 저장
    pred = predictions["joint_rotations"]
    pred_rot_mat = compute_rotation_matrix_from_ortho6d(
            pred.reshape(-1, 6)
        ) 
    output_numpy[:, :, :3, :3] = pred_rot_mat.view(batch_size, skeleton.nb_joints, 3, 3).cpu().detach().numpy()

    output_numpy[:, :, 3, 3] = 1.0 ## 아마도 하는 게 맞겠지?

    # (2) position 저장
    output_numpy[:, 0, 0:3, 3] = predictions["joint_positions"][:, 0, :].cpu().detach().numpy()

    with open(DATASET_SETTINGS, 'r') as f:
        config = json.load(f)

    for joints in config['skeleton']['joints']:
        idx = joints['index']
        offset = joints['offset']
        output_numpy[:, idx, 0, 3] = offset['x']
        output_numpy[:, idx, 1, 3] = offset['y']
        output_numpy[:, idx, 2, 3] = offset['z']

    ## (3) npy 저장
    print(f"Shape of {np_output_path}: {output_numpy.shape}")
    np.save(np_output_path, output_numpy)
