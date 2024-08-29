import os
import numpy as np
from numba import njit
import open3d as o3d
import datetime
import scipy.interpolate as interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
import transform_utils as T
import yaml

# ===============================================
# = optimization utils
# ===============================================
def normalize_vars(vars, og_bounds):
    """
    Given 1D variables and bounds, normalize the variables to [-1, 1] range.
    """
    normalized_vars = np.empty_like(vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        normalized_vars[i] = (vars[i] - b_min) / (b_max - b_min) * 2 - 1
    return normalized_vars

def unnormalize_vars(normalized_vars, og_bounds):
    """
    Given 1D variables in [-1, 1] and original bounds, denormalize the variables to the original range.
    """
    vars = np.empty_like(normalized_vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        vars[i] = (normalized_vars[i] + 1) / 2 * (b_max - b_min) + b_min
    return vars

def calculate_collision_cost(poses, sdf_func, collision_points, threshold):
    assert poses.shape[1:] == (4, 4)
    transformed_pcs = batch_transform_points(collision_points, poses)
    transformed_pcs_flatten = transformed_pcs.reshape(-1, 3)  # [num_poses * num_points, 3]
    signed_distance = sdf_func(transformed_pcs_flatten) + threshold  # [num_poses * num_points]
    signed_distance = signed_distance.reshape(-1, collision_points.shape[0])  # [num_poses, num_points]
    non_zero_mask = signed_distance > 0
    collision_cost = np.sum(signed_distance[non_zero_mask])
    return collision_cost

@njit(cache=True, fastmath=True)
def consistency(poses_a, poses_b, rot_weight=0.5):
    assert poses_a.shape[1:] == (4, 4) and poses_b.shape[1:] == (4, 4), 'poses must be of shape (N, 4, 4)'
    min_distances = np.zeros(len(poses_a), dtype=np.float64)
    for i in range(len(poses_a)):
        min_distance = 9999999
        a = poses_a[i]
        for j in range(len(poses_b)):
            b = poses_b[j]
            pos_distance = np.linalg.norm(a[:3, 3] - b[:3, 3])
            rot_distance = angle_between_rotmat(a[:3, :3], b[:3, :3])
            distance = pos_distance + rot_distance * rot_weight
            min_distance = min(min_distance, distance)
        min_distances[i] = min_distance
    return np.mean(min_distances)

def transform_keypoints(transform, keypoints, movable_mask):
    assert transform.shape == (4, 4)
    transformed_keypoints = keypoints.copy()
    if movable_mask.sum() > 0:
        transformed_keypoints[movable_mask] = np.dot(keypoints[movable_mask], transform[:3, :3].T) + transform[:3, 3]
    return transformed_keypoints

@njit(cache=True, fastmath=True)
def batch_transform_points(points, transforms):
    """
    Apply multiple of transformation to point cloud, return results of individual transformations.
    Args:
        points: point cloud (N, 3).
        transforms: M 4x4 transformations (M, 4, 4).
    Returns:
        np.array: point clouds (M, N, 3).
    """
    assert transforms.shape[1:] == (4, 4), 'transforms must be of shape (M, 4, 4)'
    transformed_points = np.zeros((transforms.shape[0], points.shape[0], 3))
    for i in range(transforms.shape[0]):
        pos, R = transforms[i, :3, 3], transforms[i, :3, :3]
        transformed_points[i] = np.dot(points, R.T) + pos
    return transformed_points

@njit(cache=True, fastmath=True)
def get_samples_jitted(control_points_homo, control_points_quat, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size):
    assert control_points_homo.shape[1:] == (4, 4)
    # calculate number of samples per segment
    num_samples_per_segment = np.empty(len(control_points_homo) - 1, dtype=np.int64)
    for i in range(len(control_points_homo) - 1):
        start_pos = control_points_homo[i, :3, 3]
        start_rotmat = control_points_homo[i, :3, :3]
        end_pos = control_points_homo[i+1, :3, 3]
        end_rotmat = control_points_homo[i+1, :3, :3]
        pos_diff = np.linalg.norm(start_pos - end_pos)
        rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
        pos_num_steps = np.ceil(pos_diff / opt_interpolate_pos_step_size)
        rot_num_steps = np.ceil(rot_diff / opt_interpolate_rot_step_size)
        num_path_poses = int(max(pos_num_steps, rot_num_steps))
        num_path_poses = max(num_path_poses, 2)  # at least 2 poses, start and end
        num_samples_per_segment[i] = num_path_poses
    # fill in samples
    num_samples = num_samples_per_segment.sum()
    samples_7 = np.empty((num_samples, 7))
    sample_idx = 0
    for i in range(len(control_points_quat) - 1):
        start_pos, start_xyzw = control_points_quat[i, :3], control_points_quat[i, 3:]
        end_pos, end_xyzw = control_points_quat[i+1, :3], control_points_quat[i+1, 3:]
        # using proper quaternion slerp interpolation
        poses_7 = np.empty((num_samples_per_segment[i], 7))
        for j in range(num_samples_per_segment[i]):
            alpha = j / (num_samples_per_segment[i] - 1)
            pos = start_pos * (1 - alpha) + end_pos * alpha
            blended_xyzw = T.quat_slerp_jitted(start_xyzw, end_xyzw, alpha)
            pose_7 = np.empty(7)
            pose_7[:3] = pos
            pose_7[3:] = blended_xyzw
            poses_7[j] = pose_7
        samples_7[sample_idx:sample_idx+num_samples_per_segment[i]] = poses_7
        sample_idx += num_samples_per_segment[i]
    assert num_samples >= 2, f'num_samples: {num_samples}'
    return samples_7, num_samples

@njit(cache=True, fastmath=True)
def path_length(samples_homo):
    assert samples_homo.shape[1:] == (4, 4), 'samples_homo must be of shape (N, 4, 4)'
    pos_length = 0
    rot_length = 0
    for i in range(len(samples_homo) - 1):
        pos_length += np.linalg.norm(samples_homo[i, :3, 3] - samples_homo[i+1, :3, 3])
        rot_length += angle_between_rotmat(samples_homo[i, :3, :3], samples_homo[i+1, :3, :3])
    return pos_length, rot_length

# ===============================================
# = others
# ===============================================
def get_callable_grasping_cost_fn(env):
    def get_grasping_cost(keypoint_idx):
        keypoint_object = env.get_object_by_keypoint(keypoint_idx)
        return -env.is_grasping(candidate_obj=keypoint_object) + 1  # return 0 if grasping an object, 1 if not grasping any object
    return get_grasping_cost

def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, 'configs/config.yaml')
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'

def angle_between_quats(q1, q2):
    """Angle between two quaternions"""
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1, 1))

def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Filter points by taking only points within workspace bounds.
    """
    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask

def print_opt_debug_dict(debug_dict):
    print('\n' + '#' * 40)
    print(f'# Optimization debug info:')
    max_key_length = max(len(str(k)) for k in debug_dict.keys())
    for k, v in debug_dict.items():
        if isinstance(v, int) or isinstance(v, float):
            print(f'# {k:<{max_key_length}}: {v:.05f}')
        elif isinstance(v, list) and all(isinstance(x, int) or isinstance(x, float) for x in v):
            print(f'# {k:<{max_key_length}}: {np.array(v).round(5)}')
        else:
            print(f'# {k:<{max_key_length}}: {v}')
    print('#' * 40 + '\n')

def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    
def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e

def load_functions_from_txt(txt_path, get_grasping_cost_fn):
    if txt_path is None:
        return []
    # load txt file
    with open(txt_path, 'r') as f:
        functions_text = f.read()
    # execute functions
    gvars_dict = {
        'np': np,
        'get_grasping_cost_by_keypoint_idx': get_grasping_cost_fn,
    }  # external library APIs
    lvars_dict = dict()
    exec_safe(functions_text, gvars=gvars_dict, lvars=lvars_dict)
    return list(lvars_dict.values())

@njit(cache=True, fastmath=True)
def angle_between_rotmat(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    return np.arccos(cos_theta)

def fit_b_spline(control_points):
    # determine appropriate k
    k = min(3, control_points.shape[0]-1)
    spline = interpolate.splprep(control_points.T, s=0, k=k)
    return spline

def sample_from_spline(spline, num_samples):
    sample_points = np.linspace(0, 1, num_samples)
    if isinstance(spline, RotationSpline):
        samples = spline(sample_points).as_matrix()  # [num_samples, 3, 3]
    else:
        assert isinstance(spline, tuple) and len(spline) == 2, 'spline must be a tuple of (tck, u)'
        tck, u = spline
        samples = interpolate.splev(np.linspace(0, 1, num_samples), tck)  # [spline_dim, num_samples]
        samples = np.array(samples).T  # [num_samples, spline_dim]
    return samples

def linear_interpolate_poses(start_pose, end_pose, num_poses):
    """
    Interpolate between start and end pose.
    """
    assert num_poses >= 2, 'num_poses must be at least 2'
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    slerp = Slerp([0, 1], R.from_matrix([start_rotmat, end_rotmat]))
    poses = []
    for i in range(num_poses):
        alpha = i / (num_poses - 1)
        pos = start_pos * (1 - alpha) + end_pos * alpha
        rotmat = slerp(alpha).as_matrix()
        if start_pose.shape == (6,):
            euler = T.mat2euler(rotmat)
            poses.append(np.concatenate([pos, euler]))
        elif start_pose.shape == (4, 4):
            pose = np.eye(4)
            pose[:3, :3] = rotmat
            pose[:3, 3] = pos
            poses.append(pose)
        elif start_pose.shape == (7,):
            quat = T.mat2quat(rotmat)
            pose = np.concatenate([pos, quat])
            poses.append(pose)
    return np.array(poses)

def spline_interpolate_poses(control_points, num_steps):
    """
    Interpolate between through the control points using spline interpolation.
    1. Fit a b-spline through the positional terms of the control points.
    2. Fit a RotationSpline through the rotational terms of the control points.
    3. Sample the b-spline and RotationSpline at num_steps.

    Args:
        control_points: [N, 6] position + euler or [N, 4, 4] pose or [N, 7] position + quat
        num_steps: number of poses to interpolate
    Returns:
        poses: [num_steps, 6] position + euler or [num_steps, 4, 4] pose or [num_steps, 7] position + quat
    """
    assert num_steps >= 2, 'num_steps must be at least 2'
    if isinstance(control_points, list):
        control_points = np.array(control_points)
    if control_points.shape[1] == 6:
        control_points_pos = control_points[:, :3]  # [N, 3]
        control_points_euler = control_points[:, 3:]  # [N, 3]
        control_points_rotmat = []
        for control_point_euler in control_points_euler:
            control_points_rotmat.append(T.euler2mat(control_point_euler))
        control_points_rotmat = np.array(control_points_rotmat)  # [N, 3, 3]
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        control_points_pos = control_points[:, :3, 3]  # [N, 3]
        control_points_rotmat = control_points[:, :3, :3]  # [N, 3, 3]
    elif control_points.shape[1] == 7:
        control_points_pos = control_points[:, :3]
        control_points_rotmat = []
        for control_point_quat in control_points[:, 3:]:
            control_points_rotmat.append(T.quat2mat(control_point_quat))
        control_points_rotmat = np.array(control_points_rotmat)
    else:
        raise ValueError('control_points not recognized')
    # remove the duplicate points (threshold 1e-3)
    diff = np.linalg.norm(np.diff(control_points_pos, axis=0), axis=1)
    mask = diff > 1e-3
    # always keep the first and last points
    mask = np.concatenate([[True], mask[:-1], [True]])
    control_points_pos = control_points_pos[mask]
    control_points_rotmat = control_points_rotmat[mask]
    # fit b-spline through positional terms control points
    pos_spline = fit_b_spline(control_points_pos)
    # fit RotationSpline through rotational terms control points
    times = pos_spline[1]
    rotations = R.from_matrix(control_points_rotmat)
    rot_spline = RotationSpline(times, rotations)
    # sample from the splines
    pos_samples = sample_from_spline(pos_spline, num_steps)  # [num_steps, 3]
    rot_samples = sample_from_spline(rot_spline, num_steps)  # [num_steps, 3, 3]
    if control_points.shape[1] == 6:
        poses = []
        for i in range(num_steps):
            pose = np.concatenate([pos_samples[i], T.mat2euler(rot_samples[i])])
            poses.append(pose)
        poses = np.array(poses)
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        poses = np.empty((num_steps, 4, 4))
        poses[:, :3, :3] = rot_samples
        poses[:, :3, 3] = pos_samples
        poses[:, 3, 3] = 1
    elif control_points.shape[1] == 7:
        poses = np.empty((num_steps, 7))
        for i in range(num_steps):
            quat = T.mat2quat(rot_samples[i])
            pose = np.concatenate([pos_samples[i], quat])
            poses[i] = pose
    return poses

def get_linear_interpolation_steps(start_pose, end_pose, pos_step_size, rot_step_size):
    """
    Given start and end pose, calculate the number of steps to interpolate between them.
    Args:
        start_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        end_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        pos_step_size: position step size
        rot_step_size: rotation step size
    Returns:
        num_path_poses: number of poses to interpolate
    """
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    pos_diff = np.linalg.norm(start_pos - end_pos)
    rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
    pos_num_steps = np.ceil(pos_diff / pos_step_size)
    rot_num_steps = np.ceil(rot_diff / rot_step_size)
    num_path_poses = int(max(pos_num_steps, rot_num_steps))
    num_path_poses = max(num_path_poses, 2)  # at least start and end poses
    return num_path_poses

def farthest_point_sampling(pc, num_points):
    """
    Given a point cloud, sample num_points points that are the farthest apart.
    Use o3d farthest point sampling.
    """
    assert pc.ndim == 2 and pc.shape[1] == 3, "pc must be a (N, 3) numpy array"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    downpcd_farthest = pcd.farthest_point_down_sample(num_points)
    return np.asarray(downpcd_farthest.points)