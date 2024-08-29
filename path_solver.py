import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.interpolate import RegularGridInterpolator
import copy
import time
import transform_utils as T
from utils import (
    farthest_point_sampling,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
    normalize_vars,
    unnormalize_vars,
    get_samples_jitted,
    calculate_collision_cost,
    path_length,
    transform_keypoints,
)

# ====================================
# = objective function
# ====================================
def objective(opt_vars,
                og_bounds,
                start_pose,
                end_pose,
                keypoints_centered,
                keypoint_movable_mask,
                path_constraints,
                sdf_func,
                collision_points_centered,
                opt_interpolate_pos_step_size,
                opt_interpolate_rot_step_size,
                ik_solver,
                initial_joint_pos,
                reset_joint_pos,
                return_debug_dict=False):

    debug_dict = {}
    debug_dict['num_control_points'] = len(opt_vars) // 6

    # unnormalize variables and do conversion
    unnormalized_opt_vars = unnormalize_vars(opt_vars, og_bounds)
    control_points_euler = np.concatenate([start_pose[None], unnormalized_opt_vars.reshape(-1, 6), end_pose[None]], axis=0)  # [num_control_points, 6]
    control_points_homo = T.convert_pose_euler2mat(control_points_euler)  # [num_control_points, 4, 4]
    control_points_quat = T.convert_pose_mat2quat(control_points_homo)  # [num_control_points, 7]
    # get dense samples
    poses_quat, num_poses = get_samples_jitted(control_points_homo, control_points_quat, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size)
    poses_homo = T.convert_pose_quat2mat(poses_quat)
    debug_dict['num_poses'] = num_poses
    start_idx, end_idx = 1, num_poses - 1  # exclude start and goal

    cost = 0
    # collision cost
    if collision_points_centered is not None:
        collision_cost = 0.5 * calculate_collision_cost(poses_homo[start_idx:end_idx], sdf_func, collision_points_centered, 0.20)
        debug_dict['collision_cost'] = collision_cost
        cost += collision_cost

    # penalize path length
    pos_length, rot_length = path_length(poses_homo)
    approx_length = pos_length + rot_length * 1.0
    path_length_cost = 4.0 * approx_length
    debug_dict['path_length_cost'] = path_length_cost
    cost += path_length_cost

    # reachability cost
    ik_cost = 0
    reset_reg_cost = 0
    debug_dict['ik_pos_error'] = []
    debug_dict['ik_feasible'] = []
    max_iterations = 20
    for control_point_homo in control_points_homo:
        ik_result = ik_solver.solve(
                        control_point_homo,
                        max_iterations=max_iterations,
                        initial_joint_pos=initial_joint_pos,
                    )
        debug_dict['ik_pos_error'].append(ik_result.position_error)
        debug_dict['ik_feasible'].append(ik_result.success)
        ik_cost += 20.0 * (ik_result.num_descents / max_iterations)
        if ik_result.success:
            reset_reg = np.linalg.norm(ik_result.cspace_position[:-1] - reset_joint_pos[:-1])
            reset_reg = np.clip(reset_reg, 0.0, 3.0)
        else:
            reset_reg = 3.0
        reset_reg_cost += 0.2 * reset_reg
    debug_dict['ik_pos_error'] = np.array(debug_dict['ik_pos_error'])
    debug_dict['ik_feasible'] = np.array(debug_dict['ik_feasible'])
    debug_dict['ik_cost'] = ik_cost
    debug_dict['reset_reg_cost'] = reset_reg_cost
    cost += ik_cost

    # # path constraint violation cost
    debug_dict['path_violation'] = None
    if path_constraints is not None and len(path_constraints) > 0:
        path_constraint_cost = 0
        path_violation = []
        for pose in poses_homo[start_idx:end_idx]:
            transformed_keypoints = transform_keypoints(pose, keypoints_centered, keypoint_movable_mask)
            for constraint in path_constraints:
                violation = constraint(transformed_keypoints[0], transformed_keypoints[1:])
                path_violation.append(violation)
                path_constraint_cost += np.clip(violation, 0, np.inf)
        path_constraint_cost = 200.0*path_constraint_cost
        debug_dict['path_constraint_cost'] = path_constraint_cost
        debug_dict['path_violation'] = path_violation
        cost += path_constraint_cost

    debug_dict['total_cost'] = cost

    if return_debug_dict:
        return cost, debug_dict

    return cost


class PathSolver:
    """
    Given a goal pose and a start pose, solve for a sequence of intermediate poses for the end effector to follow.
    
    Optimization variables:
    - sequence of intermediate control points
    """

    def __init__(self, config, ik_solver, reset_joint_pos):
        self.config = config
        self.ik_solver = ik_solver
        self.reset_joint_pos = reset_joint_pos
        self.last_opt_result = None
        # warmup
        self._warmup()

    def _warmup(self):
        start_pose = np.array([0.0, 0.0, 0.3, 0, 0, 0, 1])
        end_pose = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1])
        keypoints = np.random.rand(10, 3)
        keypoint_movable_mask = np.random.rand(10) > 0.5
        path_constraints = []
        sdf_voxels = np.zeros((10, 10, 10))
        collision_points = np.random.rand(100, 3)
        self.solve(start_pose, end_pose, keypoints, keypoint_movable_mask, path_constraints, sdf_voxels, collision_points, None, from_scratch=True)
        self.last_opt_result = None

    def _setup_sdf(self, sdf_voxels):
        # create callable sdf function with interpolation
        x = np.linspace(self.config['bounds_min'][0], self.config['bounds_max'][0], sdf_voxels.shape[0])
        y = np.linspace(self.config['bounds_min'][1], self.config['bounds_max'][1], sdf_voxels.shape[1])
        z = np.linspace(self.config['bounds_min'][2], self.config['bounds_max'][2], sdf_voxels.shape[2])
        sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
        return sdf_func

    def _check_opt_result(self, opt_result, path_quat, debug_dict, og_bounds):
        # accept the opt_result if it's only terminated due to iteration limit
        if (not opt_result.success and ('maximum' in opt_result.message.lower() or 'iteration' in opt_result.message.lower() or 'not necessarily' in opt_result.message.lower())):
            opt_result.success = True
        elif not opt_result.success:
            opt_result.message += '; invalid solution'
        # check whether path constraints are satisfied
        if debug_dict['path_violation'] is not None:
            path_violation = np.array(debug_dict['path_violation'])
            opt_result.message += f'; path_violation: {path_violation}'
            path_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in path_violation])
            if not path_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; path constraint not satisfied'
        return opt_result

    def _center_collision_points_and_keypoints(self, ee_pose, collision_points, keypoints, keypoint_movable_mask):
        ee_pose_homo = T.pose2mat([ee_pose[:3], T.euler2quat(ee_pose[3:])])
        centering_transform = np.linalg.inv(ee_pose_homo)
        collision_points_centered = np.dot(collision_points, centering_transform[:3, :3].T) + centering_transform[:3, 3]
        keypoints_centered = transform_keypoints(centering_transform, keypoints, keypoint_movable_mask)
        return collision_points_centered, keypoints_centered

    def solve(self,
            start_pose,
            end_pose,
            keypoints,
            keypoint_movable_mask,
            path_constraints,
            sdf_voxels,
            collision_points,
            initial_joint_pos,
            from_scratch=False):
        """
        Args:
            - start_pose (np.ndarray): [7], [x, y, z, qx, qy, qz, qw]
            - end_pose (np.ndarray): [7], [x, y, z, qx, qy, qz, qw]
            - keypoints (np.ndarray): [num_keypoints, 3]
            - keypoint_movable_mask (bool): whether the keypoints are on the object being grasped
            - path_constraints (List[Callable]): path constraints
            - sdf_voxels (np.ndarray): [H, W, D]
            - collision_points (np.ndarray): [num_points, 3], point cloud of the object being grasped
            - initial_joint_pos (np.ndarray): [N] initial joint positions of the robot.
            - from_scratch (bool): whether to start from scratch

        Returns:
            - opt_result (scipy.optimize.OptimizeResult): optimization opt_result
            - debug_dict (dict): debug information
        """
        # downsample collision points
        if collision_points is not None and collision_points.shape[0] > self.config['max_collision_points']:
            collision_points = farthest_point_sampling(collision_points, self.config['max_collision_points'])
        sdf_func = self._setup_sdf(sdf_voxels)

        # ====================================
        # = setup bounds
        # ====================================
        # calculate an appropriate number of control points, including start and goal
        num_control_points = get_linear_interpolation_steps(start_pose, end_pose, self.config['opt_pos_step_size'], self.config['opt_rot_step_size'])
        num_control_points = np.clip(num_control_points, 3, 6)
        # transform to euler representation
        start_pose = np.concatenate([start_pose[:3], T.quat2euler(start_pose[3:])])
        end_pose = np.concatenate([end_pose[:3], T.quat2euler(end_pose[3:])])

        # bounds for decision variables
        og_bounds = [(b_min, b_max) for b_min, b_max in zip(self.config['bounds_min'], self.config['bounds_max'])] + \
                    [(-np.pi, np.pi) for _ in range(3)]
        og_bounds *= (num_control_points - 2)
        og_bounds = np.array(og_bounds, dtype=np.float64)
        bounds = [(-1, 1)] * len(og_bounds)
        num_vars = len(bounds)

        # ====================================
        # = setup initial guess
        # ====================================
        # use previous opt_result as initial guess if available
        if not from_scratch and self.last_opt_result is not None:
            init_sol = self.last_opt_result.x
            # if there are more control points in this iter, fill the rest with the last value + small noise
            if len(init_sol) < num_vars:
                new_x0 = np.empty(num_vars)
                new_x0[:len(init_sol)] = init_sol
                for i in range(len(init_sol), num_vars, 6):
                    new_x0[i:i+6] = init_sol[-6:] + np.random.randn(6) * 0.01
                init_sol = new_x0
            # otherwise, use the last num_vars values
            else:
                init_sol = init_sol[-num_vars:]
        # initial guess as linear interpolation
        else:
            from_scratch = True
            interp_poses = linear_interpolate_poses(start_pose, end_pose, num_control_points)  # [num_control_points, 6]
            init_sol = interp_poses[1:-1].flatten()  # [num_control_points-2, 6]
            init_sol = normalize_vars(init_sol, og_bounds)

        # clip the initial guess to be within bounds
        for i, (b_min, b_max) in enumerate(bounds):
            init_sol[i] = np.clip(init_sol[i], b_min, b_max)

        # ====================================
        # = other setup
        # ====================================
        collision_points_centered, keypoints_centered = self._center_collision_points_and_keypoints(start_pose, collision_points, keypoints, keypoint_movable_mask)
        aux_args = (og_bounds,
                    start_pose,
                    end_pose,
                    keypoints_centered,
                    keypoint_movable_mask,
                    path_constraints,
                    sdf_func,
                    collision_points_centered,
                    self.config['opt_interpolate_pos_step_size'],
                    self.config['opt_interpolate_rot_step_size'],
                    self.ik_solver,
                    initial_joint_pos,
                    self.reset_joint_pos)

        # ====================================
        # = solve optimization
        # ====================================
        start = time.time()
        # use global optimization for the first iteration
        if from_scratch:
            opt_result = dual_annealing(
                func=objective,
                bounds=bounds,
                args=aux_args,
                maxfun=self.config['sampling_maxfun'],
                x0=init_sol,
                no_local_search=True,
                minimizer_kwargs={
                    'method': 'SLSQP',
                    'options': self.config['minimizer_options'],
                },
            )
        # use gradient-based local optimization for the following iterations
        else:
            opt_result = minimize(
                fun=objective,
                x0=init_sol,
                args=aux_args,
                bounds=bounds,
                method='SLSQP',
                options=self.config['minimizer_options'],
            )
        solve_time = time.time() - start

        # ====================================
        # = post-process opt_result
        # ====================================
        if isinstance(opt_result.message, list):
            opt_result.message = opt_result.message[0]
        # rerun to get debug info
        _, debug_dict = objective(opt_result.x, *aux_args, return_debug_dict=True)
        debug_dict['sol'] = opt_result.x.reshape(-1, 6)
        debug_dict['msg'] = opt_result.message
        debug_dict['solve_time'] = solve_time
        debug_dict['from_scratch'] = from_scratch
        debug_dict['type'] = 'path_solver'
        # unnormailze
        sol = unnormalize_vars(opt_result.x, og_bounds)
        # add end pose
        poses_euler = np.concatenate([sol.reshape(-1, 6), end_pose[None]], axis=0)
        poses_quat = T.convert_pose_euler2quat(poses_euler)  # [num_control_points, 7]
        opt_result = self._check_opt_result(opt_result, poses_quat, debug_dict, og_bounds)
        # cache opt_result for future use if successful
        if opt_result.success:
            self.last_opt_result = copy.deepcopy(opt_result)
        return poses_quat, debug_dict