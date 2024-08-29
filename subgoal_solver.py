import numpy as np
import time
import copy
from scipy.optimize import dual_annealing, minimize
from scipy.interpolate import RegularGridInterpolator
import transform_utils as T
from utils import (
    transform_keypoints,
    calculate_collision_cost,
    normalize_vars,
    unnormalize_vars,
    farthest_point_sampling,
    consistency,
)
def objective(opt_vars,
            og_bounds,
            keypoints_centered,
            keypoint_movable_mask,
            goal_constraints,
            path_constraints,
            sdf_func,
            collision_points_centered,
            init_pose_homo,
            ik_solver,
            initial_joint_pos,
            reset_joint_pos,
            is_grasp_stage,
            return_debug_dict=False):

    debug_dict = {}
    # unnormalize variables and do conversion
    opt_pose = unnormalize_vars(opt_vars, og_bounds)
    opt_pose_homo = T.pose2mat([opt_pose[:3], T.euler2quat(opt_pose[3:])])

    cost = 0
    # collision cost
    if collision_points_centered is not None:
        collision_cost = 0.8 * calculate_collision_cost(opt_pose_homo[None], sdf_func, collision_points_centered, 0.10)
        debug_dict['collision_cost'] = collision_cost
        cost += collision_cost

    # stay close to initial pose
    init_pose_cost = 1.0 * consistency(opt_pose_homo[None], init_pose_homo[None], rot_weight=1.5)
    debug_dict['init_pose_cost'] = init_pose_cost
    cost += init_pose_cost

    # reachability cost (approximated by number of IK iterations + regularization from reset joint pos)
    max_iterations = 20
    ik_result = ik_solver.solve(
                    opt_pose_homo,
                    max_iterations=max_iterations,
                    initial_joint_pos=initial_joint_pos,
                )
    ik_cost = 20.0 * (ik_result.num_descents / max_iterations)
    debug_dict['ik_feasible'] = ik_result.success
    debug_dict['ik_pos_error'] = ik_result.position_error
    debug_dict['ik_cost'] = ik_cost
    cost += ik_cost
    if ik_result.success:
        reset_reg = np.linalg.norm(ik_result.cspace_position[:-1] - reset_joint_pos[:-1])
        reset_reg = np.clip(reset_reg, 0.0, 3.0)
    else:
        reset_reg = 3.0
    reset_reg_cost = 0.2 * reset_reg
    debug_dict['reset_reg_cost'] = reset_reg_cost
    cost += reset_reg_cost

    # grasp metric (better performance if using anygrasp or force-based grasp metrics)
    if is_grasp_stage:
        preferred_dir = np.array([0, 0, -1]) 
        grasp_cost = -np.dot(opt_pose_homo[:3, 0], preferred_dir) + 1  # [0, 1]
        grasp_cost = 10.0 * grasp_cost
        debug_dict['grasp_cost'] = grasp_cost
        cost += grasp_cost

    # goal constraint violation cost
    debug_dict['subgoal_constraint_cost'] = None
    debug_dict['subgoal_violation'] = None
    if goal_constraints is not None and len(goal_constraints) > 0:
        subgoal_constraint_cost = 0
        transformed_keypoints = transform_keypoints(opt_pose_homo, keypoints_centered, keypoint_movable_mask)
        subgoal_violation = []
        for constraint in goal_constraints:
            violation = constraint(transformed_keypoints[0], transformed_keypoints[1:])
            subgoal_violation.append(violation)
            subgoal_constraint_cost += np.clip(violation, 0, np.inf)
        subgoal_constraint_cost = 200.0*subgoal_constraint_cost
        debug_dict['subgoal_constraint_cost'] = subgoal_constraint_cost
        debug_dict['subgoal_violation'] = subgoal_violation
        cost += subgoal_constraint_cost
    
    # path constraint violation cost
    debug_dict['path_violation'] = None
    if path_constraints is not None and len(path_constraints) > 0:
        path_constraint_cost = 0
        transformed_keypoints = transform_keypoints(opt_pose_homo, keypoints_centered, keypoint_movable_mask)
        path_violation = []
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


class SubgoalSolver:
    def __init__(self, config, ik_solver, reset_joint_pos):
        self.config = config
        self.ik_solver = ik_solver
        self.reset_joint_pos = reset_joint_pos
        self.last_opt_result = None
        # warmup
        self._warmup()

    def _warmup(self):
        ee_pose = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1])
        keypoints = np.random.rand(10, 3)
        keypoint_movable_mask = np.random.rand(10) > 0.5
        goal_constraints = []
        path_constraints = []
        sdf_voxels = np.zeros((10, 10, 10))
        collision_points = np.random.rand(100, 3)
        self.solve(ee_pose, keypoints, keypoint_movable_mask, goal_constraints, path_constraints, sdf_voxels, collision_points, True, None, from_scratch=True)
        self.last_opt_result = None

    def _setup_sdf(self, sdf_voxels):
        # create callable sdf function with interpolation
        x = np.linspace(self.config['bounds_min'][0], self.config['bounds_max'][0], sdf_voxels.shape[0])
        y = np.linspace(self.config['bounds_min'][1], self.config['bounds_max'][1], sdf_voxels.shape[1])
        z = np.linspace(self.config['bounds_min'][2], self.config['bounds_max'][2], sdf_voxels.shape[2])
        sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
        return sdf_func

    def _check_opt_result(self, opt_result, debug_dict):
        # accept the opt_result if it's only terminated due to iteration limit
        if (not opt_result.success and ('maximum' in opt_result.message.lower() or 'iteration' in opt_result.message.lower() or 'not necessarily' in opt_result.message.lower())):
            opt_result.success = True
        elif not opt_result.success:
            opt_result.message += '; invalid solution'
        # check whether goal constraints are satisfied
        if debug_dict['subgoal_violation'] is not None:
            goal_constraints_results = np.array(debug_dict['subgoal_violation'])
            opt_result.message += f'; goal_constraints_results: {goal_constraints_results} (higher is worse)'
            goal_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in goal_constraints_results])
            if not goal_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; goal not satisfied'
        # check whether path constraints are satisfied
        if debug_dict['path_violation'] is not None:
            path_constraints_results = np.array(debug_dict['path_violation'])
            opt_result.message += f'; path_constraints_results: {path_constraints_results}'
            path_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in path_constraints_results])
            if not path_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; path not satisfied'
        # check whether ik is feasible
        if 'ik_feasible' in debug_dict and not debug_dict['ik_feasible']:
            opt_result.success = False
            opt_result.message += f'; ik not feasible'
        return opt_result
    
    def _center_collision_points_and_keypoints(self, ee_pose_homo, collision_points, keypoints, keypoint_movable_mask):
        centering_transform = np.linalg.inv(ee_pose_homo)
        collision_points_centered = np.dot(collision_points, centering_transform[:3, :3].T) + centering_transform[:3, 3]
        keypoints_centered = transform_keypoints(centering_transform, keypoints, keypoint_movable_mask)
        return collision_points_centered, keypoints_centered

    def solve(self,
            ee_pose,
            keypoints,
            keypoint_movable_mask,
            goal_constraints,
            path_constraints,
            sdf_voxels,
            collision_points,
            is_grasp_stage,
            initial_joint_pos,
            from_scratch=False,
            ):
        """
        Args:
            - ee_pose (np.ndarray): [7], [x, y, z, qx, qy, qz, qw] end effector pose.
            - keypoints (np.ndarray): [M, 3] keypoint positions.
            - keypoint_movable_mask (bool): [M] boolean array indicating whether the keypoint is on the grasped object.
            - goal_constraints (List[Callable]): subgoal constraint functions.
            - path_constraints (List[Callable]): path constraint functions.
            - sdf_voxels (np.ndarray): [X, Y, Z] signed distance field of the environment.
            - collision_points (np.ndarray): [N, 3] point cloud of the object.
            - is_grasp_stage (bool): whether the current stage is a grasp stage.
            - initial_joint_pos (np.ndarray): [N] initial joint positions of the robot.
            - from_scratch (bool): whether to start from scratch.
        Returns:
            - result (scipy.optimize.OptimizeResult): optimization result.
            - debug_dict (dict): debug information.
        """
        # downsample collision points
        if collision_points is not None and collision_points.shape[0] > self.config['max_collision_points']:
            collision_points = farthest_point_sampling(collision_points, self.config['max_collision_points'])
        sdf_func = self._setup_sdf(sdf_voxels)
        # ====================================
        # = setup bounds and initial guess
        # ====================================
        ee_pose = ee_pose.astype(np.float64)
        ee_pose_homo = T.pose2mat([ee_pose[:3], ee_pose[3:]])
        ee_pose_euler = np.concatenate([ee_pose[:3], T.quat2euler(ee_pose[3:])])
        # normalize opt variables to [0, 1]
        pos_bounds_min = self.config['bounds_min']
        pos_bounds_max = self.config['bounds_max']
        rot_bounds_min = np.array([-np.pi, -np.pi, -np.pi])  # euler angles
        rot_bounds_max = np.array([np.pi, np.pi, np.pi])  # euler angles
        og_bounds = [(b_min, b_max) for b_min, b_max in zip(np.concatenate([pos_bounds_min, rot_bounds_min]), np.concatenate([pos_bounds_max, rot_bounds_max]))]
        bounds = [(-1, 1)] * len(og_bounds)
        if not from_scratch and self.last_opt_result is not None:
            init_sol = self.last_opt_result.x
        else:
            init_sol = normalize_vars(ee_pose_euler, og_bounds)  # start from the current pose
            from_scratch = True

        # ====================================
        # = other setup
        # ====================================
        collision_points_centered, keypoints_centered = self._center_collision_points_and_keypoints(ee_pose_homo, collision_points, keypoints, keypoint_movable_mask)
        aux_args = (og_bounds,
                    keypoints_centered,
                    keypoint_movable_mask,
                    goal_constraints,
                    path_constraints,
                    sdf_func,
                    collision_points_centered,
                    ee_pose_homo,
                    self.ik_solver,
                    initial_joint_pos,
                    self.reset_joint_pos,
                    is_grasp_stage)

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
                no_local_search=False,
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
        debug_dict['sol'] = opt_result.x
        debug_dict['msg'] = opt_result.message
        debug_dict['solve_time'] = solve_time
        debug_dict['from_scratch'] = from_scratch
        debug_dict['type'] = 'subgoal_solver'
        # unnormailze
        sol = unnormalize_vars(opt_result.x, og_bounds)
        sol = np.concatenate([sol[:3], T.euler2quat(sol[3:])])
        opt_result = self._check_opt_result(opt_result, debug_dict)
        # cache opt_result for future use if successful
        if opt_result.success:
            self.last_opt_result = copy.deepcopy(opt_result)
        return sol, debug_dict