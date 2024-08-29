"""
Adapted from OmniGibson and the Lula IK solver
"""
import omnigibson.lazy as lazy
import numpy as np

class IKSolver:
    """
    Class for thinly wrapping Lula IK solver
    """

    def __init__(
        self,
        robot_description_path,
        robot_urdf_path,
        eef_name,
        reset_joint_pos,
        world2robot_homo,
    ):
        # Create robot description, kinematics, and config
        self.robot_description = lazy.lula.load_robot(robot_description_path, robot_urdf_path)
        self.kinematics = self.robot_description.kinematics()
        self.config = lazy.lula.CyclicCoordDescentIkConfig()
        self.eef_name = eef_name
        self.reset_joint_pos = reset_joint_pos
        self.world2robot_homo = world2robot_homo

    def solve(
        self,
        target_pose_homo,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        position_weight=1.0,
        orientation_weight=0.05,
        max_iterations=150,
        initial_joint_pos=None,
    ):
        """
        Backs out joint positions to achieve desired @target_pos and @target_quat

        Args:
            target_pose_homo (np.ndarray): [4, 4] homogeneous transformation matrix of the target pose in world frame
            position_tolerance (float): Maximum position error (L2-norm) for a successful IK solution
            orientation_tolerance (float): Maximum orientation error (per-axis L2-norm) for a successful IK solution
            position_weight (float): Weight for the relative importance of position error during CCD
            orientation_weight (float): Weight for the relative importance of position error during CCD
            max_iterations (int): Number of iterations used for each cyclic coordinate descent.
            initial_joint_pos (None or n-array): If specified, will set the initial cspace seed when solving for joint
                positions. Otherwise, will use self.reset_joint_pos

        Returns:
            ik_results (lazy.lula.CyclicCoordDescentIkResult): IK result object containing the joint positions and other information.
        """
        # convert target pose to robot base frame
        target_pose_robot = np.dot(self.world2robot_homo, target_pose_homo)
        target_pose_pos = target_pose_robot[:3, 3]
        target_pose_rot = target_pose_robot[:3, :3]
        ik_target_pose = lazy.lula.Pose3(lazy.lula.Rotation3(target_pose_rot), target_pose_pos)
        # Set the cspace seed and tolerance
        initial_joint_pos = self.reset_joint_pos if initial_joint_pos is None else np.array(initial_joint_pos)
        self.config.cspace_seeds = [initial_joint_pos]
        self.config.position_tolerance = position_tolerance
        self.config.orientation_tolerance = orientation_tolerance
        self.config.ccd_position_weight = position_weight
        self.config.ccd_orientation_weight = orientation_weight
        self.config.max_num_descents = max_iterations
        # Compute target joint positions
        ik_results = lazy.lula.compute_ik_ccd(self.kinematics, ik_target_pose, self.eef_name, self.config)
        return ik_results