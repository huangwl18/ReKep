import open3d as o3d
import numpy as np
import matplotlib
import cv2
import transform_utils as T
from utils import filter_points_by_bounds, batch_transform_points

def add_to_visualize_buffer(visualize_buffer, visualize_points, visualize_colors):
    assert visualize_points.shape[0] == visualize_colors.shape[0], f'got {visualize_points.shape[0]} for points and {visualize_colors.shape[0]} for colors'
    if len(visualize_points) == 0:
        return
    assert visualize_points.shape[1] == 3
    assert visualize_colors.shape[1] == 3
    # assert visualize_colors.max() <= 1.0 and visualize_colors.min() >= 0.0
    visualize_buffer["points"].append(visualize_points)
    visualize_buffer["colors"].append(visualize_colors)

def generate_nearby_points(point, num_points_per_side=5, half_range=0.005):
    if point.ndim == 1:
        offsets = np.linspace(-1, 1, num_points_per_side)
        offsets_meshgrid = np.meshgrid(offsets, offsets, offsets)
        offsets_array = np.stack(offsets_meshgrid, axis=-1).reshape(-1, 3)
        nearby_points = point + offsets_array * half_range
        return nearby_points.reshape(-1, 3)
    else:
        assert point.shape[1] == 3, "point must be (N, 3)"
        assert point.ndim == 2, "point must be (N, 3)"
        # vectorized version
        offsets = np.linspace(-1, 1, num_points_per_side)
        offsets_meshgrid = np.meshgrid(offsets, offsets, offsets)
        offsets_array = np.stack(offsets_meshgrid, axis=-1).reshape(-1, 3)
        nearby_points = point[:, None, :] + offsets_array
        return nearby_points

class Visualizer:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.color = np.array([0.05, 0.55, 0.26])
        self.world2viewer = np.array([
            [0.3788, 0.3569, -0.8539, 0.0],
            [0.9198, -0.0429, 0.3901, 0.0],
            [-0.1026, 0.9332, 0.3445, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]).T

    def show_img(self, rgb):
        cv2.imshow('img', rgb[..., ::-1])
        cv2.waitKey(0)
        print('showing image, click on the window and press "ESC" to close and continue')
        cv2.destroyAllWindows()
    
    def show_pointcloud(self, points, colors):
        # transform to viewer frame
        points = np.dot(points, self.world2viewer[:3, :3].T) + self.world2viewer[:3, 3]
        # clip color to [0, 1]
        colors = np.clip(colors, 0.0, 1.0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))  # float64 is a lot faster than float32 when added to o3d later
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        print('visualizing pointcloud, click on the window and press "ESC" to close and continue')
        o3d.visualization.draw_geometries([pcd])

    def _get_scene_points_and_colors(self):
        # scene
        cam_obs = self.env.get_cam_obs()
        scene_points = []
        scene_colors = []
        for cam_id in range(len(cam_obs)):
            cam_points = cam_obs[cam_id]['points'].reshape(-1, 3)
            cam_colors = cam_obs[cam_id]['rgb'].reshape(-1, 3) / 255.0
            # clip to workspace
            within_workspace_mask = filter_points_by_bounds(cam_points, self.bounds_min, self.bounds_max, strict=False)
            cam_points = cam_points[within_workspace_mask]
            cam_colors = cam_colors[within_workspace_mask]
            scene_points.append(cam_points)
            scene_colors.append(cam_colors)
        scene_points = np.concatenate(scene_points, axis=0)
        scene_colors = np.concatenate(scene_colors, axis=0)
        return scene_points, scene_colors

    def visualize_subgoal(self, subgoal_pose):
        visualize_buffer = {
            "points": [],
            "colors": []
        }
        # scene
        scene_points, scene_colors = self._get_scene_points_and_colors()
        add_to_visualize_buffer(visualize_buffer, scene_points, scene_colors)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # subgoal
        collision_points = self.env.get_collision_points(noise=False)
        # transform collision points to the subgoal frame
        ee_pose = self.env.get_ee_pose()
        ee_pose_homo = T.convert_pose_quat2mat(ee_pose)
        centering_transform = np.linalg.inv(ee_pose_homo)
        collision_points_centered = np.dot(collision_points, centering_transform[:3, :3].T) + centering_transform[:3, 3]
        transformed_collision_points = batch_transform_points(collision_points_centered, subgoal_pose_homo[None]).reshape(-1, 3)
        collision_points_colors = np.array([self.color] * len(collision_points))
        add_to_visualize_buffer(visualize_buffer, transformed_collision_points, collision_points_colors)
        # add keypoints
        keypoints = self.env.get_keypoint_positions()
        num_keypoints = keypoints.shape[0]
        color_map = matplotlib.colormaps["gist_rainbow"]
        keypoints_colors = [color_map(i / num_keypoints)[:3] for i in range(num_keypoints)]
        for i in range(num_keypoints):
            nearby_points = generate_nearby_points(keypoints[i], num_points_per_side=6, half_range=0.009)
            nearby_colors = np.tile(keypoints_colors[i], (nearby_points.shape[0], 1))
            nearby_colors = 0.5 * nearby_colors + 0.5 * np.array([1, 1, 1])
            add_to_visualize_buffer(visualize_buffer, nearby_points, nearby_colors)
        # visualize
        visualize_points = np.concatenate(visualize_buffer["points"], axis=0)
        visualize_colors = np.concatenate(visualize_buffer["colors"], axis=0)
        self.show_pointcloud(visualize_points, visualize_colors)

    def visualize_path(self, path):
        visualize_buffer = {
            "points": [],
            "colors": []
        }
        # scene
        scene_points, scene_colors = self._get_scene_points_and_colors()
        add_to_visualize_buffer(visualize_buffer, scene_points, scene_colors)
        # draw curve based on poses
        for t in range(len(path) - 1):
            start = path[t][:3]
            end = path[t + 1][:3]
            num_interp_points = int(np.linalg.norm(start - end) / 0.0002)
            interp_points = np.linspace(start, end, num_interp_points)
            interp_colors = np.tile([0.0, 0.0, 0.0], (num_interp_points, 1))
            # add a tint of white (the higher the j, the more white)
            whitening_coef = 0.3 + 0.5 * (t / len(path))
            interp_colors = (1 - whitening_coef) * interp_colors + whitening_coef * np.array([1, 1, 1])
            add_to_visualize_buffer(visualize_buffer, interp_points, interp_colors)
        # subsample path with a fixed step size
        step_size = 0.05
        subpath = [path[0]]
        for i in range(1, len(path) - 1):
            dist = np.linalg.norm(np.array(path[i][:3]) - np.array(subpath[-1][:3]))
            if dist > step_size:
                subpath.append(path[i])
        subpath.append(path[-1])
        path = np.array(subpath)
        path_length = path.shape[0]
        # path points
        collision_points = self.env.get_collision_points(noise=False)
        num_points = collision_points.shape[0]
        start_pose = self.env.get_ee_pose()
        centering_transform = np.linalg.inv(T.convert_pose_quat2mat(start_pose))
        collision_points_centered = np.dot(collision_points, centering_transform[:3, :3].T) + centering_transform[:3, 3]
        poses_homo = T.convert_pose_quat2mat(path[:, :7])  # the last number is gripper action
        transformed_collision_points = batch_transform_points(collision_points_centered, poses_homo).reshape(-1, 3)  # (num_poses, num_points, 3)
        # calculate color based on the timestep
        collision_points_colors = np.ones([path_length, num_points, 3]) * self.color[None, None]
        for t in range(path_length):
            whitening_coef = 0.3 + 0.5 * (t / path_length)
            collision_points_colors[t] = (1 - whitening_coef) * collision_points_colors[t] + whitening_coef * np.array([1, 1, 1])
        collision_points_colors = collision_points_colors.reshape(-1, 3)
        add_to_visualize_buffer(visualize_buffer, transformed_collision_points, collision_points_colors)
        # keypoints
        keypoints = self.env.get_keypoint_positions()
        num_keypoints = keypoints.shape[0]
        color_map = matplotlib.colormaps["gist_rainbow"]
        keypoints_colors = [color_map(i / num_keypoints)[:3] for i in range(num_keypoints)]
        for i in range(num_keypoints):
            nearby_points = generate_nearby_points(keypoints[i], num_points_per_side=6, half_range=0.009)
            nearby_colors = np.tile(keypoints_colors[i], (nearby_points.shape[0], 1))
            nearby_colors = 0.5 * nearby_colors + 0.5 * np.array([1, 1, 1])
            add_to_visualize_buffer(visualize_buffer, nearby_points, nearby_colors)
        # visualize
        visualize_points = np.concatenate(visualize_buffer["points"], axis=0)
        visualize_colors = np.concatenate(visualize_buffer["colors"], axis=0)
        self.show_pointcloud(visualize_points, visualize_colors)