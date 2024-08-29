from omnigibson.sensors.vision_sensor import VisionSensor
import transform_utils as T
import numpy as np

class OGCamera:
    """
    Defines the camera class
    """
    def __init__(self, og_env, config) -> None:        
        self.cam = insert_camera(name=config['name'], og_env=og_env, width=config['resolution'], height=config['resolution'])
        self.cam.set_position_orientation(config['position'], config['orientation'])
        self.intrinsics = get_cam_intrinsics(self.cam)
        self.extrinsics = get_cam_extrinsics(self.cam)

    def get_params(self):
        """
        Get the intrinsic and extrinsic parameters of the camera
        """
        return {"intrinsics": self.intrinsics, "extrinsics": self.extrinsics}
    
    def get_obs(self):
        """
        Gets the image observation from the camera.
        Assumes have rendered befor calling this function.
        No semantic handling here for now.
        """
        obs = self.cam.get_obs()
        ret = {}
        ret["rgb"] = obs[0]["rgb"][:,:,:3]  # H, W, 3
        ret["depth"] = obs[0]["depth_linear"]  # H, W
        ret["points"] = pixel_to_3d_points(ret["depth"], self.intrinsics, self.extrinsics)  # H, W, 3
        ret["seg"] = obs[0]["seg_semantic"]  # H, W
        ret["intrinsic"] = self.intrinsics
        ret["extrinsic"] = self.extrinsics
        return ret

def insert_camera(name, og_env, width=480, height=480):
    try:
        cam = VisionSensor(
            prim_path=f"/World/{name}",
            name=name,
            image_width=width,
            image_height=height,
            modalities=["rgb", "depth_linear", "seg_semantic"]
        )
    except TypeError:
        cam = VisionSensor(
            relative_prim_path=f"/{name}",
            name=name,
            image_width=width,
            image_height=height,
            modalities=["rgb", "depth_linear", "seg_semantic"]
        )
    
    try:
        cam.load()
    except TypeError:
        cam.load(og_env.scene)
    cam.initialize()
    return cam

def get_cam_intrinsics(cam):
    """
    Get the intrinsics matrix for a VisionSensor object
    ::param cam: VisionSensor object
    ::return intrinsics: 3x3 numpy array
    """
    img_width = cam.image_width
    img_height = cam.image_height

    if img_width != img_height:
        raise ValueError("Only square images are supported")

    apert = cam.prim.GetAttribute("horizontalAperture").Get()
    focal_len_in_pixel = cam.focal_length * img_width / apert

    intrinsics = np.eye(3)
    intrinsics[0,0] = focal_len_in_pixel
    intrinsics[1,1] = focal_len_in_pixel
    intrinsics[0,2] = img_width / 2
    intrinsics[1,2] = img_height / 2

    return intrinsics

def get_cam_extrinsics(cam):
    return T.pose_inv(T.pose2mat(cam.get_position_orientation()))

def pixel_to_3d_points(depth_image, intrinsics, extrinsics):
    # Get the shape of the depth image
    H, W = depth_image.shape

    # Create a grid of (x, y) coordinates corresponding to each pixel in the image
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    # Unpack the intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Convert pixel coordinates to normalized camera coordinates
    z = depth_image
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    # Stack the coordinates to form (H, W, 3)
    camera_coordinates = np.stack((x, y, z), axis=-1)

    # Reshape to (H*W, 3) for matrix multiplication
    camera_coordinates = camera_coordinates.reshape(-1, 3)

    # Convert to homogeneous coordinates (H*W, 4)
    camera_coordinates_homogeneous = np.hstack((camera_coordinates, np.ones((camera_coordinates.shape[0], 1))))

    # additional conversion to og convention
    T_mod = np.array([[1., 0., 0., 0., ],
              [0., -1., 0., 0.,],
              [0., 0., -1., 0.,],
              [0., 0., 0., 1.,]])
    camera_coordinates_homogeneous = camera_coordinates_homogeneous @ T_mod

    # Apply extrinsics to get world coordinates
    # world_coordinates_homogeneous = camera_coordinates_homogeneous @ extrinsics.T
    world_coordinates_homogeneous = T.pose_inv(extrinsics) @ (camera_coordinates_homogeneous.T)
    world_coordinates_homogeneous = world_coordinates_homogeneous.T

    # Convert back to non-homogeneous coordinates
    world_coordinates = world_coordinates_homogeneous[:, :3] / world_coordinates_homogeneous[:, 3, np.newaxis]

    # Reshape back to (H, W, 3)
    world_coordinates = world_coordinates.reshape(H, W, 3)

    return world_coordinates

def point_to_pixel(pt, intrinsics, extrinsics):
    """
    pt -- (N, 3) 3d points in world frame
    intrinsics -- (3, 3) intrinsics matrix
    extrinsics -- (4, 4) extrinsics matrix
    """
    pt_in_cam = extrinsics @ np.hstack((pt, np.ones((pt.shape[0], 1)))).T # (4, N)
    # multiply y, z by -1
    pt_in_cam[1, :] *= -1
    pt_in_cam[2, :] *= -1
    pt_in_cam = pt_in_cam[:3, :]
    pt_in_cam = intrinsics @ pt_in_cam
    pt_in_cam /= pt_in_cam[2, :]

    return pt_in_cam[:2, :].T