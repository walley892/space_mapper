import open3d as o3d
import json
from capture import Capture
import numpy as np
import os

def _get_intrinsics(config):
    recorder = o3d.io.AzureKinectRecorder(config, 0)
    recorder.init_sensor()
    recorder.open_record('tmp.mkv')
    f = None
    while f is None:
        f = recorder.record_frame(True, True)
    recorder.close_record()
    del recorder

    reader = o3d.io.AzureKinectMKVReader()
    reader.open('tmp.mkv')
    metadata = reader.get_metadata()
    o3d.io.write_azure_kinect_mkv_metadata('tmp.json', metadata)
    
    f = open('tmp.json')
    info = json.load(f)

    mat = info['intrinsic_matrix']
    fx, fy, cx, cy = mat[0], mat[4], mat[6], mat[7]
    height = info['height']
    width = info['width']
    os.remove('tmp.mkv')
    os.remove('tmp.json')

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def _to_rgb(bgr):
    b = np.copy(bgr[:, 0])
    r = bgr[:, 2]
    bgr[:, 0] = r
    bgr[:, 2] = b
    return bgr

class KinectWrapper(object):
    def __init__(self):
        self.config = o3d.io.AzureKinectSensorConfig(
                {
                    'color_resolution': 'K4A_COLOR_RESOLUTION_3072P',
                    'depth_mode': 'K4A_DEPTH_MODE_NFOV_UNBINNED',
                    'synchronized_images_only': 'true',
                    'camera_fps': 'K4A_FRAMES_PER_SECOND_15'
                }
        )
        self.cam_to_world = np.linalg.inv(_get_intrinsics(self.config).intrinsic_matrix)
        self.ka = o3d.io.AzureKinectSensor(self.config)
        self.ka.connect(0)

    def get_capture(self):
        f = None
        while f is None:
            f = self.ka.capture_frame(True)
        return Capture.create_from_kinect_capture(f, np.copy(self.cam_to_world))
    
    def get_verts_and_colors(self):
        cap = self.ka.get_capture()
        while not (np.any(cap.depth) and np.any(cap.color)):
            cap = self.ka.get_capture()
        points = cap.depth_point_cloud.reshape(-1, 3)
        colors = ka.color_image_to_depth_camera(cap.color, cap.depth, self.ka.calibration, True)
        colors = colors.reshape(-1, 4)
        colors = _to_rgb(colors)*(1.0/255)
        return points*0.05, colors
        
    def get_point_cloud(self):
        points, colors = self.get_verts_and_colors()
        pc = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                points
            )
        )
        pc.colors = o3d.utility.Vector3dVector(colors[:,:3])
        return pc
