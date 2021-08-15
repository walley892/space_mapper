import numpy as np
from gpu_utils import downsample_color_image, downsample_image, create_mesh_triangles_from_depth_image, create_mesh_uvs_from_height_and_width, create_vertices_from_depth_image, create_mesh, create_point_cloud, create_vertex_uvs_from_height_and_width

class Capture(object):
    @staticmethod
    def create_from_kinect_capture(capture, cam_to_world):
        cap = Capture.create_empty()
        cap.rgb = np.asarray(capture.color)/255.0
        cap.cam_to_world = cam_to_world
        cap.depth = np.asarray(capture.depth).astype(np.float32)/1000.0
        cap.rgb_pc = cap.rgb
        return cap
    
    @staticmethod
    def create_empty():
        cap = Capture()
        cap.point_cloud = None
        cap.points = None
        cap.colors = None
        cap.mesh=None
        cap.mesh_triangles = None
        cap.mesh_vertex_uvs = None
        return cap

    def get_points(self, out = None):
        if self.points is None:
            self.points = create_vertices_from_depth_image(self.depth, self.cam_to_world)
        return self.points

    def get_colors(self):
        if self.colors is None:
            self.colors = self.rgb_pc.reshape(-1, 3)
        return self.colors
    
    def get_mesh_vertices(self, out = None):
        return self.get_points()

    def get_texture(self):
        return self.rgb

    def get_mesh_triangles(self, out = None):
        if self.mesh_triangles is None:
            self.mesh_triangles = create_mesh_triangles_from_depth_image(self.depth)
        return self.mesh_triangles

    def get_vertex_uvs(self):
        if self.mesh_vertex_uvs is None:
            self.mesh_vertex_uvs =  create_vertex_uvs_from_height_and_width(self.depth.shape[0], self.depth.shape[1])
        return self.mesh_vertex_uvs
    
    def get_triangle_uvs(self):
        if self.mesh_triangle_uvs is None:
            self.mesh_triangle_uvs =  create_mesh_uvs_from_height_and_width(self.depth.shape[0], self.depth.shape[1])
        return self.mesh_triangle_uvs
    
    def get_point_cloud(self):
        if self.point_cloud is None:
            self.point_cloud = create_point_cloud(self.rgb_pc, self.depth, self.cam_to_world)
        return self.point_cloud

    def downsample(self, downsample_n):
        cap = Capture.create_empty()
        cap.rgb = np.copy(self.rgb)
        cap.rgb_pc, _ = downsample_color_image(self.rgb, self.cam_to_world, downsample_n)
        cap.depth, cap.cam_to_world = downsample_image(self.depth, self.cam_to_world, downsample_n)
        cap.point_cloud=None
        cap.mesh=None
        cap.points = None
        cap.colors = None
        return cap

    #TODO: rename with  alphas
    def get_verts_and_colors(self):
        return self.flat_points, self.flat_colors_with_alphas
    
    def get_mesh(self):
        if self.mesh is None:
            self.mesh = create_mesh(self.rgb, self.depth, self.cam_to_world)
        return self.mesh

    def export(self, path):
        np.save(path+'_rgb', self.rgb)
        np.save(path+'_rgb_pc', self.rgb_pc)
        np.save(path+'_depth', self.depth)
        np.save(path+'_cam_to_world', self.cam_to_world)

    @staticmethod
    def restore(path):
        cap = Capture.create_empty()
        cap.rgb = np.load(path+'_rgb.npy')
        cap.rgb_pc = np.load(path+'_rgb_pc.npy')
        cap.depth = np.load(path+'_depth.npy')
        cap.cam_to_world = np.load(path+'_cam_to_world.npy')
        return cap
