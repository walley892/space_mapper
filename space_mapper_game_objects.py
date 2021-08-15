from kinect_wrapper import KinectWrapper
from registration import rough_capture_transformation
from engine.game_object import GameObject
from engine.rendering.standard_renderers import ColoredPointCloudRenderer, MultiTexturedMeshRenderer
from engine.rendering.renderer_factory import RenderMode
from engine.standard_components import ObjectMover
from gpu_utils import add_triangles_from_capture_2
from utils import homogeneous_transform
import numpy as np

class MultiTexturedMesh(GameObject):
    def __init__(self):
        super().__init__()
        self.renderer = MultiTexturedMeshRenderer()
        self.add_renderer(self.renderer)
        self.renderer.render_mode = RenderMode.ELEMENTS

    def update_mesh(self, verts, elements, texs, uvs, material_ids):
        self.renderer.vertex_buffer = verts.astype(np.float32)
        self.renderer.tex = np.stack(texs)
        self.renderer.elements = elements.astype(np.uint32).flatten()
        self.renderer.vertex_uvs = uvs
        self.renderer.vertex_material_id = material_ids
        self.renderer.n_materials = len(texs)

class PointCloud(GameObject):
    def __init__(self):
        super().__init__()
        self.renderer = ColoredPointCloudRenderer()
        self.add_renderer(self.renderer)

    def update_points_and_colors(self, verts, colors):
        self.renderer.vertex_buffer = verts
        self.renderer.color_buffer = colors
        self.renderer.n_points = len(verts)

class KinectCapturer(GameObject):
    def __init__(self):
        super().__init__()
        self.state = 'capturing'
        self.render_mode = 'points'

        self.k = KinectWrapper()
        self.k.get_capture()
        self.current_capture = self.k.get_capture().downsample(24)
        
        self.captures = []
        # maps from captures[i] -> captures[i+1]
        self.rough_transformations = []
        self.current_pc = None
        self.current_mesh = None
        self.preview_pc = None
        self.preview_mesh = None
        self.preview_transformation = None
        
        self.mesh = MultiTexturedMesh()
        self.point_cloud = PointCloud()
        self.mesh.set_parent(self)
        self.point_cloud.set_parent(self)
        self.mesh.active = False

        self.add_component(ObjectMover())
        self.render_capture()
   
    def render_capture(self):
        if self.render_mode == 'mesh':
            self.mesh.update_mesh(
                self.current_capture.get_points().astype(np.float32),
                self.current_capture.get_mesh_triangles().astype(np.uint32),
                (self.current_capture.get_texture().astype(np.float32),),
                self.current_capture.get_vertex_uvs().astype(np.float32),
                np.ones(len(self.current_capture.get_points())//3).astype(np.uint32)
            )
        else:
            self.point_cloud.update_points_and_colors(
                self.current_capture.get_points().astype(np.float32),
                self.current_capture.get_colors().astype(np.float32),
            )

    def update(self):
        if self.state == 'capturing':
            self.capture()
            self.render_capture()
        if self.state == 'captured':
            self.setup_preview()
        if self.state == 'previewing':
            self.render_preview()
        if self.state == 'preview_current':
            self.render_current_map()

    def setup_preview(self):
        if self.current_pc is None:
            self.preview_pc = (
                self.current_capture.get_points().astype(np.float32),
                self.current_capture.get_colors().astype(np.float32),
            )
            cdn = self.current_capture.downsample(2)
            self.preview_mesh = (
                cdn.get_points().astype(np.float32),
                cdn.get_mesh_triangles().astype(np.uint32),
                [cdn.get_texture().astype(np.float32)],
                cdn.get_vertex_uvs().astype(np.float32),
                np.ones(len(cdn.get_points())//3).astype(np.uint32)
            )
        else:
            transformation = rough_capture_transformation(self.captures[-1], self.current_capture)
            self.preview_transformation = transformation
            new_points = homogeneous_transform(
                self.current_pc[0], 
                transformation
            )
            self.preview_pc = (
                np.concatenate((
                    new_points.reshape(-1, 3),
                    self.current_capture.get_points().reshape(-1, 3),
            )).astype(np.float32),
                np.concatenate((
                    self.current_pc[1],
                    self.current_capture.get_colors().astype(np.float32),
                )),
            )
            
            transformations = [np.linalg.inv(transformation)]

            for i, t in enumerate(reversed(self.rough_transformations)):
                transformations.append(np.matmul(np.linalg.inv(t), transformations[-1]))


            self.preview_mesh = add_triangles_from_capture_2(
                self.current_capture.downsample(2),
                (
                    homogeneous_transform(
                        self.current_mesh[0],
                        transformation,
                    ),
                    self.current_mesh[1],
                    self.current_mesh[2],
                    self.current_mesh[3],
                    self.current_mesh[4]
                ),
                list(map(lambda x: x.downsample(2), self.captures)),
                list(reversed(transformations))
            )
        self.state = 'previewing'
        return
    
    def render_preview(self):
        if self.render_mode == 'points':
            self.point_cloud.update_points_and_colors(
                self.preview_pc[0],
                self.preview_pc[1],
            )
        else:
            self.mesh.update_mesh(
                self.preview_mesh[0],
                self.preview_mesh[1],
                self.preview_mesh[2],
                self.preview_mesh[3],
                self.preview_mesh[4],
            )
    
    def render_current_map(self):
        if self.render_mode == 'points':
            self.point_cloud.update_points_and_colors(
                self.current_pc[0],
                self.current_pc[1],
            )
        else:
            self.mesh.update_mesh(
                self.current_mesh[0],
                self.current_mesh[1],
                self.current_mesh[2],
                self.current_mesh[3],
                self.current_mesh[4],
            )

    def capture(self):
        self.current_capture = self.k.get_capture().downsample(4)

    def take_capture(self):
        self.state = 'captured'

    def abort_capture(self):
        self.state = 'capturing'

    def commit_capture(self):
        self.current_pc = self.preview_pc
        self.current_mesh = self.preview_mesh
        self.captures.append(self.current_capture)
        if self.preview_transformation is not None:
            self.rough_transformations.append(self.preview_transformation)
        self.state = 'capturing'

    def keyboard_callback(self, key):
        if key == b'p':
            self.change_render_mode()
        if key == b'i':
            self.take_capture()
        if key == b'u':
            self.abort_capture()
        if key == b'o':
            self.commit_capture()
        if key == b'9':
            self.export()
    
    def toggle_preview(self):
        if self.state == 'preview_current':
            self.state = 'capturing'
        else:
            self.state = 'preview_current'

    def export(self, path='captures/'):
        import os
        try:
            os.mkdir(path)
        except:
            import shutil
            shutil.rmtree(path)
            os.mkdir(path)
        for i, capture in enumerate(self.captures):
            capture.export('{}/capture_{}'.format(path, i))
        for i, t in enumerate(self.rough_transformations):
            np.save('{}/transformation_{}_{}'.format(path, i, i + 1), t)
    
    def change_render_mode(self):
        if self.render_mode == 'points':
            self.render_mode = 'mesh'
            self.mesh.active = True
            self.point_cloud.active = False
        else:
            self.render_mode = 'points'
            self.mesh.active = False
            self.point_cloud.active = True
