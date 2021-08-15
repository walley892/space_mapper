import pycuda.driver as cuda
from allocator import *
import numpy as np
from pycuda.autoinit import context
import open3d as o3d
from gpu_module import gpu_mod

N_THREADS = 512
N_BLOCKS = 32

_create_vertices_from_depth_image_big = gpu_mod.get_function("create_vertices_from_depth_image_big")
_create_mesh_triangles_from_depth_image = gpu_mod.get_function("create_mesh_triangles_from_depth_image")
_create_mesh_triangles_from_depth_image_big = gpu_mod.get_function("create_mesh_triangles_from_depth_image_big")
_create_mesh_triangles_from_depth_image_with_mask = gpu_mod.get_function("create_mesh_triangles_from_depth_image_with_mask")
_create_mesh_triangles_from_depth_image_with_mask_big = gpu_mod.get_function("create_mesh_triangles_from_depth_image_with_mask_big")
_create_mesh_uvs_from_height_and_width = gpu_mod.get_function("create_mesh_uvs_from_height_and_width")
_create_mesh_uvs_from_height_and_width_with_mask = gpu_mod.get_function("create_mesh_uvs_from_height_and_width_with_mask")
_create_mesh_uvs_from_height_and_width_big = gpu_mod.get_function("create_mesh_uvs_from_height_and_width_big")
_create_vertex_uvs_from_height_and_width_big = gpu_mod.get_function("create_vertex_uvs_from_height_and_width_big")

_downsample_image = gpu_mod.get_function("downsample_image")
_downsample_image_big = gpu_mod.get_function("downsample_image_big")
_calculate_invalid_mesh_uv_indices = gpu_mod.get_function("calculate_invalid_mesh_uv_indices")
_downsample_color_image = gpu_mod.get_function("downsample_color_image")
_downsample_color_image_big = gpu_mod.get_function("downsample_color_image_big")
_compute_overlaps = gpu_mod.get_function("compute_overlaps")
_compute_overlaps_big = gpu_mod.get_function("compute_overlaps_big")
_compute_im1_coords = gpu_mod.get_function("compute_im1_coords")
_copy_to_im1_frame = gpu_mod.get_function("copy_to_im1_frame")

def create_vertices_from_depth_image(depth_img, cam_to_world, xoffset = 0, yoffset = 0, out=None):
    img_buffer, img_buffer_index = get_buffer_for_image(depth_img.shape, np.float64)
    verts_buffer, verts_buffer_index = get_buffer_for_image((depth_img.shape[0], depth_img.shape[1], 3), np.float64)
    cam_to_world_buffer, cam_to_world_buffer_index = get_buffer_for_image((3, 3), np.float64)
    cuda.memcpy_htod(img_buffer.gpudata, depth_img.astype(np.float64))
    cuda.memcpy_htod(cam_to_world_buffer.gpudata, cam_to_world.astype(np.float64))
    height = np.int32(depth_img.shape[0])
    width = np.int32(depth_img.shape[1])
    num = height*width
    stride = np.int32(max(((num)//(N_THREADS-1))//N_BLOCKS, 1))
    n_blocks = N_BLOCKS
    if stride == 1:
        n_blocks = 1
    _create_vertices_from_depth_image_big(
        img_buffer, 
        verts_buffer, 
        cam_to_world_buffer, 
        width, 
        stride,
        num,
        np.int32(xoffset),
        np.int32(yoffset),
        block=(min(N_THREADS, num),1,1),
        grid=(n_blocks, 1)
    )
    context.synchronize()
    if out is None:
        out = np.empty(np.prod(depth_img.shape)*3).astype(np.float64)
    cuda.memcpy_dtoh(out, verts_buffer.gpudata)
    mark_buffer_free(img_buffer_index)
    mark_buffer_free(verts_buffer_index)
    mark_buffer_free(cam_to_world_buffer_index)
    return out

def downsample_color_image_gpu(depth_img, cam_to_world, downsample_n, out=None):
    img_buffer, img_buffer_index = get_buffer_for_image(depth_img.shape, np.float64)
    img_dn_buffer, img_dn_buffer_index = get_buffer_for_image((depth_img.shape[0]//downsample_n, depth_img.shape[1]//downsample_n, 3), np.float64)
    cam_to_world_buffer, cam_to_world_buffer_index = get_buffer_for_image((3, 3), np.float64)
    cuda.memcpy_htod(img_buffer.gpudata, depth_img.astype(np.float64))
    cuda.memcpy_htod(cam_to_world_buffer.gpudata, cam_to_world.astype(np.float64))
    height_dn = depth_img.shape[0]//downsample_n
    width_dn = depth_img.shape[1]//downsample_n
    num = np.int32(height_dn*width_dn)
    stride = np.int32(max((num+1)//N_THREADS, 2))
    stride = np.int32(max(((num+1)//(N_THREADS-1))//N_BLOCKS, 1))
    n_blocks = N_BLOCKS
    if stride == 1:
        n_blocks = 1
    _downsample_color_image_big(
        img_buffer,
        img_dn_buffer,
        np.int32(downsample_n),
        np.int32(depth_img.shape[1]),
        stride,
        num,
        block=(min(N_THREADS, num), 1, 1),
        grid=(n_blocks, 1)
    )
    context.synchronize()
    out = cuda.from_device(img_dn_buffer.gpudata, (num*3, ), np.float64)
    '''
    if not out:
        out = np.zeros((depth_img.shape[0]//downsample_n)*(depth_img.shape[1]//downsample_n)*3).astype(np.float64)
    cuda.memcpy_dtoh(out, img_dn_buffer.gpudata)
    '''
    mark_buffer_free(img_buffer_index)
    mark_buffer_free(img_dn_buffer_index)
    mark_buffer_free(cam_to_world_buffer_index)
    return out


def downsample_image_gpu(depth_img, cam_to_world, downsample_n, out=None):
    img_buffer, img_buffer_index = get_buffer_for_image(depth_img.shape, np.float64)
    img_dn_buffer, img_dn_buffer_index = get_buffer_for_image((depth_img.shape[0]//downsample_n, depth_img.shape[1]//downsample_n), np.float64)
    cam_to_world_buffer, cam_to_world_buffer_index = get_buffer_for_image((3, 3), np.float64)
    cuda.memcpy_htod(img_buffer.gpudata, depth_img.astype(np.float64))
    cuda.memcpy_htod(cam_to_world_buffer.gpudata, cam_to_world.astype(np.float64))
    ''' 
    _downsample_image(
        img_buffer,
        img_dn_buffer,
        np.int32(downsample_n),
        np.int32(depth_img.shape[1]),
        block=(depth_img.shape[0]//downsample_n, 1, 1)
    )
    '''
    height_dn = depth_img.shape[0]//downsample_n
    width_dn = depth_img.shape[1]//downsample_n
    num = np.int32(height_dn*width_dn)
    stride = np.int32(max(((num+1)//(N_THREADS-1))//N_BLOCKS, 1))
    n_blocks = N_BLOCKS
    if stride == 1:
        n_blocks = 1
    _downsample_image_big(
        img_buffer,
        img_dn_buffer,
        np.int32(downsample_n),
        np.int32(depth_img.shape[1]),
        stride,
        num, 
        block=(min(N_THREADS, num), 1, 1),
        grid=(n_blocks, 1),
    )
    context.synchronize()
    out = cuda.from_device(img_dn_buffer.gpudata, (num, ), np.float64)
    '''
    if not out:
        out = np.zeros((depth_img.shape[0]//downsample_n)*(depth_img.shape[1]//downsample_n)).astype(np.float64)
    cuda.memcpy_dtoh(out, img_dn_buffer.gpudata)
    '''
    mark_buffer_free(img_buffer_index)
    mark_buffer_free(img_dn_buffer_index)
    mark_buffer_free(cam_to_world_buffer_index)
    return out

def compute_im1_coords(img_1, img_2, world_to_one, two_to_world, two_to_one, out = None):
    img_2_buffer, img_2_buffer_index = get_buffer_for_image(img_2.shape, np.float64)
    im1_coords_buffer, im1_coords_buffer_index = get_buffer_for_image((img_2.shape[0], img_2.shape[1], 2), np.int32)
    world_to_one_buffer, world_to_one_buffer_index = get_buffer_for_image((3, 3), np.float64)
    two_to_world_buffer, two_to_world_buffer_index = get_buffer_for_image((3, 3), np.float64)
    two_to_one_buffer, two_to_one_buffer_index = get_buffer_for_image((4, 4), np.float64)
    cuda.memcpy_htod(world_to_one_buffer.gpudata, world_to_one.astype(np.float64))
    cuda.memcpy_htod(two_to_world_buffer.gpudata, two_to_world.astype(np.float64))
    cuda.memcpy_htod(two_to_one_buffer.gpudata, two_to_one.astype(np.float64))
    cuda.memcpy_htod(img_2_buffer.gpudata, img_2.astype(np.float64))
    height_1, width_1 = img_1.shape
    height_2, width_2 = img_2.shape
    _compute_im1_coords(
        img_2_buffer,
        world_to_one_buffer,
        two_to_world_buffer,
        two_to_one_buffer,
        np.int32(width_1),
        np.int32(width_2),
        np.int32(height_1),
        im1_coords_buffer,
        block=(height_2, 1, 1)
    )

    context.synchronize()
    if not out:
        out = np.zeros(np.prod(img_2.shape)*2).astype(np.int32)
    cuda.memcpy_dtoh(out, im1_coords_buffer.gpudata)
    mark_buffer_free(im1_coords_buffer_index)
    mark_buffer_free(world_to_one_buffer_index)
    mark_buffer_free(two_to_one_buffer_index)
    mark_buffer_free(img_2_buffer_index)
    return out

def copy_to_im1_frame(img_1, img_2, img_big, world_to_one, two_to_world, two_to_one, xoffset, yoffset):
    img_2_buffer, img_2_buffer_index = get_buffer_for_image(img_2.shape, np.float64)
    img_big_buffer, img_big_buffer_index = get_buffer_for_image(img_big.shape, np.float64)
    world_to_one_buffer, world_to_one_buffer_index = get_buffer_for_image((3, 3), np.float64)
    two_to_world_buffer, two_to_world_buffer_index = get_buffer_for_image((3, 3), np.float64)
    two_to_one_buffer, two_to_one_buffer_index = get_buffer_for_image((4, 4), np.float64)
    cuda.memcpy_htod(world_to_one_buffer.gpudata, world_to_one.astype(np.float64))
    cuda.memcpy_htod(two_to_world_buffer.gpudata, two_to_world.astype(np.float64))
    cuda.memcpy_htod(two_to_one_buffer.gpudata, two_to_one.astype(np.float64))
    cuda.memcpy_htod(img_2_buffer.gpudata, img_2.astype(np.float64))
    cuda.memcpy_htod(img_big_buffer.gpudata, img_big.astype(np.float64))
    height_1, width_1 = img_1.shape
    height_big, width_big = img_big.shape
    height_2, width_2 = img_2.shape

    _copy_to_im1_frame(
        img_2_buffer,
        world_to_one_buffer,
        two_to_world_buffer,
        two_to_one_buffer,
        np.int32(width_1),
        np.int32(width_2),
        np.int32(width_big),
        np.int32(height_1),
        np.int32(height_2),
        np.int32(xoffset),
        np.int32(yoffset),
        img_big_buffer,
        block=(height_2, 1, 1)
    )

    context.synchronize()
    cuda.memcpy_dtoh(img_big, img_big_buffer.gpudata)
    mark_buffer_free(world_to_one_buffer_index)
    mark_buffer_free(two_to_one_buffer_index)
    mark_buffer_free(img_2_buffer_index)
    mark_buffer_free(img_big_buffer_index)
    return img_big


def compute_overlaps(img_1, img_2, world_to_one, two_to_world, two_to_one, capture_index, overlaps_out = None, index_out = None):
    img_1_buffer, img_1_buffer_index = get_buffer_for_image(img_1.shape, np.float64)
    img_2_buffer, img_2_buffer_index = get_buffer_for_image(img_2.shape, np.float64)
    overlaps_buffer, overlaps_buffer_index = get_buffer_for_image(img_2.shape, np.uint32)
    index_buffer, index_buffer_index = get_buffer_for_image(img_2.shape, np.uint32)
    world_to_one_buffer, world_to_one_buffer_index = get_buffer_for_image((3, 3), np.float64)
    two_to_world_buffer, two_to_world_buffer_index = get_buffer_for_image((3, 3), np.float64)
    two_to_one_buffer, two_to_one_buffer_index = get_buffer_for_image((4, 4), np.float64)
    cuda.memcpy_htod(world_to_one_buffer.gpudata, world_to_one.astype(np.float64))
    cuda.memcpy_htod(two_to_world_buffer.gpudata, two_to_world.astype(np.float64))
    cuda.memcpy_htod(two_to_one_buffer.gpudata, two_to_one.astype(np.float64))
    cuda.memcpy_htod(img_1_buffer.gpudata, img_1.astype(np.float64))
    cuda.memcpy_htod(img_2_buffer.gpudata, img_2.astype(np.float64))
    if overlaps_out is not None:
        cuda.memcpy_htod(overlaps_buffer.gpudata, overlaps_out.astype(np.uint32))
    if index_out is not None:
        cuda.memcpy_htod(index_buffer.gpudata, index_out.astype(np.uint32))
        
    height_1, width_1 = img_1.shape
    height_2, width_2 = img_2.shape
    num = height_2*width_2
    stride = np.int32(max((num)//(N_THREADS-1), 2))
    _compute_overlaps(
        img_1_buffer,
        img_2_buffer,
        world_to_one_buffer,
        two_to_world_buffer,
        two_to_one_buffer,
        np.int32(width_1),
        np.int32(width_2),
        np.int32(height_1),
        overlaps_buffer,
        index_buffer,
        np.uint32(capture_index),
        np.uint32(num),
        block=(height_2, 1, 1)
    )
    '''
    _compute_overlaps_big(
        img_1_buffer,
        img_2_buffer,
        world_to_one_buffer,
        two_to_world_buffer,
        two_to_one_buffer,
        np.int32(width_1),
        np.int32(width_2),
        np.int32(height_1),
        np.int32(num),
        overlaps_buffer,
        block=(min(N_THREADS, num), 1, 1)
    )
    '''

    context.synchronize()
    if overlaps_out is None:
        overlaps_out = np.zeros(np.prod(img_2.shape)).astype(np.uint32)
    if index_out is None:
        index_out = np.zeros(np.prod(img_2.shape)).astype(np.uint32)
    cuda.memcpy_dtoh(overlaps_out, overlaps_buffer.gpudata)
    cuda.memcpy_dtoh(index_out, index_buffer.gpudata)
    mark_buffer_free(overlaps_buffer_index)
    mark_buffer_free(index_buffer_index)
    mark_buffer_free(world_to_one_buffer_index)
    mark_buffer_free(two_to_one_buffer_index)
    mark_buffer_free(img_1_buffer_index)
    mark_buffer_free(img_2_buffer_index)
    return overlaps_out, index_out

def create_mesh_triangles_from_depth_image(depth_img, out = None):
    img_buffer, img_buffer_index = get_buffer_for_image(depth_img.shape, np.float64)
    tris_buffer, tris_buffer_index = get_buffer_for_image((depth_img.shape[0]*depth_img.shape[1]*6, ), np.int32)
    cuda.memcpy_htod(img_buffer.gpudata, depth_img.astype(np.float64))
    height = np.int32(depth_img.shape[0])
    width = np.int32(depth_img.shape[1])
    num = height*width
    stride = np.int32(max((num)//(N_THREADS-1), 2))
    _create_mesh_triangles_from_depth_image_big(
        img_buffer, 
        tris_buffer, 
        width, 
        height,
        stride,
        num,
        block=(min(N_THREADS, num),1,1)
    )
    context.synchronize()
    if not out:
        out = np.zeros(np.prod(depth_img.shape)*6).astype(np.int32)
    cuda.memcpy_dtoh(out, tris_buffer.gpudata)
    mark_buffer_free(img_buffer_index)
    mark_buffer_free(tris_buffer_index)
    return out

def create_mesh_triangles_from_depth_image_with_mask(depth_img, mask, indices, capture_index, out = None):
    img_buffer, img_buffer_index = get_buffer_for_image(depth_img.shape, np.float64)
    mask_buffer, mask_buffer_index = get_buffer_for_image(mask.shape, np.uint32)
    index_buffer, index_buffer_index = get_buffer_for_image(mask.shape, np.uint32)
    tris_buffer, tris_buffer_index = get_buffer_for_image((depth_img.shape[0]*depth_img.shape[1]*6, ), np.int32)
    cuda.memcpy_htod(img_buffer.gpudata, depth_img.astype(np.float64))
    cuda.memcpy_htod(mask_buffer.gpudata, mask.astype(np.uint32))
    cuda.memcpy_htod(index_buffer.gpudata, indices.astype(np.uint32))
    height = np.int32(depth_img.shape[0])
    width = np.int32(depth_img.shape[1])
    num = height*width
    stride = np.int32(max((num+1)//N_THREADS, 2))
    _create_mesh_triangles_from_depth_image_with_mask(
        img_buffer, 
        tris_buffer, 
        height, 
        width, 
        mask_buffer, 
        index_buffer, 
        np.uint32(capture_index),
        np.uint32(num),
        block=(int(height-1),1,1)
    )
    context.synchronize()
    if not out:
        out = np.zeros(np.prod(depth_img.shape)*6).astype(np.int32)
    cuda.memcpy_dtoh(out, tris_buffer.gpudata)
    mark_buffer_free(img_buffer_index)
    mark_buffer_free(mask_buffer_index)
    mark_buffer_free(tris_buffer_index)
    mark_buffer_free(index_buffer_index)
    return out


def calculate_invalid_mesh_uv_indices(zero_triangle_indices, out=None):
    zero_tris_buffer, zero_tris_buffer_index = get_buffer_for_image(zero_triangle_indices.shape, np.int32)
    invalid_uvs_buffer, invalid_uvs_buffer_index = get_buffer_for_image(len(zero_triangle_indices)*3, np.int32)
    cuda.memcpy_htod(zero_tris_buffer.gpudata, zero_triangle_indices.astype(np.int32))
    num = len(zero_triangle_indices)
    stride = np.int32(max(num+1//1024, 2))
    _calculate_invalid_mesh_uv_indices(zero_tris_buffer, invalid_uvs_buffer, stride, np.int32(num), block=(min(1024, num), 1, 1))
    if not out:
        out = np.zeros(np.prod(zero_triangle_indices.shape)*3).astype(np.int32)
    cuda.memcpy_dtoh(out, invalid_uvs_buffer.gpudata)
    return out


def create_mesh_uvs_from_height_and_width_with_mask(height, width, mask, out = None):
    tri_uv_buffer, tri_uv_buffer_index = get_buffer_for_image((height*width*12, ), np.float32)
    mask_buffer, mask_buffer_index = get_buffer_for_image(mask.shape, np.bool8)
    cuda.memcpy_htod(mask_buffer.gpudata, mask.astype(np.bool8))
    height = np.int32(height)
    width = np.int32(width)
    _create_mesh_uvs_from_height_and_width_with_mask(tri_uv_buffer, height, width, mask_buffer, block=(int(height-1),1,1))
    if not out:
        out = np.zeros(height*width*12).astype(np.float32)
    cuda.memcpy_dtoh(out, tri_uv_buffer.gpudata)
    mark_buffer_free(tri_uv_buffer_index)
    mark_buffer_free(mask_buffer_index)
    return out

def create_mesh_uvs_from_height_and_width(height, width, out = None):
    tri_uv_buffer, tri_uv_buffer_index = get_buffer_for_image((height*width*12, ), np.float32)
    height = np.int32(height)
    width = np.int32(width)
    num = height*width
    stride = np.int32(max((num+1)//N_THREADS, 2))
    _create_mesh_uvs_from_height_and_width_big(
        tri_uv_buffer, 
        height, 
        width, 
        stride,
        num,
        block=(min(N_THREADS, num),1,1)
    )
    if not out:
        out = np.zeros(height*width*12).astype(np.float32)
    cuda.memcpy_dtoh(out, tri_uv_buffer.gpudata)
    mark_buffer_free(tri_uv_buffer_index)
    return out

def create_vertex_uvs_from_height_and_width(height, width, out = None):
    tri_uv_buffer, tri_uv_buffer_index = get_buffer_for_image((height*width*2, ), np.float32)
    height = np.int32(height)
    width = np.int32(width)
    num = height*width
    stride = np.int32(max((num+1)//N_THREADS, 2))
    _create_vertex_uvs_from_height_and_width_big(
        tri_uv_buffer, 
        height, 
        width, 
        stride,
        num,
        block=(min(N_THREADS, num),1,1)
    )
    if not out:
        out = np.zeros(height*width*2).astype(np.float32)
    cuda.memcpy_dtoh(out, tri_uv_buffer.gpudata)
    mark_buffer_free(tri_uv_buffer_index)
    return out

def create_point_cloud(rgb_img, depth_img, cam_to_world):
    '''
    Create and return a colored open3d point cloud from 
    color and depth images
    '''
    points = create_vertices_from_depth_image(depth_img, cam_to_world)
    colors = o3d.utility.Vector3dVector(rgb_img.reshape(-1, 3))
    pc = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(
            points.astype(np.float64).reshape(-1, 3)
        )
    )
    pc.colors = colors
    return pc

def downsample_image(img, cam_to_world, downsample_n):
    '''
    Downsample an image, return the downsampleed image and the new
    cam to world matrix
    '''
    idn = downsample_image_gpu(img, cam_to_world, downsample_n)
    c_2_w = np.copy(cam_to_world)
    scale = np.identity(3)
    for i in range(2):
        c_2_w[i, i] *= downsample_n
    return idn.reshape(img.shape[0]//downsample_n, img.shape[1]//downsample_n), c_2_w

def downsample_color_image(img, cam_to_world, downsample_n):
    '''
    Downsample an image, return the downsampleed image and the new
    cam to world matrix
    '''
    idn = downsample_color_image_gpu(img, cam_to_world, downsample_n)
    c_2_w = np.copy(cam_to_world)
    for i in range(3):
        c_2_w[i, i] *= downsample_n
    return idn.reshape(img.shape[0]//downsample_n, img.shape[1]//downsample_n, 3).astype(np.float32), c_2_w

def _get_tex(rgb_img):
    return o3d.geometry.Image(
        (rgb_img*255).astype(np.uint8)
    )

def create_mesh(rgb_img, depth_img, intrinsic_params):
    '''
    Create and return an open3d mesh from color
    and depth images
    '''
    pc = create_point_cloud(rgb_img, depth_img, intrinsic_params)
    print('pc')
    tri_uvs = create_mesh_uvs_from_height_and_width(depth_img.shape[0], depth_img.shape[1])
    print('uvs')
    tris = create_mesh_triangles_from_depth_image(depth_img)
    print('tris')

    mesh = o3d.geometry.TriangleMesh(
        pc.points,
        o3d.utility.Vector3iVector(
            tris.reshape(-1, 3).astype(np.int32)
        ),
    )
    print('mesh')
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        tri_uvs.reshape(-1, 2).astype(np.float64)
    )
    print('uvs')
    mesh.textures = [_get_tex(rgb_img)]
    print('tex')
    mesh.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(
            np.prod(depth_img.shape)*2, 
            np.int32
        )
    )
    print('ids')
    return mesh

def add_triangles_from_capture_2(
        capture, 
        existing_mesh, 
        existing_captures, 
        maps, 
        maxxoffset=0,
        minxoffset=0,
        maxyoffset=0,
        minyoffset=0,
    ):
    # type: Capture, o3d.geometry.Mesh, List[Capture], List[np.array(4, 4)]
    # maps[i]: capture.point_cloud -> existing_captures[i].point_cloud
    pts1 = np.asarray(existing_mesh[0]).reshape(-1, 3)
    pts2 = np.asarray(capture.get_points()).reshape(-1, 3)

    tris_1 = np.asarray(existing_mesh[1]).reshape(-1, 3)

    overlaps = np.zeros(capture.depth.shape, np.uint32).flatten()
    indices = np.zeros(capture.depth.shape, np.uint32).flatten()
    for i, c in enumerate(existing_captures):
        overlaps, indices = compute_overlaps(
            c.depth, 
            capture.depth, 
            np.linalg.inv(c.cam_to_world),
            capture.cam_to_world, 
            maps[i],
            i+1,
            overlaps,
            indices,
        )

    tris_2 = create_mesh_triangles_from_depth_image_with_mask(capture.depth, overlaps, indices, pts1.shape[0]).reshape(-1, 3)
    zero_triangles = np.argwhere(np.all(tris_2==0, axis=1)==True)
    zero_tri_uvs = calculate_invalid_mesh_uv_indices(zero_triangles)
    
    tris = np.concatenate(
        (
            tris_1.reshape(-1, 3),
            np.delete(tris_2, zero_triangles, axis=0) + pts1.shape[0]
        )
    )
    pts = np.concatenate((pts1, pts2))
    '''
    tri_uvs = np.concatenate(
        (
            np.asarray(existing_mesh[3]).reshape(-1, 2),
            np.delete(np.asarray(capture.get_mesh().triangle_uvs),zero_tri_uvs,axis=0)
        )
    )
    '''
    tri_uvs = np.concatenate(
        (
            np.asarray(existing_mesh[3]).reshape(-1, 2),
            capture.get_vertex_uvs().reshape(-1, 2)
        )
    ).astype(np.float32)
    textures = existing_mesh[2] + [capture.get_texture()]
    '''
    t1 = np.zeros(existing_mesh[2][0].shape)
    t1[:,:] = np.array([1, 0, 0])
    t2 = np.zeros(capture.get_texture().shape)
    t2[:,:] = np.array([0, 1, 0])
    textures = [t1, t2]
    '''
    '''
    triangle_material_ids = np.concatenate(
        (
            np.asarray(existing_mesh[4]),
            np.repeat(np.delete(np.asarray(capture.get_mesh().triangle_material_ids), zero_triangles, axis=0) + len(existing_mesh[2]), 3)
        )
    )
    '''
    triangle_material_ids = np.concatenate(
        (
            np.asarray(existing_mesh[4]),
            np.ones(len(pts2)).astype(np.uint32) + len(existing_mesh[2])
        )
    )
    return (pts, tris, textures, tri_uvs, triangle_material_ids) 

def add_triangles_from_capture(
        capture, 
        existing_mesh, 
        existing_captures, 
        maps, 
        maxxoffset=0,
        minxoffset=0,
        maxyoffset=0,
        minyoffset=0,
    ):
    # type: Capture, o3d.geometry.Mesh, List[Capture], List[np.array(4, 4)]
    # maps[i]: capture.point_cloud -> existing_captures[i].point_cloud
    pts1 = np.asarray(existing_mesh.vertices)
    pts2 = np.asarray(capture.get_mesh().vertices)

    tris_1 = np.asarray(existing_mesh.triangles)

    overlaps = np.zeros(capture.depth.shape, np.uint32).flatten()
    indices = np.zeros(capture.depth.shape, np.uint32).flatten()
    for i, c in enumerate(existing_captures):
        overlaps, indices = compute_overlaps(
            c.depth, 
            capture.depth, 
            np.linalg.inv(c.cam_to_world),
            capture.cam_to_world, 
            maps[i],
            i+1,
            overlaps,
            indices,
        )

    tris_2 = create_mesh_triangles_from_depth_image_with_mask(capture.depth, overlaps, indices, pts1.shape[0]).reshape(-1, 3)
    zero_triangles = np.argwhere(np.all(tris_2==0, axis=1)==True)
    zero_tri_uvs = calculate_invalid_mesh_uv_indices(zero_triangles)
    print(len(zero_triangles))
    tris = np.concatenate(
        (
            tris_1.reshape(-1, 3),
            np.delete(tris_2, zero_triangles, axis=0) + pts1.shape[0]
        )
    )

    pts = np.concatenate((pts1, pts2))
    
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            pts
        ),
        o3d.utility.Vector3iVector(
            tris
        ),
    )
    meshh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            pts2
        ),
        o3d.utility.Vector3iVector(
            tris_2
        ),
    )
    tri_uvs = np.concatenate(
        (
            np.asarray(existing_mesh.triangle_uvs),
            np.delete(np.asarray(capture.get_mesh().triangle_uvs),zero_tri_uvs,axis=0)
        )
    )
    textures = existing_mesh.textures + capture.get_mesh().textures

    triangle_material_ids = np.concatenate(
        (
            np.asarray(existing_mesh.triangle_material_ids),
            np.delete(np.asarray(capture.get_mesh().triangle_material_ids), zero_triangles, axis=0) + len(existing_mesh.textures)
        )
    )
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        tri_uvs
    )
    mesh.textures = textures
    mesh.triangle_material_ids = o3d.utility.IntVector(
        triangle_material_ids
    )
    return mesh
