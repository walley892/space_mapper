import pycuda.autoinit
from pycuda.compiler import SourceModule
gpu_mod = SourceModule("""
__device__ double3 matmul(double* matrix, double3 pt){
    return {
        matrix[0]*pt.x + matrix[1]*pt.y + matrix[2]*pt.z,
        matrix[3]*pt.x + matrix[4]*pt.y + matrix[5]*pt.z,
        matrix[6]*pt.x + matrix[7]*pt.y + matrix[8]*pt.z,
    };
}
__device__ double4 matmul4(double* matrix, double4 pt){
    return {
        matrix[0]*pt.x + matrix[1]*pt.y + matrix[2]*pt.z + matrix[3]*pt.w,
        matrix[4]*pt.x + matrix[5]*pt.y + matrix[6]*pt.z + matrix[7]*pt.w,
        matrix[8]*pt.x + matrix[9]*pt.y + matrix[10]*pt.z + matrix[11]*pt.w,
        matrix[12]*pt.x + matrix[13]*pt.y + matrix[14]*pt.z + matrix[15]*pt.w,
    };
}
__device__ int2 im_2_down(int2 coord, int downsample_n){
    return {coord.x/downsample_n, coord.y*downsample_n};
}

__device__ int2 down_2_im(int2 coord, int downsample_n){
    return {coord.x*downsample_n, coord.y*downsample_n};
}

__device__ int coord_to_index(int2 coord, int width){
    return coord.x*width + coord.y;
}

__device__ int2 index_to_coord(int index, int width){
    return {index/width, index%width};
}

__device__ int2 floor2(double2 p){
    int x = floor(p.x);
    int y = floor(p.y);
    return {x, y};
}


__device__ int2 round2(double2 p){
    int x = round(p.x);
    int y = round(p.y);
    return {x, y};
}

__device__ bool not_too_far(double one, double two, double three){
    if(abs(one - two) > 0.05){
        return false;
    }
    if(abs(one - three) > 0.05){
        return false;
    }
    if(abs(two - three) > 0.05){
        return false;
    }
    return true;
}

__global__ void create_vertices_from_depth_image_2(double* depth_img, double* out, double* cam_to_world, int width, int height, int xoffset, int yoffset){
    int i = threadIdx.x; 
    for(int j = 0; j < width; ++j){
        int index = coord_to_index({i, j}, width);
        double3 pt = {0, 0, 0};
        if(depth_img[index] != 0){
            pt = matmul(cam_to_world, {(double)(j+yoffset), (double)(i+xoffset), 1});
        }
        out[index*3] = pt.x*depth_img[index];
        out[index*3+1] = pt.y*depth_img[index];
        out[index*3+2] = depth_img[index];
    }
}

__global__ void create_vertices_from_depth_image_big(double* depth_img, double* out, double* cam_to_world, int width, int stride, int num, int xoffset, int yoffset){
    int tIdx = threadIdx.x + blockDim.x*blockIdx.x; 
    for(int index = tIdx*stride; index < (tIdx+1)*stride && index < num; index++){
        int2 ij = index_to_coord(index, width); 
        double3 pt = {0, 0, 0};
        if(depth_img[index] != 0){
            pt = matmul(cam_to_world, {(double)(ij.y+yoffset), (double)(ij.x+xoffset), 1});
        }
        out[index*3] = pt.x*depth_img[index];
        out[index*3+1] = pt.y*depth_img[index];
        out[index*3+2] = depth_img[index];
    }
}

__global__ void downsample_image_big(double* img, double* out, int downsample_n, int width, int stride, int num){
    int tIdx = threadIdx.x + blockDim.x*blockIdx.x; 
    int downsample_width = width/downsample_n;
    for(int index_dn = tIdx*stride; index_dn < (tIdx+1)*stride && index_dn < num; ++index_dn){
        int index = coord_to_index(
            down_2_im(
                index_to_coord(index_dn, downsample_width),
                downsample_n
            ), 
            width
        );
        out[index_dn] = img[index];
    }
}

__global__ void downsample_color_image_big(double* img, double* out, int downsample_n, int width, int stride, int num){
    int tIdx = threadIdx.x + blockDim.x*blockIdx.x; 
    int downsample_width = width/downsample_n;
    for(int index_dn = tIdx*stride; index_dn < (tIdx+1)*stride && index_dn < num; ++index_dn){
        int index = coord_to_index(
            down_2_im(
                index_to_coord(index_dn, downsample_width),
                downsample_n
            ), 
            width
        );
        out[index_dn*3] = img[index*3];
        out[index_dn*3 + 1] = img[index*3 + 1];
        out[index_dn*3+ 2] = img[index*3+ 2];
    }
}

__global__ void create_mesh_triangles_from_depth_image_big(double* depth_img, int* out, int width, int height,int stride, int num){
    int tIdx = threadIdx.x; 
    for(int index = tIdx*stride; index < (tIdx+1)*stride && index < num; ++index){
        int2 ij = index_to_coord(index, width); 
        int i = ij.x;
        int j = ij.y;
        
        if(i <= 0 || i >= height -1 || j <= 0 || j >= width - 1){
            out[index*6] = 0;
            out[index*6+1] = 0;
            out[index*6+2] = 0;
            out[index*6+3] = 0;
            out[index*6+4] = 0;
            out[index*6+5] = 0;
            continue;
        }

        int n1 = coord_to_index({i + 1, j}, width);
        int n2 = coord_to_index({i+1,j+1}, width);
        
        int n4 = coord_to_index({i,j+1}, width);
        int n3 = coord_to_index({i+1,j+1}, width);

        if(!(depth_img[index] == 0 || depth_img[n1] == 0 || depth_img[n2] == 0)
            && not_too_far(depth_img[index], depth_img[n1], depth_img[n2])
        ){
            out[index*6] = index;
            out[index*6+1] = n1;
            out[index*6+2] = n2;
        }else{
            out[index*6] = 0;
            out[index*6+1] = 0;
            out[index*6+2] = 0;
        }
        if(!(depth_img[index] == 0 || depth_img[n3] == 0 || depth_img[n4] == 0)
            && not_too_far(depth_img[index], depth_img[n3], depth_img[n4])
        ){
            out[index*6+3] = index;
            out[index*6+4] = n3;
            out[index*6+5] = n4;
        }else{
            out[index*6+3] = 0;
            out[index*6+4] = 0;
            out[index*6+5] = 0;
        }
    }
}

__global__ void create_mesh_triangles_from_depth_image_with_mask_big(double* depth_img, int* out, int width, int stride, int num, bool* mask){
    int tIdx = threadIdx.x; 
    for(int index = tIdx*stride; index < (tIdx+1)*stride && index < num; ++index){
        int2 ij = index_to_coord(index, width); 
        int i = ij.x;
        int j = ij.y;
     
        int n1 = coord_to_index({i + 1, j}, width);
        int n2 = coord_to_index({i+1,j+1}, width);
        
        int n4 = coord_to_index({i,j+1}, width);
        int n3 = coord_to_index({i+1,j+1}, width);
        
        if(!(depth_img[index] == 0 || depth_img[n1] == 0 || depth_img[n2] == 0) 
            && !mask[index] && !mask[n1] && !mask[n2]
            && not_too_far(depth_img[index], depth_img[n1], depth_img[n2])
            ){
            out[index*6] = index;
            out[index*6+1] = n1;
            out[index*6+2] = n2;
            
        }else{
            out[index*6] = 0;
            out[index*6+1] = 0;
            out[index*6+2] = 0;
        }
        if(!(depth_img[index] == 0 || depth_img[n3] == 0 || depth_img[n4] == 0) 
            && !mask[index] && !mask[n3] && !mask[n4]
            && not_too_far(depth_img[index], depth_img[n3], depth_img[n4])
            ){
            out[index*6+3] = index;
            out[index*6+4] = n3;
            out[index*6+5] = n4;
        }else{
            out[index*6+3] = 0;
            out[index*6+4] = 0;
            out[index*6+5] = 0;
        }
    }
}

__global__ void create_mesh_uvs_from_height_and_width_big(float* out, int height, int width, int stride, int num){
    int tIdx = threadIdx.x; 
    for(int index = tIdx*stride; index < (tIdx+1)*stride && index < num; ++index){
        int2 ij = index_to_coord(index, width); 
        int i = ij.x;
        int j = ij.y;

        out[index*12+1] = i/(float)height;
        out[index*12] = j/(float)width;

        out[index*12+2] = j/(float)width;
        out[index*12+3] = (i+1)/(float)height;
        
        out[index*12+4] = (j+1)/(float)width;
        out[index*12+5] = (i+1)/(float)height;
 
        out[index*12+7] = i/(float)height;
        out[index*12+6] = j/(float)width;
       
        out[index*12+10] = (j+1)/(float)width;
        out[index*12+11] = i/(float)height;
        
        out[index*12+8] = (j+1)/(float)width;
        out[index*12+9] = (i+1)/(float)height;
    }
}


__global__ void create_vertex_uvs_from_height_and_width_big(float* out, int height, int width, int stride, int num){
    int tIdx = threadIdx.x; 
    for(int index = tIdx*stride; index < (tIdx+1)*stride && index < num; ++index){
        int2 ij = index_to_coord(index, width); 
        int i = ij.x;
        int j = ij.y;

        out[index*2+1] = i/(float)height;
        out[index*2] = j/(float)width;
    }
}

__global__ void compute_overlaps_big(double* img_1, double* img_2, double* world_to_one, double* two_to_world, double* two_to_one, int width_1, int width_2, int height_1, int stride, int num, bool* out){
    int tIdx = threadIdx.x; 
    for(int index2 = tIdx*stride; index2 < (tIdx+1)*stride && index2 < num; ++index2){
        int2 pt2 = index_to_coord(index2, width_2); 
        if(img_2[index2] == 0){
            continue;
        }

        double3 pt2_world = matmul(two_to_world, {(double)pt2.y, (double)pt2.x, 1.0});
        pt2_world = {
            pt2_world.x*img_2[index2],
            pt2_world.y*img_2[index2],
            img_2[index2],
        };
        double4 pt21_world = matmul4(two_to_one, {pt2_world.x, pt2_world.y, pt2_world.z, 1.0});

        double3 pt21 = matmul(world_to_one, {pt21_world.x/pt21_world.z, pt21_world.y/pt21_world.z, pt21_world.z/img_2[index2]});
        if(pt21.y > 0 && pt21.y < height_1 - 12 && pt21.x > 0 && pt21.x < width_1){
            int index1 = coord_to_index(round2({pt21.y, pt21.x}), width_1);
            //if(img_1[index1] != 0){
                out[index2] = true;
            //}
        }
    }
}


__global__ void downsample_image(double* img, double* out, int downsample_n, int width){
    int idn = threadIdx.x; 
    int downsample_width = width/downsample_n;
    for(int jdn = 0; jdn < downsample_width; ++jdn){
        int index = coord_to_index(down_2_im({idn, jdn}, downsample_n), width);
        int index_dn = coord_to_index({idn, jdn}, downsample_width);
        out[index_dn] = img[index];
    }
}

__global__ void downsample_color_image(double* img, double* out, int downsample_n, int width){
    int idn = threadIdx.x; 
    int downsample_width = width/downsample_n;
    for(int jdn = 0; jdn < downsample_width; ++jdn){
        int index = coord_to_index(down_2_im({idn, jdn}, downsample_n), width);
        int index_dn = coord_to_index({idn, jdn}, downsample_width);
        out[index_dn*3] = img[index*3];
        out[index_dn*3 + 1] = img[index*3 + 1];
        out[index_dn*3+ 2] = img[index*3+ 2];
    }
}


__global__ void create_vertices_from_depth_image(double* depth_img, double* out, float fx, float fy, float cx, float cy, int height, int width){
    int i = threadIdx.x; 
    for(int j = 0; j < width; ++j){
        int index = coord_to_index({i, j}, width);
        float z = depth_img[index];
        float x = (j-cx)*(z/fx);
        float y = (i-cy)*(z/fy);
        out[index*3] = x;
        out[index*3+1] = y;
        out[index*3+2] = z;
    }
}
__global__ void compute_overlaps(double* img_1, double* img_2, double* world_to_one, double* two_to_world, double* two_to_one, int width_1, int width_2, int height_1, uint* out, uint* index_out,  uint capture_index){
    int i = threadIdx.x;
    for(int j = 0; j < width_2; ++j){
        int2 pt2 = {i, j};
        int index2 = coord_to_index(pt2, width_2);
        if(img_2[index2] == 0){
            continue;
        }

        double3 pt2_world = matmul(two_to_world, {(double)pt2.y, (double)pt2.x, 1.0});
        pt2_world = {
            pt2_world.x*img_2[index2],
            pt2_world.y*img_2[index2],
            img_2[index2],
        };
        double4 pt21_world = matmul4(two_to_one, {pt2_world.x, pt2_world.y, pt2_world.z, 1.0});

        double3 pt21 = matmul(world_to_one, {pt21_world.x/pt21_world.z, pt21_world.y/pt21_world.z, 1});
        if(pt21.y >  0 && pt21.y < height_1 && pt21.x > 0 && pt21.x < width_1){
            int index1 = coord_to_index(floor2({pt21.y, pt21.x}), width_1);
            if(img_1[index1] != 0){
                if(abs(img_1[index1] - pt21_world.z) < 0.05){
                    if(out[index2] == 0 && index_out[index2] == 0){
                        out[index2] = (uint)capture_index;
                        index_out[index2] = (uint) index1;
                    }
                }
            }
        }
    }
}

__global__ void create_mesh_triangles_from_depth_image_with_mask(double* depth_img, int* out, int height, int width, uint* mask, uint* indices, uint n_captures, uint n_per_capture){
    int i = threadIdx.x;

    for(int j = 0; j < width-1; ++j){
        int index = coord_to_index({i, j}, width);
        int n1 = coord_to_index({i + 1, j}, width);
        int n2 = coord_to_index({i+1,j+1}, width);
        
        int n4 = coord_to_index({i,j+1}, width);
        int n3 = coord_to_index({i+1,j+1}, width);
        
        if (not_too_far(depth_img[index], depth_img[n1], depth_img[n2])){

        if(!(depth_img[index] == 0 || depth_img[n1] == 0 || depth_img[n2] == 0) 
            && (mask[index] == 0 && mask[n1] ==0 && mask[n2] == 0))
            {
            out[index*6] = index;
            out[index*6+1] = n1;
            out[index*6+2] = n2;
            
        }else{
            if(mask[index] == 0 && mask[n1] != 0 && mask[n2] != 0  && depth_img[index] != 0){
                out[index*6] = index;
                out[index*6+1] = (((mask[n1]-1)*n_per_capture)+indices[n1]) - n_captures;
                out[index*6+2] = (((mask[n2]-1)*n_per_capture)+indices[n2]) - n_captures;
                
             }
            else if(mask[index] == 0 && mask[n1] == 0 && mask[n2] != 0 && depth_img[index] != 0 && depth_img[n1] != 0){
                out[index*6] = index;
                out[index*6+1] = n1;
                out[index*6+2] = (((mask[n2]-1)*n_per_capture)+indices[n2]) - n_captures;
            }
            else if(mask[index] == 0 && mask[n2] == 0 && mask[n1] != 0 && depth_img[index] != 0 && depth_img[n2] != 0){
                out[index*6] = index;
                out[index*6+1] = (((mask[n1]-1)*n_per_capture)+indices[n1]) - n_captures;
                out[index*6+2] = n2;
            }
            else if(mask[index] != 0 && mask[n1] == 0 && mask[n2] == 0 && depth_img[n1] != 0 && depth_img[n2] != 0){
                out[index*6] = (((mask[index]-1)*n_per_capture)+indices[index]) - n_captures;
                out[index*6+1] = n1;
                out[index*6+2] = n2;
            }
            else if(mask[index] != 0 && mask[n1] != 0 && mask[n2] == 0  && depth_img[n2] != 0){
                out[index*6] = (((mask[index]-1)*n_per_capture)+indices[index]) - n_captures;
                out[index*6+1] = (((mask[n1]-1)*n_per_capture)+indices[n1]) - n_captures;
                out[index*6+2] = n2;
            }
            else if(mask[index] != 0 && mask[n2] != 0 && mask[n1] == 0 && depth_img[n1] != 0){
                out[index*6] = (((mask[index]-1)*n_per_capture)+indices[index]) - n_captures;
                out[index*6+1] = n1;
                out[index*6+2] = (((mask[n2]-1)*n_per_capture)+indices[n2]) - n_captures;
            }
            else{
                out[index*6] = 0;
                out[index*6+1] = 0;
                out[index*6+2] = 0;
            }
        }
}
        else{
                out[index*6] = 0;
                out[index*6+1] = 0;
                out[index*6+2] = 0;

        }

        if (not_too_far(depth_img[index], depth_img[n3], depth_img[n4])){

        if(!(depth_img[index] == 0 || depth_img[n3] == 0 || depth_img[n4] == 0) 
            && (mask[index] == 0 && mask[n3] ==0 && mask[n4] == 0)
            ){
            out[index*6+3] = index;
            out[index*6+4] = n3;
            out[index*6+5] = n4;
        }else{
            if(mask[index] == 0 && mask[n3] != 0 && mask[n4] != 0  && depth_img[index] != 0){
                
                out[index*6+3] = index;
                out[index*6+4] = (((mask[n3]-1)*n_per_capture)+indices[n3]) - n_captures;
               out[index*6+5] = (((mask[n4]-1)*n_per_capture)+indices[n4]) - n_captures;
                
            }
            else if(mask[index] == 0 && mask[n3] == 0 && mask[n4] != 0 && depth_img[index] != 0 && depth_img[n3] != 0){
                out[index*6+3] = index;
                out[index*6+4] = n3;
                out[index*6+5] = (((mask[n4]-1)*n_per_capture)+indices[n4]) - n_captures;
            }
            else if(mask[index] == 0 && mask[n4] == 0 && mask[n3] != 0 && depth_img[index] != 0 && depth_img[n4] != 0){
                out[index*6+3] = index;
                out[index*6+4] = (((mask[n3]-1)*n_per_capture)+indices[n3]) - n_captures;
                out[index*6+5] = n4;
            }
            
            else if(mask[index] != 0 && mask[n3] == 0 && mask[n4] == 0 && depth_img[n3] != 0 && depth_img[n4] != 0){
                out[index*6+3] = (((mask[index]-1)*n_per_capture)+indices[index]) - n_captures;
                out[index*6+4] = n3;
                out[index*6+5] = n4;
            }
            else if(mask[index] != 0 && mask[n3] != 0 && mask[n4] == 0  && depth_img[n4] != 0){
                out[index*6+3] = (((mask[index]-1)*n_per_capture)+indices[index]) - n_captures;
                out[index*6+4] = (((mask[n3]-1)*n_per_capture)+indices[n3]) - n_captures;
                out[index*6+5] = n4;
            }
            else if(mask[index] != 0 && mask[n4] != 0 && mask[n3] == 0 && depth_img[n3] != 0){
                out[index*6+3] = (((mask[index]-1)*n_per_capture)+indices[index]) - n_captures;
                out[index*6+4] = n3;
                out[index*6+5] = (((mask[n4]-1)*n_per_capture)+indices[n4]) - n_captures;
            }
            else{
                out[index*6+3] = 0;
                out[index*6+4] = 0;
                out[index*6+5] = 0;
            }
        }
        }
        else{
                out[index*6+3] = 0;
                out[index*6+4] = 0;
                out[index*6+5] = 0;
        }
        
    }
}

__global__ void create_mesh_triangles_from_depth_image(double* depth_img, int* out, int height, int width){
    int i = threadIdx.x;

    for(int j = 0; j < width-1; ++j){
        int index = coord_to_index({i, j}, width);
        int n1 = coord_to_index({i + 1, j}, width);
        int n2 = coord_to_index({i+1,j+1}, width);
        
        int n4 = coord_to_index({i,j+1}, width);
        int n3 = coord_to_index({i+1,j+1}, width);

        if(!(depth_img[index] == 0 || depth_img[n1] == 0 || depth_img[n2] == 0)
            && not_too_far(depth_img[index], depth_img[n1], depth_img[n2])
        ){
            out[index*6] = index;
            out[index*6+1] = n1;
            out[index*6+2] = n2;
        }else{
            out[index*6] = 0;
            out[index*6+1] = 0;
            out[index*6+2] = 0;
        }
        if(!(depth_img[index] == 0 || depth_img[n3] == 0 || depth_img[n4] == 0)
            && not_too_far(depth_img[index], depth_img[n3], depth_img[n4])
        ){
            out[index*6+3] = index;
            out[index*6+4] = n3;
            out[index*6+5] = n4;
        }else{
            out[index*6+3] = 0;
            out[index*6+4] = 0;
            out[index*6+5] = 0;
        }
    }
}


__global__ void create_mesh_uvs_from_height_and_width_with_mask(float* out, int height, int width, bool* mask){

    int i = threadIdx.x;

    for(int j = 0; j < width-1; ++j){
        int index = i * width + j;
        if(mask[index]){
            continue;
        }

        out[index*12+1] = i/(float)height;
        out[index*12] = j/(float)width;

        out[index*12+2] = j/(float)width;
        out[index*12+3] = (i+1)/(float)height;
        
        out[index*12+4] = (j+1)/(float)width;
        out[index*12+5] = (i+1)/(float)height;
 
        out[index*12+7] = i/(float)height;
        out[index*12+6] = j/(float)width;
       
        out[index*12+10] = (j+1)/(float)width;
        out[index*12+11] = i/(float)height;
        
        out[index*12+8] = (j+1)/(float)width;
        out[index*12+9] = (i+1)/(float)height;
    }
}
__global__ void calculate_invalid_mesh_uv_indices(int* zero_triangles, int* out, int stride, int num){
    int start = threadIdx.x;
    for(int idx = start*stride; idx < (start+1)*stride && idx < num; ++idx){
        int triid = zero_triangles[idx];
        out[3*idx] = 3*triid;
        out[3*idx+1] = 3*triid+1;
        out[3*idx+2] = 3*triid+2;
    }
}
__global__ void create_mesh_uvs_from_height_and_width(float* out, int height, int width){

    int i = threadIdx.x;

    for(int j = 0; j < width-1; ++j){
        int index = i * width + j;

        out[index*12+1] = i/(float)height;
        out[index*12] = j/(float)width;

        out[index*12+2] = j/(float)width;
        out[index*12+3] = (i+1)/(float)height;
        
        out[index*12+4] = (j+1)/(float)width;
        out[index*12+5] = (i+1)/(float)height;
 
        out[index*12+7] = i/(float)height;
        out[index*12+6] = j/(float)width;
       
        out[index*12+10] = (j+1)/(float)width;
        out[index*12+11] = i/(float)height;
        
        out[index*12+8] = (j+1)/(float)width;
        out[index*12+9] = (i+1)/(float)height;
    }
}

__global__ void compute_im1_coords(double* img_2, double* world_to_one, double* two_to_world, double* two_to_one, int width_1, int width_2, int height_1, int height_2, int* out){
    int i = threadIdx.x;
    for(int j = 0; j < width_2; ++j){
        int2 pt2 = {i, j};
        int index2 = coord_to_index(pt2, width_2);
        out[index2*2] = 0;
        out[index2*2+1] = 0;
        if(img_2[index2] == 0){
            continue;
        }

        double3 pt2_world = matmul(two_to_world, {(double)pt2.y, (double)pt2.x, 1.0});
        pt2_world = {
            pt2_world.x*img_2[index2],
            pt2_world.y*img_2[index2],
            img_2[index2],
        };
        double4 pt21_world = matmul4(two_to_one, {pt2_world.x, pt2_world.y, pt2_world.z, 1.0});

        double3 pt21 = matmul(world_to_one, {pt21_world.x/pt21_world.z, pt21_world.y/pt21_world.z, 1});
        int2 pt21im1 = round2({pt21.y, pt21.x});
        out[index2*2] = pt21im1.x;
        out[index2*2 + 1] = pt21im1.y;
    }
}

__global__ void copy_to_im1_frame(double* img_2, double* world_to_one, double* two_to_world, double* two_to_one, int width_1, int width_2, int width_big, int height_1, int height_2, int xoffset, int yoffset, double* out){
    int i = threadIdx.x;
    for(int j = 0; j < width_2; ++j){
        int2 pt2 = {i, j};
        int index2 = coord_to_index(pt2, width_2);
        if(img_2[index2] == 0){
            continue;
        }

        double3 pt2_world = matmul(two_to_world, {(double)pt2.y, (double)pt2.x, 1.0});
        pt2_world = {
            pt2_world.x*img_2[index2],
            pt2_world.y*img_2[index2],
            img_2[index2],
        };
        double4 pt21_world = matmul4(two_to_one, {pt2_world.x, pt2_world.y, pt2_world.z, 1.0});

        double3 pt21 = matmul(world_to_one, {pt21_world.x/pt21_world.z, pt21_world.y/pt21_world.z, 1.0});
        int2 pt21im1 = round2({pt21.y, pt21.x});
        int index_big = coord_to_index({pt21im1.x + xoffset, pt21im1.y + yoffset}, width_big);
        if(out[index_big] == 0){
            out[index_big] = pt21_world.z;
        }
    }
}
""")


