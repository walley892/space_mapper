from copy import deepcopy
import numpy as np
import gc
import open3d as o3d
_params = {
        'voxel_dn_amt_global': 0.05,
        'voxel_dn_amt_local' : 0.002,
        'registration_type_global': 'point',
        'registration_type_local': 'point',
}
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    nz = np.nonzero(np.sum(pcd_down.points, axis=1))

    pcd_down = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(
            np.asarray(pcd_down.points)[nz]
        )
    )

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size*0.7
    '''
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    '''
    reg_type = _params['registration_type_global']
    reg = o3d.pipelines.registration.TransformationEstimationPointToPlane() if reg_type=='plane' else o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        reg,
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.99),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ]
        #[]
        , o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 1000))
    return result


def transform_local(p1, p2, size, reg_type, trans_init, max_iteration = 100, rel_fit = 1e-07, rel_rmse = 1e-07):
    reg = o3d.pipelines.registration.TransformationEstimationPointToPlane() if reg_type=='plane' else o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
    p1dn = p1.voxel_down_sample(size)
    p2dn = p2.voxel_down_sample(size)

    p1dn.estimate_normals()
    p2dn.estimate_normals()
    
    conv = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration = max_iteration,
        relative_fitness=rel_fit,
        relative_rmse=rel_rmse
    )

    trans = o3d.pipelines.registration.registration_icp(
        p1dn,
        p2dn,
        size,
        trans_init,
        reg,
        conv
    ) 
    return trans

def transform_colored_icp(p1, p2, size, trans_init):
    p1dn = p1.voxel_down_sample(size)
    p2dn = p2.voxel_down_sample(size)

    p1dn.estimate_normals()
    p2dn.estimate_normals()

    conv = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration = 100,
        relative_fitness=1e-08,
        relative_rmse=1e-08
    )
    trans = o3d.pipelines.registration.registration_colored_icp(
        p1dn,
        p2dn,
        0.0005,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        conv
    )

    return trans




def scaling(scale):
    mat = np.identity(4)
    for i in range(3):
        mat[i,i] = scale
    return mat

def transform_pcs_best(pc1_big, pc2_big): 
    pc1_big = deepcopy(pc1_big)
    pc2_big = deepcopy(pc2_big)
    t_12_rough = rough_transformation(pc1_big, pc2_big, 0.1)
    pc1_big.transform(scaling(10))
    pc2_big.transform(scaling(10))
    if t_12_rough is None:
        return None
    print('rough rmse, fitness: {}, {}'.format(t_12_rough.inlier_rmse, t_12_rough.fitness))
    t_12_refined = transform_local(pc1_big, pc2_big, 0.1, 'point', t_12_rough.transformation, 2000, 1e-08, 1e-08)
    print('refined rmse, fitness: {}, {}'.format(t_12_refined.inlier_rmse, t_12_refined.fitness))
    return np.matmul(np.matmul(np.linalg.inv(scaling(10)),t_12_refined.transformation), scaling(10))
    

def rough_capture_transformation(c1, c2):
    return rough_transformation(c1.get_point_cloud(), c2.get_point_cloud(), 0.1).transformation


def better_rough_transformation(p1, p2):
    best_trans = None
    for i in range(5):
        trans = rough_transformation(p1, p2, 1.5 + (0.1*i))
        if trans.inlier_rmse > 0.0001:
            if best_trans is None or best_trans.inlier_rmse > trans.inlier_rmse:
                best_trans = trans
    return best_trans

def rough_transformation(p1, p2, size):
    pc1d, pc1f = preprocess_point_cloud(p1, size)
    pc2d, pc2f = preprocess_point_cloud(p2, size)

    trans_init = execute_global_registration(pc1d, pc2d, pc1f, pc2f, size)
    del pc1d
    del pc2d
    gc.collect()
    return trans_init 


