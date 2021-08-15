import numpy as np

def homogeneous_transform(pts, transform):
    new_points = pts.reshape(-1, 3)
    new_points_homogeneous = np.ones((new_points.shape[0], 4))
    new_points_homogeneous[:,:3] = new_points
    new_points_transformed = np.matmul(new_points_homogeneous, transform.transpose())
    return new_points_transformed[:,:3].flatten()
