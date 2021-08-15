import open3d as o3d
import numpy as np
from registration import *
from gpu_utils import add_triangles_from_capture

class GroupStitcher(object):
    def __init__(self):
        self.captures = []
        self.maps = []
    def make_mesh(self, downsample_n = 12):
        mesh = self.captures[0].downsample(downsample_n).get_mesh()
        for i, capture in enumerate(self.captures[1:]):
            cdn = capture.downsample(downsample_n)
            cdn.get_mesh().transform(self.pose_graph.nodes[i+1].pose)
            maps = [np.matmul(np.linalg.inv(node.pose), self.pose_graph.nodes[i+1].pose) for node in self.pose_graph.nodes[:i+1]]
            mesh = add_triangles_from_capture(
                cdn,
                mesh,
                self.captures[:i+1],
                maps,
            )
        return mesh

    def stitch(self, progress_path = None):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(odometry)
        )
        self.maps = {}

        #maps[i]: i+1 -> i
        for i, cap in enumerate(self.captures):
            for j, cap2 in enumerate(self.captures[i+1:i+3]):
                trans = None
                if progress_path is not None:
                    try:
                        trans = np.load(progress_path+'/edge_{}_{}_transformation.npy'.format(i,i+j+1))
                        print(i, j, 'loaded')
                    except:
                        trans = None
                if trans is None:
                    trans = transform_pcs_best(
                        cap.get_point_cloud(),
                        cap2.get_point_cloud()
                    )
                    cap.point_cloud = None
                    cap2.point_cloud = None
                    if trans is None:
                        print(i, i+1+j)
                        continue
                    if progress_path is not None:
                        np.save(progress_path+'/edge_{}_{}_transformation'.format(i,i+j+1), trans)
                
                self.maps[(i, i+j+1)] = trans
                p1dn = cap.get_point_cloud().voxel_down_sample(0.02)
                p2dn = cap2.get_point_cloud().voxel_down_sample(0.02)
                information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    p1dn, p2dn, 0.02,
                    trans
                )
                
                if j == 0:
                    odometry = np.matmul(odometry, np.linalg.inv(trans))
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            odometry
                        )
                    )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i,
                        i+1+j,
                        trans,
                        information_matrix,
                        uncertain=j!=0
                    )
                )
                print(i, i+1+j)
            if i == 0:
                for j, cap2 in enumerate(self.captures[-1:]):
                    trans = None
                    if progress_path is not None:
                        try:
                            trans = np.load(progress_path+'/edge_{}_{}_transformation.npy'.format(i,i+j+1))
                            print(i, j, 'loaded')
                        except:
                            trans = None
                    if trans is None:
                        trans = transform_pcs_best(
                            cap.get_point_cloud(),
                            cap2.get_point_cloud()
                        )
                        cap.point_cloud = None
                        cap2.point_cloud = None
                        if trans is None:
                            print(i, i+1+j)
                            continue
                        if progress_path is not None:
                            np.save(progress_path+'/edge_{}_{}_transformation'.format(i,i+j+1), trans)
                    
                    self.maps[(i, i+j+1)] = trans
                    p1dn = cap.get_point_cloud().voxel_down_sample(0.02)
                    p2dn = cap2.get_point_cloud().voxel_down_sample(0.02)
                    information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                        p1dn, p2dn, 0.02,
                        trans
                    )
                    
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i,
                            i+1+j,
                            trans,
                            information_matrix,
                            uncertain=True
                        )
                    )
                    print("LOOP")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.02,
            edge_prune_threshold=0.6e-7,
            reference_node=0,
            preference_loop_closure=0.3
        )
        

        conv = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        conv.max_iteration = 10000
        conv.max_iteration_lm = 1000
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            conv,
            option
        )
        for i, c in enumerate(self.captures):
            c.get_point_cloud().transform(
                pose_graph.nodes[i].pose
            )
        self.pose_graph = pose_graph
        return pose_graph
    def export(self, path):
        for i, capture in enumerate(self.captures):
            capture.export(path + 'capture_{}'.format(i))

        for i, node in enumerate(self.pose_graph.nodes):
            np.save(
                path+'node_{}_pose'.format(i),
                node.pose    
            )

        for edge in self.pose_graph.edges:
            np.save(
                path+'edge_{}_{}_transformation'.format(edge.source_node_id, edge.target_node_id),
                edge.transformation
            )
            np.save(
                path+'edge_{}_{}_information'.format(edge.source_node_id, edge.target_node_id),
                edge.information
            )
    @staticmethod
    def restore(path):
        g = GroupStitcher()
        directory = os.listdir(path)
        captures = [item for item in directory if 'cam_to_world' in item]
        nodes = [item for item in directory if 'node' in item]
        edges = sorted([item for item in directory if 'edge' in item])
        g.captures = [None for _ in captures]
        pose_graph = o3d.pipelines.registration.PoseGraph()
        nodez = [None for _ in captures]
        for capture_path in captures:
            sp = capture_path.split('_')
            c = Capture.restore(path+'/'+'{}_{}'.format(sp[0], sp[1]))
            index = int(capture_path.split('_')[1])
            g.captures[index] = c
        for node_path in sorted(nodes):
            pose = np.load(path+'/'+node_path)
            index = int(node_path.split('_')[1])
            if index < len(nodez):
                nodez[index] = o3d.pipelines.registration.PoseGraphNode(pose)
        pose_graph.nodes = o3d.pipelines.registration.PoseGraphNodeVector(nodez)
        for i in range(0,len(edges),2):
            info_path = edges[i]
            trans_path = edges[i+1]
            transformation = np.load(path+'/'+trans_path)
            information = np.load(path+'/'+info_path)
            source_node_id = int(info_path.split('_')[1])
            target_node_id = int(info_path.split('_')[2])
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_node_id,
                    target_node_id,
                    transformation,
                    uncertain=False
                )
            )
        g.pose_graph = pose_graph
        return g

def get_num_captures_in_dir(d):
    import os
    ls = set(os.listdir(d))
    ls = set([l for l in ls if 'capture_' in l])
    nums = set([int(l.split('_')[1]) for l in ls])
    return max(nums) + 1
    

if __name__ == '__main__':
    from capture import Capture
    import sys
    restore_dir = 'captures'
    if len(sys.argv) > 1:
        restore_dir = sys.argv[1]
    num_captures = get_num_captures_in_dir(restore_dir)
    g = GroupStitcher()
    g.captures = [Capture.restore('{}/capture_{}'.format(restore_dir, i)) for i in range(num_captures)]
    g.stitch()
    m = g.make_mesh(1)
