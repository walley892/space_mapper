# Space mapper

This package allows users to create 3d meshes (.stl format) of real-world spaces using the Microsoft Azure Kinect, stitching multiple rgbd images into a single mesh.
This package also features a method to reduce the total number of triangles in the final mesh by culling redundant trianges from overlapping images.


# Features
* A custom algorithm for culling redundant triangles to create one cohesive mesh
* An intuitive UI allowing users to progressively build a 3d mesh from multiple rgbd captures
* CUDA acceleration for point cloud and mesh generation

# Installation
1. Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
2. Install my custom [game engine](https://github.com/walley892/engine)
3.
```
git clone https://github.com/walley892/engine
cd space_mapper
// possibly inside of a virtual environment if that's your thing
pip3 install -r requirements.txt
```

# Use
```
python3 space_mapper_application.py
// Once you've exported your captures
python3 stitch.py
```
