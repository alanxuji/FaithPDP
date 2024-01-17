# FaithPDP
Source code for the paper "Faithful Parallel Density-Peaks Clustering Using Inverse Density-Distance Condition"

## Usage:
1. Unzip the zip file to get the file folders.
2. Run FFP-DP-5Spiral-S.py to verify the effectiveness of sequential version of Algorithm FaithPDP
3. Run FFP-DP-5Spiral.py to see the speed up by using MPI parallelism of FaithPDP. Before this, make sure mpi4py is installed properly. If you want to run on a cluster, make sure filenames and path names of this project on each computing node are identical, and the cluster is properly configured w.r.t. IP, SSH, etc. A conda environment with the same name on each node is recommended.
4. For any problem with running the code, feel free to contact me at jixu AT gzu.edu.cn.
