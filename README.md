Exabuilder
==========

This repo contains the sample code for the paper:

S. Zellmann and I. Wald
*"From Soup to Bricks: Fast Clustering of Fine-Grained AMR Hierarchies for
Rendering on GPUs"*, LDAV 2025.

Pre-print: https://pds.uni-koeln.de/sites/pds/szellma1/prepress.pdf

The sample code can be built with CMake and using the NVIDIA CUDA compiler
(nvcc):

```
mkdir build
cd build
CMAKE_CUDA_COMPILER=<path-to-nvcc> cmake ..
make
```

The input is a linear list of AMR cells in the format:
```
struct AMRCell
{
  int3 pos;   // in finest-level coordinates
  int  level; // smaller => finer
};
```

The output are bricks in ExaBricks format:
```
struct ExaBrick
{
  int3 lower; // position in finest-level coordinates
  int3 size;  // in voxels
  int  level;
};
```

Apps that can consume this input are:

Sample code for the paper: I. Wald, S. Zellmann, W. Usher, N. Morrical, U. Lang, V.
Pascussi, "Ray Tracing Structured AMR Data Using ExaBricks", IEEE Visualization
2020, SciVis Papers:
https://github.com/owl-project/owlExaBrick

Sample code for the paper: Stefan Zellmann, Qi Wu, Kwan-Liu Ma, Ingo Wald:
"Memory-Efficient GPU Volume Path Tracing of AMR Data Using the Dual Mesh"
Computer Graphics Forum, Vol. 42, Issue 3, 2023 (Proceedings of EuroVis 2023)
https://github.com/owl-project/owlExaStitcher
