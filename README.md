# MultiSlice Electron Tomography (MSET) Package

This package is implemented for 4D-STEM based electron tomography.

For the detail, please see the paper: J. Lee, M. Lee, Y.K. Park, C. Ophus, and Y. Yang, Multislice Electron Tomography Using Four-dimensional Scanning Electron Microscopy, *Physical Review Applied* (accepted), arXiv:2210.12636.

#### Phase retrival 3D reconstruction algorithm using 4D-STEM tilt series dataset
- Object reconstruction
- probe optimization
- scan position optimization

This reconstruction method requires much computaional resources.
Please check the GPU memory and computation time if you want to increase volume size.


### Requirements
- matlab (>= 2018a)
- CUDA (11.4 was tested)

### How to start
Run ./main.m in MATLAB environment.

### Compile for CUDA version MSET
Run ./src/CUDA/compile.m in MATLAB environment.


