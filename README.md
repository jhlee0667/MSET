# MultiSlice Electron Tomography (MSET) Package

This package is implemented for 4D-STEM based electron tomography.

#### Phase retrival algorithm using 4D-STEM tilt series dataset
- 3D Object reconstruction
- Probe optimization
- Scan position optimization

For the detail, please see the paper: J. Lee, M. Lee, Y.K. Park, C. Ophus, and Y. Yang, Multislice Electron Tomography Using Four-dimensional Scanning Electron Microscopy, *Physical Review Applied* (accepted), arXiv:2210.12636.

This reconstruction method requires much computaional resources.
Please check the GPU memory and computation time if you want to increase volume size.


### Requirements
- MATLAB (>= 2018a)
- CUDA (11.3 was tested)

### How to start
Run ./main.m in MATLAB environment.

### Compile for CUDA version MSET
Run ./src/CUDA/compile.m in MATLAB environment.


