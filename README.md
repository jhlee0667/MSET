# MultiSlice Electron Tomography (MSET) Package

This package is implemented for 4D-STEM based electron tomography.

#### Phase retrieval algorithm using 4D-STEM tilt series dataset
- 3D object reconstruction
- Probe wavefunction optimization
- Scan position optimization

For details, please see the paper: J. Lee, M. Lee, Y.K. Park, C. Ophus, and Y. Yang, Multislice Electron Tomography Using Four-dimensional Scanning Electron Microscopy, Phys. Rev. Appl. 19, 054062 (2023).

This reconstruction method requires a lot of computational resources.
Please check the GPU memory and computation time to increase the volume size.


### Requirements
- MATLAB (>= 2018a)
- CUDA (11.3 was tested)

### How to start
Run ./main.m in MATLAB environment.

### Compile for CUDA version MSET
Run ./src/CUDA/compile.m in MATLAB environment.


