# MultiSlice Electron Tomography (MSET) Package

This package is implemented for 4D-STEM based electron tomography.

For the detail, please see the paper: J. Lee, M. Lee, Y.K. Park, C. Ophus, and Y. Yang, Multislice Electron Tomography Using Four-dimensional Scanning Electron Microscopy, *Phyiscal Review Applied* (accepted), arXiv:2210.12636.

This reconstruction require much computaional resource.
Please check the GPU memory and computation time if you want to increase volume size.

### Requirements
- matlab (>= 2018a)
- CUDA (11.4 was tested)

### How to start
run ./main.m

### Compile for CUDA version MSET
run ./src/CUDA/compile.m in matlab environment


# further developing..
- probe optimization
- scan position optimization
not fully implemented. (only MSET cpu version is open.)
