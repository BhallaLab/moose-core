#<center>HSolve with GPU Acceleration</center>
---
####Author: Dharma teja
####Date: 14 May 2016

---

Introduction
---
This version of HSolve integrates CUDA codes to the existing HSolve library.  

Changes are made in the following files: 

* AdvanceChannel.cu [Added]
* CudaGlobal.h [Added]
* CMakeLists.txt [Modified]
* ../CMakeLists.txt [Modified]
* HinesMatrix.{cpp/h} [Modified]
* HSolveActive.{cpp/h} [Modified]
* HSolvePassive.{cpp/h} [Modified]
* HSolveActiveSetup.{cpp/h} [Modified]
* RateLookup.{cpp/h} [Modified]

To compile the library, please make a change in ../CMakeLists.txt:

line 94: `set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_xx,code=sm_xx)`

Please change "xx" to your device computing capability and architecture.

 
Breakdown of Codes
---

###CudaGlobal.h

Contain Marco definitions, bit mask definitions and functions and CUDA utility functions to be used by many of the CUDA codes. Some control variables can also be modified here.

`BLOCK_WIDTH` <br>
Determine the number of CUDA threads in a CUDA block. Optimal setting might be `128` or `256`. Do not exceed `1024`, otherwise CUDA calls cannot be initialised and errors will be incurred.

`cudaSafeCall`<br>
Check if the wrapped CUDA call returns `cudaSuccess`. If error, print out the error message. 

`cudaCheckError`<br>
Check if there is any error left in the CUDA context. If any, print out the error message. Note that the error might be a leftover from the previous program or more likely the previous cuda call that is not carefully checked.

TODO (description of changes)

###AdvanceChannel.cu


###HSolveActive.cpp


###HSovleActive.h
 

###HSolveActiveSetup.cpp





