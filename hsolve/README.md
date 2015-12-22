#<center>HSolve with GPU Acceleration</center>
---
####Author: Hu Wenyan
####Date: 30 Aug 2015

---

Introduction
---
This version of HSolve integrates CUDA codes to the existing HSolve library. The speedup one can gain from this implementation depends on the GPU model. A powerful GPU may speed up the HSolve process by 1.5x to 2x while an out-of-date one may actually slow it down. 

Changes are made in the following files: 

* AdvanceChannel.cu [Added]
* CudaGlobal.h [Added]
* CMakeLists.txt [Modified]
* ../CMakeLists.txt [Modified]
* HSolveActive.cpp [Modified]
* HSolveActive.h [Modified]
* HSolveActiveSetup.cpp [Modified]
* RateLookup.cu [Modified]
* RateLookup.h [Modified]

To compile the library, please make a change in ../CMakeLists.txt:

line 94: `set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_xx,code=sm_xx)`

Please change "xx" to your device computing capability and architecture.

 
Breakdown of Codes
---
###AdvanceChannel.cu

`HSolveActive::copy_to_device`<br>
Copy row array to device. Put into AdvanceChannel.cu in order to isolate CUDA calls from HSolveActive.cpp.

`advanceChannel_kernel`<br>
The kernel function to be executed on each CUDA thread and do the actual computation. Current version of the kernel function is designed to deal with one channel per thread. It is proved to be more efficient than one gate per thread.

`HSolveActive::copy_data`<br>
Copy static data from host to device. Will check if the copy has been done before and therefore only executed once during the whole computation.

`HSolveActive::advanceChannel_gpu`<br>
Driver function for the kernel. It is in charge of data conversion, memory allocation, data transfer and some cleanup stuff. It will call `advanceChannel_kernel` for the actually calculation.

###CMakeLists.txt

`Ratelookup.cpp` is commented out from the library to be built since it is replaced by `Ratelookup.cu`. All `*.cu` files are added to the `hsolve_cuda` library by default.

###CudaGlobal.h

Contain Marco definitions, bit mask definitions and functions and CUDA utility functions to be used by many of the CUDA codes. Some control variables can also be modified here.

`BLOCK_WIDTH` <br>
Determine the number of CUDA threads in a CUDA block. Optimal setting might be `128` or `256`. Do not exceed `1024`, otherwise CUDA calls cannot be initialised and errors will be incurred.

`cudaSafeCall`<br>
Check if the wrapped CUDA call returns `cudaSuccess`. If error, print out the error message. 

`cudaCheckError`<br>
Check if there is any error left in the CUDA context. If any, print out the error message. Note that the error might be a leftover from the previous program or more likely the previous cuda call that is not carefully checked.

###HSolveActive.cpp

`HSolveActive::HSolveActive`
Add initialisation for some newly added variables.

`update_info`
A debug function to profile the performance of selected modules.

`HSolveActive::step`
Add calls to `update_info` to profile individual performance of selected modules.

`HSolveActive::advanceChannels`<br>
Add calls to `advanceChannel_gpu`. __Major changes are done here.__  
1. Check if static data has been copied to device. If not, get it done through relevant functions.  
2. Look up in the table for row info. Call `row_gpu` if more than 1024 instances are involved otherwise stay with CPU codes.  
3. Copy non-static data to device.  
4. Launch `advanceChannel_gpu` to do the actual calculation on GPU.  

###HSovleActive.h

Add header infos. 

###HSolveActiveSetup.cpp

`HSolveActive::readHHChannels`<br>
Bit mask channel infos to an unsigned long long value and pack them up in a vector.

###RateLookup.cu

`void LookupTable::row(double x,double& row)`<br>
Calculate row as a double value instead of row struct.

`LookupTable::copy_table`  
Copy the lookup table to device memory. Only need to be done once since the table is static and remains unchanged.  

`row_kernel`  
The GPU Lookup kernel to be executed on each CUDA thread. It does the same lookup as the pervious lookup function.

`LookupTable::row_gpu`  
Driver function for `row_kernel`. Convert data, transfer data and manage memory.

`LookupTable::lookup`  
Deprecated lookup function. It is kept here only for testing purposes. Developed by the previous GSOC participant.



