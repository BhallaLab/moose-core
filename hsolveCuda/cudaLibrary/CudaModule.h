#ifndef CUDA_MODULE_H
#define CUDA_MODULE_H

extern int 	cudaModule_setup (
		const unsigned int __cellNumber,
		const unsigned int __nComp,
		float * host_fullMatrix,
		float * host_V,
		float * host_Cm,
		float * host_Em,
		float * host_Rm,
		float * host_Ra);

extern void 	cudaModule_discard();

extern void 	cudaModule_updateMatrix(float dt);
extern void 	cudaMosule_forwardElimination();
extern void 	cudaMosule_backwardSubstitute();
extern int 	cudaModule_getB(float * B);
extern int 	cudaModule_test();
#endif
