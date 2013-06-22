/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Multiscale Object Oriented Simulation Environment.
 **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

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
