/*
 * Definitions.hpp
 *
 *  Created on: 17/06/2009
 *      Author: rcamargo
 */

#ifndef DEFINITIONS_HPP_
#define DEFINITIONS_HPP_

//#define MPI_GPU_NN

//#include <time.h>
#include <stdlib.h>

#define PI 3.14159

#define GENSPIKETIMELIST_SIZE 2

#define FTYPE_FLOAT
//#define FTYPE_DOUBLE

#if defined(FTYPE_FLOAT)
	typedef float ftype;
	#define MPI_FTYPE MPI_FLOAT
	#define MPI_UCOMP MPI_UNSIGNED_SHORT
#elif defined(FTYPE_DOUBLE)
	typedef double ftype;
	#define MPI_FTYPE MPI_DOUBLE
	#define MPI_UCOMP MPI_UNSIGNED_SHORT
#endif

//typedef double ftype;

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
	typedef unsigned __int64 uint64;
	typedef unsigned short ucomp;
#elif defined(__APPLE__)
    typedef unsigned long long uint64;
    typedef unsigned short ucomp;
    struct random_data {};
    void random_r( struct random_data *buf, int32_t *del );
    void initstate_r(unsigned int seed, char *statebuf, size_t statelen, struct random_data *buf);
#else
    typedef unsigned long long uint64;
    typedef unsigned short ucomp;
#endif

#endif /* DEFINITIONS_HPP_ */
