/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <vector>
#include <stdio.h>
using namespace std;

#include "RateLookup.h"

#ifndef USE_CUDA
//#define USE_CUDA
#endif

#ifndef DEBUG
//#define DEBUG
#endif

#ifndef DEBUG_VERBOSE
//#define DEBUG_VERBOSE
#endif

#ifndef DEBUG_STEP
//#define DEBUG_STEP
#endif

#ifdef USE_CUDA
#define BLOCK_WIDTH 256
#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#endif


LookupTable::LookupTable(
	double min, double max, unsigned int nDivs, unsigned int nSpecies )
{
	min_ = min;
	max_ = max;
	// Number of points is 1 more than number of divisions.
	// Then add one more since we may interpolate at the last point in the table.
	nPts_ = nDivs + 1 + 1;
	dx_ = ( max - min ) / nDivs;
	// Every row has 2 entries for each type of gate
	nColumns_ = 2 * nSpecies;
	
	//~ interpolate_.resize( nSpecies );
	table_.resize( nPts_ * nColumns_ );
	
}

void LookupTable::addColumns(
	int species,
	const vector< double >& C1,
	const vector< double >& C2 )
	//~ const vector< double >& C2,
	//~ bool interpolate )
{
	vector< double >::const_iterator ic1 = C1.begin();
	vector< double >::const_iterator ic2 = C2.begin();
	vector< double >::iterator iTable = table_.begin() + 2 * species;
	// Loop until last but one point
	for ( unsigned int igrid = 0; igrid < nPts_ - 1 ; ++igrid ) {
		*( iTable )     = *ic1;
		*( iTable + 1 ) = *ic2;
		
		iTable += nColumns_;
		++ic1, ++ic2;
	}
	// Then duplicate the last point
	*( iTable )     = C1.back();
	*( iTable + 1 ) = C2.back();
	
	//~ interpolate_[ species ] = interpolate;
}

void LookupTable::column( unsigned int species, LookupColumn& column )
{
	column.column = 2 * species;
	//~ column.interpolate = interpolate_[ species ];
}

void LookupTable::row( double x, LookupRow& row )
{
	if ( x < min_ )
		x = min_;
	else if ( x > max_ )
		x = max_;
	
	double div = ( x - min_ ) / dx_;
	unsigned int integer = ( unsigned int )( div );
	
	row.fraction = div - integer;
	row.row = &( table_.front() ) + integer * nColumns_;
	row.rowIndex = integer * nColumns_;
}

#ifdef USE_CUDA

__global__
void
row_kernel(double * d_x, 
		   LookupRow * d_row, 
		   double min,
		   double max, 
		   double dx,
		   unsigned int nColumns, 
		   unsigned int size,
		   size_t address)
{
			   
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(tid >= size) return;
	
	//if(tid == 0) printf("kernel launch successful!\n");
	
	double x = d_x[tid];
	
	if ( x < min )
		x = min;
	else if ( x > max )
		x = max;
	
	double div = ( x - min ) / dx;
	unsigned int integer = ( unsigned int )( div );
	
	d_row[tid].fraction = div - integer;
	d_row[tid].row = reinterpret_cast<double*>(address + sizeof(double) * integer * nColumns);	
	d_row[tid].rowIndex = integer * nColumns;
}

void LookupTable::row_gpu(vector<double>::iterator& x, vector<LookupRow>::iterator& row, unsigned int size){

#ifdef DEBUG_VERBOSE
	printf("start row_gpu calculation...\n");
#endif	
	std::vector<double> h_x(size);
	std::copy(x, x + size, h_x.begin());
	
	thrust::device_vector<double> d_x(size);
	thrust::device_vector<LookupRow> d_row(size);
	
	double * d_x_p = thrust::raw_pointer_cast(d_x.data());
	LookupRow * d_row_p = thrust::raw_pointer_cast(d_row.data());
	
	cudaMemcpy(d_x_p, &h_x.front(), sizeof(double)*size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize(); 
	
	h_x.clear();
	
    const dim3 gridSize(size/BLOCK_WIDTH + 1, 1, 1);
    const dim3 blockSize(BLOCK_WIDTH,1,1);
    
    size_t address = reinterpret_cast<size_t>(&table_.front());
    
    row_kernel<<<gridSize, blockSize>>>(d_x_p, d_row_p, min_, max_, dx_, nColumns_, size, address);	
    
    cudaThreadSynchronize();
    cudaDeviceSynchronize(); 
#ifdef DEBUG_VERBOSE    
    printf("kernel launch finished...\n");
#endif
    LookupRow * h_row;
    h_row = (LookupRow *) malloc(sizeof(LookupRow)*size);
    cudaMemcpy(h_row, d_row_p, sizeof(LookupRow)*size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
    std::copy(h_row, h_row+size, row);
    
    free(h_row);
#ifdef DEBUG_VERBOSE    
    printf("finish row_gpu calculation...\n");
#endif 
}
#endif

void LookupTable::lookup(
	const LookupColumn& column,
	const LookupRow& row,
	double& C1,
	double& C2 )
{
	double a, b;
	double *ap, *bp;
	
	ap = row.row + column.column;
	
	//~ if ( ! column.interpolate ) {
		//~ C1 = *ap;
		//~ C2 = *( ap + 1 );
		//~ 
		//~ return;
	//~ }
	
	bp = ap + nColumns_;
	
	a = *ap;
	b = *bp;
	C1 = a + ( b - a ) * row.fraction;
	
	a = *( ap + 1 );
	b = *( bp + 1 );
	C2 = a + ( b - a ) * row.fraction;
}
