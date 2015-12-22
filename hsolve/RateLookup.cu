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

#include "CudaGlobal.h"
#include "RateLookup.h"

#ifdef USE_CUDA
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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
	is_set_ = false;
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
/*
 * Calculate row as a double value instead of row struct.
 */
void LookupTable::row(double x,double& row)
{
	if ( x < min_ )
		x = min_;
	else if ( x > max_ )
		x = max_;
	
	double div = ( x - min_ ) / dx_;
	unsigned int integer = ( unsigned int )( div );
	row = integer * nColumns_ + (div - integer);
}


/*
 * Copy the lookup table into device memory.
 * Only need to be done once since the table is
 * static and remains unchanged.
 */
void LookupTable::copy_table()
{

	int size =LookupTable::table_.size();
	if(size <= 0) 
	{
		size = 0;
	}

	if(size > 0)
	{
		cudaSafeCall(cudaMalloc((void **)&(LookupTable::table_d),
								size * sizeof(double))); 

		cudaSafeCall(cudaMemcpy(LookupTable::table_d,
								&(LookupTable::table_.front()), 
								size * sizeof(double), 
								cudaMemcpyHostToDevice));		
	}

}
double * LookupTable::get_state_d()
{
	return  LookupTable::state_d;
}
double * LookupTable::get_table_d()
{
	return  LookupTable::table_d;
}
bool LookupTable::is_set()
{
	return LookupTable::is_set_;
}
bool LookupTable::set_is_set(bool set_val)
{
	LookupTable::is_set_ = set_val;
	return LookupTable::is_set();
}
unsigned int LookupTable::get_num_of_points()
{
    return LookupTable::nPts_;
}
     
unsigned int LookupTable::get_num_of_columns()
{
    return LookupTable::nColumns_;
}
vector<double> LookupTable::get_table()
{
    return LookupTable::table_;
}

double LookupTable::get_min()
{
	return min_;
}

double LookupTable::get_max()
{
	return max_;
}

double LookupTable::get_dx()
{
	return dx_;
}

/*
 * GPU lookup kernel using row structs
 */
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

/*
 * GPU lookup kernel using double values for rows
 */
__global__
void
row_kernel(double * d_x, 
		   double * row, 
		   double min,
		   double max, 
		   double dx,
		   unsigned int nColumns, 
		   unsigned int size)
{
			   
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(tid >= size) return;
	
	double x = d_x[tid];
	
	if ( x < min )
		x = min;
	else if ( x > max )
		x = max;
	
	double div = ( x - min ) / dx;
	unsigned int integer = ( unsigned int )( div );
	
	row[tid] = integer * nColumns + double(div - integer);
	
}


/*
 * Driver function for lookup kernel using row struct.
 */
void LookupTable::row_gpu(vector<double>::iterator& x, vector<LookupRow>::iterator& row, unsigned int size)
{

#ifdef DEBUG_VERBOSE
    printf("start row_gpu calculation...\n");
#endif	

    printf("A\n");
    thrust::device_vector<double> d_x(size);
    thrust::device_vector<LookupRow> d_row(size);

    thrust::copy(x, x + size, d_x.begin());
    thrust::copy(row, row + size, d_row.begin());

    printf("AA\n");

    double * d_x_p = thrust::raw_pointer_cast(d_x.data());
    LookupRow * d_row_p = thrust::raw_pointer_cast(d_row.data());
    dim3 gridSize(size/BLOCK_WIDTH + 1, 1, 1);
    dim3 blockSize(BLOCK_WIDTH,1,1);

    if(size <= BLOCK_WIDTH)
    {
        gridSize.x = 1;
        blockSize.x = size;
    }

    size_t address = reinterpret_cast<size_t>(&table_.front());

    printf("AAA\n");
    row_kernel<<<gridSize, blockSize>>>(d_x_p, 
            d_row_p, 
            min_, 
            max_, 
            dx_, 
            nColumns_, 
            size, 
            address);	

    cudaSafeCall(cudaDeviceSynchronize()); 

#ifdef DEBUG_VERBOSE    
    printf("kernel launch finished...\n");
#endif
    thrust::copy(d_row.begin(), d_row.end(), row);

#ifdef DEBUG_VERBOSE    
    printf("finish row_gpu calculation...\n");
#endif 
}

/*
 * Driver function for lookup kernel using double values.
 */
void LookupTable::row_gpu(vector<double>::iterator& x, double ** row, unsigned int size)
{

#ifdef DEBUG_VERBOSE
	printf("start row_gpu calculation...\n");
#endif	

	thrust::device_vector<double> d_x(size);	
	cudaSafeCall(cudaMalloc((void**)row, sizeof(double) * size));	
	thrust::copy(x, x + size, d_x.begin());
	double * d_x_p = thrust::raw_pointer_cast(d_x.data());

    dim3 gridSize(size/BLOCK_WIDTH + 1, 1, 1);
    dim3 blockSize(BLOCK_WIDTH,1,1);

    if(size <= BLOCK_WIDTH)
    {
    	gridSize.x = 1;
    	blockSize.x = size;
    }
    
    row_kernel<<<gridSize, blockSize>>>(d_x_p, 
    									*row, 
    									min_, 
    									max_, 
    									dx_, 
    									nColumns_, 
    									size);	
    cudaCheckError();
    cudaSafeCall(cudaDeviceSynchronize()); 

#ifdef DEBUG_VERBOSE    
    printf("kernel launch finished...\n");
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
