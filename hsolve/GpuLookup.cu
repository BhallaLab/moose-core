#include <cuda.h>
#include <iostream>
#include "GpuLookup.h"

__global__ void lookup_kernel(double *row_array, double *column_array, double *table_d, unsigned int *nRows_d, unsigned int *nColumns_d, double *istate, double dt)
{

	int tId = threadIdx.x;

	int row = row_array[tId];
	double fraction = row_array[tId]-row;

	double column = column_array[tId];

	double *a = table_d, *b = table_d;

	a += (int)(row + column * (*nRows_d));
	b += (int)(row + column * (*nRows_d) + *nRows_d);

	double C1 = *a + (*(a+1) - *a) * fraction;
	double C2 = *b + (*(b+1) - *b) * fraction;

	double temp = 1.0 + dt / 2.0 * C2;
    istate[tId] = ( istate[tId] * ( 2.0 - temp ) + dt * C1 ) / temp; 
}

__global__ void do_nothing(double *result_d)
{
	int tId = threadIdx.x;
	result_d[tId] = 0;
}


GpuLookupTable::GpuLookupTable()
{

}

GpuLookupTable::GpuLookupTable(double *min, double *max, int *nDivs, unsigned int nSpecies)
{
	min_ = *min;
	max_ = *max;
	// Number of points is 1 more than number of divisions.
	// Then add one more since we may interpolate at the last point in the table.
	nPts_ = *nDivs + 1 + 1;
	dx_= ( *max - *min ) / *nDivs;
	// Every row has 2 entries for each type of gate
	nColumns_ = 0;//2 * nSpecies;

	cudaMalloc((void **)&min_d, sizeof(double));
	cudaMalloc((void **)&max_d, sizeof(double));
	cudaMalloc((void **)&nPts_d, sizeof(unsigned int));
	cudaMalloc((void **)&dx_d, sizeof(double));
	cudaMalloc((void **)&nColumns_d, sizeof(unsigned int));


 	cudaMemcpy( min_d, min, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( max_d, max, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( nPts_d, &nPts_, sizeof(unsigned int), cudaMemcpyHostToDevice);
 	cudaMemcpy( dx_d, &dx_, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( nColumns_d, &nColumns_, sizeof(unsigned int), cudaMemcpyHostToDevice);
}

void GpuLookupTable::findRow(double *V, double *rows, int size)
{
	 for (int i=0; i<size; i++)
	{
		double x = V[i];

		if ( x < min_ )
			x = min_;
		else if ( x > max_ )
			x = max_;
	 	rows[i] = ( x - min_ ) / dx_;
	// 	//std::cout << "&&&&" << rows[i] << "\n";
	}
}

void GpuLookupTable::sayHi()
{
	std::cout << "Hi there! ";
}

// Columns are arranged in memory as    |	They are visible as
// 										|	
//										|	Column 1 	Column 2 	Column 3 	Column 4
// C1(Type 1)							|	C1(Type 1) 	C2(Type 1)	C1(Type 2)	C2(Type 2)
// C2(Type 1)							|	
// C1(Type 2)							|
// C2(Type 2)							|
// .									|
// .									|
// .									|


void GpuLookupTable::addColumns(int species, double *C1, double *C2)
{
	double *table_temp_d;

	cudaMalloc((void **)&table_temp_d, (nPts_ * (nColumns_+2)) * sizeof(double));

	//Copy old values to new table
	cudaMemcpy(table_temp_d, table_d, (nPts_ * (nColumns_)) * sizeof(double), cudaMemcpyDeviceToDevice);

	//Free memory occupied by the old table
	cudaFree(table_d);
	table_d = table_temp_d;

	//Get iTable to point to last element in the table
	double *iTable = table_d + (nPts_ * nColumns_);
	
	// Loop until last but one point
	for (int i=0; i<nPts_-1; i++ )
	{
		//std::cout << i << " " << C1[i] << " " << C2[i] << "\n";
		cudaMemcpy(iTable, &C1[i], sizeof(double), cudaMemcpyHostToDevice);
		iTable++;
		
	}
	// Then duplicate the last point
	cudaMemcpy(iTable, &C1[nPts_-2], sizeof(double), cudaMemcpyHostToDevice);
	iTable++;

	//Similarly for C2
	for (int i=0; i<nPts_-1; i++ )
	{
		cudaMemcpy(iTable, &C2[i], sizeof(double), cudaMemcpyHostToDevice);
		iTable++;
	}
	cudaMemcpy(iTable, &C2[nPts_-2], sizeof(double), cudaMemcpyHostToDevice);
	iTable++;

	nColumns_ += 2;


}

void GpuLookupTable::lookup(double *row, double *column, double *istate, double dt, unsigned int set_size)
{
	double *row_array_d;
	double *column_array_d;

	// result_ = new double[set_size];

	// cudaMalloc((void **)&result_d, set_size*sizeof(double));

	cudaMalloc((void **)&row_array_d, set_size*sizeof(double));
	cudaMalloc((void **)&column_array_d, set_size*sizeof(double));
	cudaMalloc((void **)&istate_d, set_size*sizeof(double));

	cudaMemcpy(row_array_d, row, set_size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(column_array_d, column, set_size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(istate_d, istate, set_size*sizeof(double), cudaMemcpyHostToDevice);

	lookup_kernel<<<1,set_size>>>(row_array_d, column_array_d, table_d, nPts_d, nColumns_d, istate_d, dt);

	// cudaMemcpy(result_, result_d, set_size*sizeof(double), cudaMemcpyDeviceToHost);

	// std::cout << "Gpu Lookup result :  "; 
	// for (int i=0; i<set_size; i++)
	// 	std::cout << result_[i] << " ";
	// std::cout << "\n";

	// cudaMemcpy(result, result_, set_size*sizeof(double), cudaMemcpyHostToHost);
	cudaMemcpy(istate, istate_d, set_size*sizeof(double), cudaMemcpyDeviceToHost);
}
