#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include "hsolve_gpu.h"

int BLOCK_WIDTH = 512;
int SIZE = 1000;
int SAMPLE = 51200;



int testCudaHSolve(int argc, char * argv[])
{

    if(argc > 1)
    {
        SIZE = atoi(argv[1]);
    }
    if(argc > 2)
    {
        SAMPLE = atoi(argv[2]);
    }
    if(argc > 3)
    {
        BLOCK_WIDTH = atoi(argv[3]);
    }
    printf("Setting table size to %d ...\n", SIZE);
    printf("Setting data size to %d ...\n", SAMPLE);
    printf("Setting kernel block width to %d ...\n", BLOCK_WIDTH);

    double min = 1.0, max = 10.0;
    int divisions = SIZE;
    int nSpecies = 2; //Never used
    int species = 0; //Never used
    int i;
    double dt = 0.05;
    printf("Start declaring memory...\n");

    /** Using Hardcoded Sizes **/
    // double rows[SAMPLE];
    // double columns[SAMPLE];
    // double C1[SIZE];
    // double C2[SIZE];
    // double voltages[SAMPLE];
    // double states[SAMPLE];
    /**************************/

    /** Using Sizes From Arguments **/
    double * rows, * columns, *C1, *C2, *voltages, *states;
    rows = (double *)malloc(sizeof(double) * SAMPLE);
    columns = (double *) malloc(sizeof(double) * SAMPLE);
    C1 = (double *)malloc(sizeof(double) * SIZE);
    C2 = (double *)malloc(sizeof(double) * SIZE);
    voltages = (double *)malloc(sizeof(double) * SAMPLE);
    states = (double *)malloc(sizeof(double) * SAMPLE);
    /**************************/

    printf("Start reading data...\n");
    readDataDouble(C1, "C1-1.txt", SIZE);
    readDataDouble(C2, "C2-1.txt", SIZE);
    readDataDouble(voltages, "voltages.txt", SAMPLE);
    readDataDouble(columns, "columns.txt", SAMPLE); //All 0's for testing purposes.
    readDataDouble(states, "states.txt", SAMPLE);

    cudaEvent_t start, stop;
    float gputime;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);
    cudaEventRecord (start, 0);

    //printf("Start initilizing...\n");
    GpuLookupTable table(&min, &max, &divisions, nSpecies);

    //printf("Start adding columns...\n");
    table.addColumns(species, C1, C2);

    //printf("Start finding rows...\n");
    table.findRow(voltages, rows, SAMPLE);

    //printf("Start looking up...\n");
    table.lookup(rows, columns, states, dt, SAMPLE);

    cudaThreadSynchronize();
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&gputime, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("It takes %f us.\n", gputime);

    printf("The resulting states are : \n");
    for(i = 0; i < SAMPLE && i < 100; ++i)
    {
        printf("%.2f ", states[i]);
        if(i %10 == 9)
        {
            printf("\n");
        }
    }

    printf("\n");

    table.destory();

    /** Free Used Memory **/
    free(rows);
    free(columns);
    free(C1);
    free(C2);
    free(voltages);
    free(states);
    /**********************/

    return 0;
}

void readDataDouble(double * array, char * fileName, int num)
{

    int i;
    double k;
    FILE * file;
    file = fopen(fileName, "r");
    for(i = 0; i < num; ++i)
    {
        fscanf(file, "%lf", &k);
        array[i] = k;
    }
    fclose(file);
}

void readDataInt(int * array, char * fileName, int num)
{

    int i;
    FILE * file;
    file = fopen(fileName, "r");
    for(i = 0; i < num; ++i)
    {
        fscanf(file, "%d", &array[i]);
    }
    fclose(file);
}

__global__ void lookup_kernel(double *row_array, double *column_array, double *table_d, unsigned int nRows_d, unsigned int nColumns, double *istate, double dt, unsigned int set_size)
{

    int tId = threadIdx.x + blockIdx.x * blockDim.x;

    if(tId >= set_size) return;

    int row = row_array[tId];
    double fraction = row_array[tId]-row;

    double column = column_array[tId];

    double *a = table_d, *b = table_d;

    a += (int)(row + column * (nRows_d));
    b += (int)(row + column * (nRows_d) + nRows_d);

    double C1 = *a + (*(a+1) - *a) * fraction;
    double C2 = *b + (*(b+1) - *b) * fraction;

    double temp = 1.0 + dt / 2.0 * C2;
    istate[tId] = ( istate[tId] * ( 2.0 - temp ) + dt * C1 ) / temp;
}

GpuLookupTable::GpuLookupTable(double min, double max, unsigned int nDivs, unsigned int nSpecies)
{
    min_ = min;
    max_ = max;
    // Number of points is 1 more than number of divisions.
    // Then add one more since we may interpolate at the last point in the table.
    nPts_ = nDivs + 1 + 1;
    dx_= ( max - min ) / nDivs;
    // Every row has 2 entries for each type of gate
    nColumns_ = 2 * nSpecies;

}

void GpuLookupTable::destory()
{
    cudaFree(table_d);
    cudaFree(rows_d);
}

__global__
void find_row_kernel(double * V_d, double * rows_d, double min, double max, double dx, int size)
{

    int tId = threadIdx.x + blockIdx.x * blockDim.x;

    if(tId >= size) return;

    double v = V_d[tId];

    if ( v < min )
        v = min;
    else if ( v > max )
        v = max;
    rows_d[tId] = ( v - min ) / dx;

}

void GpuLookupTable::findRow(double *V, double *rows, int size)
{

    double *V_d;
    cudaMalloc(&rows_d, sizeof(double) * size);
    cudaMalloc(&V_d, sizeof(double) * size);
    cudaMemcpy(rows_d, rows, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, sizeof(double) * size, cudaMemcpyHostToDevice);
    const dim3 blockSize(size/BLOCK_WIDTH + 1, 1, 1);
    const dim3 gridSize(BLOCK_WIDTH,1,1);
    find_row_kernel<<<gridSize, blockSize>>>(V_d, rows_d, min_, max_, dx_, size);
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
    int i;
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
    // double * nc1 = (double *)malloc(sizeof(double) * nPts_);
    // double * nc2 = (double *)malloc(sizeof(double) * nPts_);
    // memcpy(nc1, C1, sizeof(double) * (nPts_ - 1));
    // nc1[nPts_ - 1] = C1[nPts_ - 2];
    // memcpy(nc2, C2, sizeof(double) * (nPts_ - 1));
    // nc2[nPts_ - 1] = C2[nPts_ - 2];

    cudaMemcpy(iTable, C1, sizeof(double) * (nPts_), cudaMemcpyHostToDevice);
    iTable += nPts_ - 1;
    cudaMemcpy(iTable, &C1[nPts_ - 2], sizeof(double), cudaMemcpyHostToDevice);
    iTable ++;

    cudaMemcpy(iTable, C2, sizeof(double) * (nPts_), cudaMemcpyHostToDevice);
    iTable += nPts_ - 1;
    cudaMemcpy(iTable, &C2[nPts_ - 2], sizeof(double), cudaMemcpyHostToDevice);
    iTable ++;

    nColumns_ += 2;

}

void GpuLookupTable::lookup(double *row, double *column, double *istate, double dt, unsigned int set_size)
{
    double *column_array_d;

    // result_ = new double[set_size];

    // cudaMalloc((void **)&result_d, set_size*sizeof(double));

    cudaMalloc((void **)&column_array_d, set_size*sizeof(double));
    cudaMalloc((void **)&istate_d, set_size*sizeof(double));

    //cudaMemcpy(row_array_d, row, set_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(column_array_d, column, set_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(istate_d, istate, set_size*sizeof(double), cudaMemcpyHostToDevice);

    const dim3 blockSize(set_size/BLOCK_WIDTH + 1, 1, 1);
    const dim3 gridSize(BLOCK_WIDTH,1,1);


    //printf("Start kernel...\n");

    lookup_kernel<<<gridSize,blockSize>>>(rows_d, column_array_d, table_d, nPts_, nColumns_, istate_d, dt, set_size);


    // cudaMemcpy(result_, result_d, set_size*sizeof(double), cudaMemcpyDeviceToHost);

    // std::cout << "Gpu Lookup result :  ";
    // for (int i=0; i<set_size; i++)
    // 	std::cout << result_[i] << " ";
    // std::cout << "\n";

    // cudaMemcpy(result, result_, set_size*sizeof(double), cudaMemcpyHostToHost);
    cudaMemcpy(istate, istate_d, set_size*sizeof(double), cudaMemcpyDeviceToHost);

    //sleep(2);
    cudaFree(column_array_d);
    cudaFree(istate_d);
}
