#ifndef _STORAGE_FORMATS_H
#define _STORAGE_FORMATS_H

typedef struct{
	int rows;
	int cols;
	int nnz;

	double* values;
	int* rowIndex;
	int* colIndex;
	int* rowPtr;
}coosr_matrix;

#endif // _STORAGE_FORMATS_H
