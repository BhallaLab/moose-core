#ifndef _STORAGE_FORMATS_H
#define _STORAGE_FORMATS_H


/*
 * COOSR storage format. For more details refer to
 * http://link.springer.com/chapter/10.1007/978-3-319-32149-3_11
 */
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
