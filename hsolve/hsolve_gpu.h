#ifndef GPU_LOOKUP_H
#define GPU_LOOKUP_H

struct GpuLookupRow
{
	double* row;		///< Pointer to the first column on a row
	double fraction;	///< Fraction of V or Ca over and above the division
						///< boundary for interpolation.
};

struct GpuLookupColumn
{
	unsigned int column;
};

class GpuLookupTable
{
	public:

		double min_, max_, dx_;
		unsigned int nPts_, nColumns_;
		double * rows_d;
		double *min_d, *max_d, *dx_d, *istate_d, *result_;
		unsigned int *nPts_d, *nColumns_d;
		double *table_d;
		//__global__ void lookup_kernel(double *row_array, double *column_array, double *table_d, unsigned int nRows_d, unsigned int nColumns_d, double *istate, double dt, unsigned int set_size);
		//__global__ void find_row_kernel(double * V_d, double * rows_d, double min, double max, double dx, int size);
		GpuLookupTable();
		GpuLookupTable(double *min, double *max, int *nDivs, unsigned int nSpecies);
		// void row(double V, double *row);
		void addColumns(int species, double *C1, double *C2);

		void findRow(double *V, double *rows, int size);
		
		void lookup(double *row, double *column, double *istate, double dt, unsigned int set_size);

		void sayHi();

		void destory();
		
};

void readDataInt(int *, char *, int);
void readDataDouble(double *, char *, int);

#endif
