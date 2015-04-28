#ifndef GPU_LOOKUP_H
#define GPU_LOOKUP_H

struct GpuLookupRow
{
	double* row;		///< Pointer to the first column on a row
	double fraction;	///< Fraction of V or Ca over and above the division
						///< boundary for interpolation.
};

struct LookupColumn
{
	LookupColumn() { ; }
	unsigned int column;
	//~ bool interpolate;
};

class GpuLookupTable
{
	public:
		GpuLookupTable(){;}
		GpuLookupTable(
			double min, 
			double max, 
			unsigned int nDivs, 
			unsigned int nSpecies);

		
		// void row(double V, double *row);
		void addColumns(int species, double *C1, double *C2);

		void findRow(double *V, double *rows, int size);
		
		void lookup(double *row, double *column, double *istate, double dt, unsigned int set_size);

		void sayHi();

		void destory();
		
	private:
		vector< double >     table_;		///< Flattened table
		double               min_;			///< min of the voltage / caConc range
		double               max_;			///< max of the voltage / caConc range
		unsigned int         nPts_;			///< Number of rows in the table.
											///< Equal to nDivs + 2, so that
											///< interpol. is safe at either end.
		double               dx_;			///< This is the smallest difference:
											///< (max - min) / nDivs
		unsigned int         nColumns_;		///< (# columns) = 2 * (# species)	

		double *rows_d;						// Device Array of rows
		double *istate_d;					// Device Array of states
		double *table_d;					// Device Array of tables	
};

void readDataInt(int *, char *, int);
void readDataDouble(double *, char *, int);

#endif
