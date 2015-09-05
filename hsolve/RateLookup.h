/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _RATE_LOOKUP_H
#define _RATE_LOOKUP_H

#include "CudaGlobal.h"

using namespace std;

struct LookupRow
{
	double* row;		///< Pointer to the first column on a row
	int rowIndex;
	double fraction;	///< Fraction of V or Ca over and above the division
						///< boundary for interpolation.
};

struct LookupColumn
{
	LookupColumn() { ; }
	unsigned int column;
	//~ bool interpolate;
};

class LookupTable
{
public:
	LookupTable() { ; }
	
	LookupTable(
		double min,					///< min of range
		double max,					///< max of range
		unsigned int nDivs,			///< number of divisions (~ no. of rows)
		unsigned int nSpecies );	///< number of species (no. of columns / 2)
	
	/// Adds the columns for a given species. Columns supplied are C1 and C2
	void addColumns(
		int species,
		const std::vector< double >& C1,
		const std::vector< double >& C2 );
		//~ const vector< double >& C2,
		//~ bool interpolate );
	
	void column(
		unsigned int species,
		LookupColumn& column );
	
	/**
	 * Returns the row corresponding to x in the "row" parameter.
	 * i.e., returns the leftover fraction and the row's start address.
	 */
	void row(double x,LookupRow& row );

#ifdef USE_CUDA
	void row(double x,double& row);
	void row_gpu(vector<double>::iterator& x, 
				 double ** row, 
				 unsigned int size);
	void row_gpu(vector<double>::iterator& x, 
						vector<LookupRow>::iterator& row, 
						unsigned int size);
    unsigned int get_num_of_points();
    unsigned int get_num_of_columns();
    vector<double> get_table();	
    double get_min();
    double get_max();
    double get_dx();		
    double * get_state_d();
    double * get_table_d();
    bool is_set();	
    bool set_is_set(bool set_val);	
    void copy_table();
#endif
	/// Actually performs the lookup and the linear interpolation
	void lookup(
		const LookupColumn& column,
		const LookupRow& row,
		double& C1,
		double& C2 );
	
private:
	//~ vector< bool >       interpolate_;
	vector< double >     table_;		///< Flattened table
	double               min_;			///< min of the voltage / caConc range
	double               max_;			///< max of the voltage / caConc range
	unsigned int         nPts_;			///< Number of rows in the table.
										///< Equal to nDivs + 2, so that
										///< interpol. is safe at either end.
	double               dx_;			///< This is the smallest difference:
										///< (max - min) / nDivs
	unsigned int         nColumns_;		///< (# columns) = 2 * (# species)

#ifdef USE_CUDA
	double				 *state_d;		///< device array of istate
	double 				 *table_d;		///< device array of the flattened table
	bool is_set_;
#endif	


};

#endif // _RATE_LOOKUP_H
