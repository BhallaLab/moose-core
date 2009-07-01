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

struct LookupRow
{
	double* row;
	double fraction;
};

struct LookupColumn
{
	LookupColumn( ) { ; }
	unsigned int column;
	bool interpolate;
};

class LookupTable
{
public:
	LookupTable( ) { ; }
	
	LookupTable(
		double min,
		double max,
		unsigned int nDivs,
		unsigned int nSpecies );
	
	void addColumns(
		int species,
		const vector< double >& C1,
		const vector< double >& C2,
		bool interpolate );
	
	void column(
		unsigned int species,
		LookupColumn& column );
	
	void row(
		double x,
		LookupRow& row );
	
	void lookup(
		const LookupColumn& column,
		const LookupRow& row,
		double& C1,
		double& C2 );
	
private:
	vector< bool >       interpolate_;
	vector< double >     table_;
	double               min_;
	double               max_;
	unsigned int         nPts_;
	double               dx_;
	unsigned int         nColumns_;
};

#endif // _RATE_LOOKUP_H
