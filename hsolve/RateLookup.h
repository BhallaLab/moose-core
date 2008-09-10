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

class RateLookupGroup;

struct LookupKey
{
	unsigned int offset1;
	unsigned int offset2;
	double fraction;
};

class RateLookup
{
public:
	RateLookup( double* base, RateLookupGroup* group, bool interpolate );
	void getKey( double x, LookupKey& key );
	void rates( const LookupKey& key, double& C1, double& C2 );
	
private:
	bool interpolate_;
	double* base_;
	RateLookupGroup* group_;
};

class RateLookupGroup
{
public:
	RateLookupGroup(
		double min, double max,
		unsigned int nDivs, unsigned int nSpecies );
	void addTable(
		int species,
		const vector< double >& C1,
		const vector< double >& C2,
		bool interpolate );
	RateLookup slice( unsigned int species );
	void getKey( double x, LookupKey& key );
	
private:
	vector< bool > interpolate_;
	vector< double > table_;
	double min_;
	double max_;
	unsigned int nPts_;
	double dx_;
	unsigned int nColumns_;
};

#endif // _RATE_LOOKUP_H
