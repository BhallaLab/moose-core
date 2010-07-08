/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <vector>
#include <string>
using namespace std;
#include "FuncTerm.h"

double SumTotal::operator() ( double t ) const
{
	double ret = 0.0;
	for( vector< const double* >::const_iterator i = mol_.begin(); 
		i != mol_.end(); i++ )
		ret += **i;
	return ret;
}

unsigned int SumTotal::getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const
{
	molIndex.resize( mol_.size() );
	for ( unsigned int i = 0; i < mol_.size(); i++ )
		molIndex[i] = mol_[i] - &S[0];
	return mol_.size();
}

const string& SumTotal::function() const
{
	static string ret = "f( arg1, arg2 ) = arg1 + arg2";
	return ret;
}
