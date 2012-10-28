/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "FuncTerm.h"
#include "SumTotalTerm.h"

double SumTotalTerm::operator() ( const double* S, double t ) const
{
	double ret = 0.0;
	for( vector< unsigned int >::const_iterator i = mol_.begin(); 
		i != mol_.end(); i++ )
		ret += S[ *i ];
	return ret;
}

unsigned int SumTotalTerm::getReactants( 
				vector< unsigned int >& molIndex) const
{
	molIndex = mol_;
	return mol_.size();
}

void SumTotalTerm::setReactants( const vector< unsigned int >& mol )
{
	mol_ = mol;
}

const string& SumTotalTerm::function() const
{
	static string ret = "f( arg1, arg2 ) = arg1 + arg2";
	return ret;
}

