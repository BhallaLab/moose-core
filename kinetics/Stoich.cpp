/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace std;

#include "RateTerm.h"
#include "SparseMatrix.h"
#include "Stoich.h"

// Update the v_ vector for individual reac velocities.
void Stoich::updateV( )
{
	// Some algorithm to assign the values from the computed rates
	// to the corresponding v_ vector entry
	// for_each( rates_.begin(), rates_.end(), assign);

	vector< RateTerm* >::const_iterator i;
	vector< double >::iterator j = v_.begin();

	for ( i = rates_.begin(); i != rates_.end(); i++)
	{
		*j++ = (**i)();
	}
}

void Stoich::updateRates( vector< double>* yprime, double dt  )
{
	updateV();

	// Much Scope for optimization here.
	vector< double >::iterator j = yprime->begin();
	for (unsigned int i = 0; i < N_.nRows(); i++) {
		*j++ = dt * N_.computeRowRate( i , v_ );
	}
}
