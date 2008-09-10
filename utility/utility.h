/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
** Development of this software was supported by
** Biophase Simulations Inc, http://www.bpsims.com
** See the file BIOPHASE.INFO for details.
**********************************************************************/

#ifndef _UTILITY_H
#define _UTILITY_H

#include "randnum/randnum.h"
#include "StringUtil.h"
#include "Property.h"
#include "PathUtility.h"
#include "ArgParser.h"
#include "randnum/NumUtil.h"
#include <cmath>
#include <limits>

/**
 * Functions for floating point comparisons
 */
template<class T>
bool isNaN( T value )
{
	return value != value;
}

template< typename T >
bool isInfinity( T value )
{
	return value == std::numeric_limits< T >::infinity();
}

/**
 * Check 2 floating-point numbers for "equality".
 * Algorithm (from Knuth) 'a' and 'b' are close if:
 *      | ( a - b ) / a | < e AND | ( a - b ) / b | < e
 * where 'e' is a small number.
 * 
 * In this function, 'e' is computed as:
 * 	    e = tolerance * machine-epsilon
 */
template< class T >
bool isClose( T a, T b, T tolerance )
{
	T epsilon = numeric_limits< T >::epsilon();
	
	if ( a == b )
		return true;
	
	if ( a == 0 || b == 0 )
		return ( fabs( a - b ) < tolerance * epsilon );
	
	return (
		fabs( ( a - b ) / a ) < tolerance * epsilon
		&&
		fabs( ( a - b ) / b ) < tolerance * epsilon
	);
}

#endif
