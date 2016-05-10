/*
 * =====================================================================================
 *
 *       Filename:  BoostSystem.cpp
 *
 *    Description:  Ode system described boost library.
 *
 *        Created:  04/11/2016 10:58:34 AM
 *       Compiler:  g++
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */

#include "BoostSys.h"
#include <iostream>
#include "VoxelPools.h"

BoostSys::BoostSys( )
{ ; }

BoostSys::~BoostSys()
{ ;  }


void BoostSys::operator()( const vector_type_ y
        , vector_type_& dydt, const double t )
{
    VoxelPools::evalRates( y, dydt, t, vp );
}
