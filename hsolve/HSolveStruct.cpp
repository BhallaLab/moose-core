/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <cmath>
#include "moose.h"
#include "HSolveStruct.h"

void ChannelStruct::setPowers(
	double Xpower, double Ypower, double Zpower )
{
	Xpower_ = Xpower;
	takeXpower_ = selectPower( Xpower );
	
	Ypower_ = Ypower;
	takeYpower_ = selectPower( Ypower );
	
	Zpower_ = Zpower;
	takeZpower_ = selectPower( Zpower );
}

double ChannelStruct::powerN( double x, double p )
{
	if ( x > 0.0 )
		return exp( p * log( x ) );
	return 0.0;
}

PFDD ChannelStruct::selectPower( double power )
{
	if ( power == 0.0 )
		return powerN;
	else if ( power == 1.0 )
		return power1;
	else if ( power == 2.0 )
		return power2;
	else if ( power == 3.0 )
		return power3;
	else if ( power == 4.0 )
		return power4;
	else
		return powerN;
}

void ChannelStruct::process( double*& state, CurrentStruct& current )
{
	double fraction = 1.0;
	
	if( Xpower_ > 0.0 )
		fraction *= takeXpower_( *( state++ ), Xpower_ );
	if( Ypower_ > 0.0 )
		fraction *= takeYpower_( *( state++ ), Ypower_ );
	if( Zpower_ > 0.0 )
		fraction *= takeZpower_( *( state++ ), Zpower_ );
	
	current.Gk = Gbar_ * fraction;
}

void CaConcStruct::process( double activation ) {
	c_ = factor1_ * c_ + factor2_ * activation;
	ca_ = ( CaBasal_ + c_ );
}
