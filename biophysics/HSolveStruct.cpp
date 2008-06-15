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
#include <queue>
#include "SynInfo.h"
#include "RateLookup.h"
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

void ChannelStruct::process( double*& state, double& gk, double& gkek )
{
	double fraction = 1.0;
	
	if( Xpower_ )
		fraction *= takeXpower_( *( state++ ), Xpower_ );
	if( Ypower_ )
		fraction *= takeYpower_( *( state++ ), Ypower_ );
	if( Zpower_ )
		fraction *= takeZpower_( *( state++ ), Zpower_ );
	
	gk = Gbar_ * fraction;
	gkek = GbarEk_ * fraction;
}

void SynChanStruct::process( ProcInfo info ) {
	while ( !pendingEvents_->empty() &&
		pendingEvents_->top().delay <= info->currTime_ ) {
		*activation_ += pendingEvents_->top().weight / info->dt_;
		pendingEvents_->pop();
	}
	X_ = *modulation_ * *activation_ * xconst1_ + X_ * xconst2_;
	Y_ = X_ * yconst1_ + Y_ * yconst2_;
	Gk_ = Y_ * norm_;
	*activation_ = 0.0;
	*modulation_ = 1.0;
}

double CaConcStruct::process( double activation ) {
	c_ = factor1_ * c_ + factor2_ * activation;
	return ( CaBasal_ + c_ );
}
