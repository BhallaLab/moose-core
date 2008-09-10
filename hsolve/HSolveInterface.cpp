/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"

const vector< Id >& HSolveActive::getCompartments( ) const
{
	return compartmentId_;
}

const vector< Id >& HSolveActive::getHHChannels( ) const
{
	return channelId_;
}

const vector< Id >& HSolveActive::getCaConcs( ) const
{
	return caConcId_;
}

double HSolveActive::getVm( unsigned int index ) const
{
	assert( index < V_.size() );
	
	if ( index < V_.size() )
		return V_[ index ];
	
	return 0.0;
}

void HSolveActive::setVm( unsigned int index, double value )
{
	assert( index < V_.size() );
	
	if ( index < V_.size() )
		V_[ index ] = value;
}

double HSolveActive::getInject( unsigned int index ) const
{
	assert( index < nCompt_ );
	
	map< unsigned int, InjectStruct >::const_iterator i;
	
	if ( index < nCompt_ ) {
		i = inject_.find( index );
		
		if ( i != inject_.end() )
			return i->second.injectBasal;
	}
	
	return 0.0;
}

void HSolveActive::setInject( unsigned int index, double value )
{
	assert( index < nCompt_ );
	
	if ( index < nCompt_ )
		inject_[ index ].injectBasal = value;
}

double HSolveActive::getIm( unsigned int index ) const
{
	assert( index < nCompt_ );
	
	if ( index >= nCompt_ )
		return 0.0;
	
	double Im =
		compartment_[ index ].EmByRm - V_[ index ] / tree_[ index ].Rm;
	
	vector< CurrentStruct >::const_iterator icurrent;
	
	if ( index == 0 )
		icurrent = current_.begin();
	else
		icurrent = currentBoundary_[ index - 1 ];
	
	for ( ; icurrent < currentBoundary_[ index ]; ++icurrent )
		Im += ( icurrent->Ek - V_[ index ] ) * icurrent->Gk;
	
	return Im;
}

void HSolveActive::addInject( unsigned int index, double value )
{
	inject_[ index ].injectVarying += value;
}

double HSolveActive::getHHChannelGbar( unsigned int index ) const
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() )
		return channel_[ index ].Gbar_;
	
	return 0.0;
}

void HSolveActive::setHHChannelGbar( unsigned int index, double value )
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() )
		channel_[ index ].Gbar_ = value;
}

double HSolveActive::getEk( unsigned int index ) const
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() )
		return current_[ index ].Ek;
	
	return 0.0;
}

void HSolveActive::setEk( unsigned int index, double value )
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() )
		current_[ index ].Ek = value;
}

double HSolveActive::getGk( unsigned int index ) const
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() )
		return current_[ index ].Gk;
	
	return 0.0;
}

void HSolveActive::setGk( unsigned int index, double value )
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() )
		current_[ index ].Gk = value;
}

double HSolveActive::getIk( unsigned int index ) const
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() ) {
		unsigned int comptIndex = chan2compt_[ index ];
		assert( comptIndex < V_.size() );
		
		return 
			( current_[ index ].Ek - V_[ comptIndex ] ) * current_[ index ].Gk;
	}
	
	return 0.0;
}

double HSolveActive::getX( unsigned int index ) const
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() ) {
		if ( ! channel_[ index ].Xpower_ )
			return 0.0;
		
		unsigned int stateIndex = chan2state_[ index ];
		assert( stateIndex < state_.size() );
		
		return state_[ stateIndex ];
	}
	
	return 0.0;
}

void HSolveActive::setX( unsigned int index, double value )
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() ) {
		if ( ! channel_[ index ].Xpower_ )
			return;
		
		unsigned int stateIndex = chan2state_[ index ];
		assert( stateIndex < state_.size() );
		
		state_[ stateIndex ] = value;
	}
}

double HSolveActive::getY( unsigned int index ) const
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() ) {
		if ( ! channel_[ index ].Ypower_ )
			return 0.0;
		
		unsigned int stateIndex = chan2state_[ index ];
		
		if ( channel_[ index ].Xpower_ )
			++stateIndex;
		
		assert( stateIndex < state_.size() );
		
		return state_[ stateIndex ];
	}
	
	return 0.0;
}

void HSolveActive::setY( unsigned int index, double value )
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() ) {
		if ( ! channel_[ index ].Ypower_ )
			return;
		
		unsigned int stateIndex = chan2state_[ index ];
		
		if ( channel_[ index ].Xpower_ )
			++stateIndex;
		
		assert( stateIndex < state_.size() );
		
		state_[ stateIndex ] = value;
	}
}

double HSolveActive::getZ( unsigned int index ) const
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() ) {
		if ( ! channel_[ index ].Zpower_ )
			return 0.0;
		
		unsigned int stateIndex = chan2state_[ index ];
		
		if ( channel_[ index ].Xpower_ )
			++stateIndex;
		if ( channel_[ index ].Ypower_ )
			++stateIndex;
		
		assert( stateIndex < state_.size() );
		
		return state_[ stateIndex ];
	}
	
	return 0.0;
}

void HSolveActive::setZ( unsigned int index, double value )
{
	assert( index < channel_.size() );
	
	if ( index < channel_.size() ) {
		if ( ! channel_[ index ].Zpower_ )
			return;
		
		unsigned int stateIndex = chan2state_[ index ];
		
		if ( channel_[ index ].Xpower_ )
			++stateIndex;
		if ( channel_[ index ].Ypower_ )
			++stateIndex;
		
		assert( stateIndex < state_.size() );
		
		state_[ stateIndex ] = value;
	}
}

double HSolveActive::getCa( unsigned int index ) const
{
	assert( index < caConc_.size() );
	
	if ( index < caConc_.size() )
		return ca_[ index ];
	
	return 0.0;
}

void HSolveActive::setCa( unsigned int index, double value )
{
	assert( index < caConc_.size() );
	
	if ( index < caConc_.size() ) {
		ca_[ index ] = value;
		caConc_[ index ].c_ = value - caConc_[ index ].CaBasal_;
	}
}
