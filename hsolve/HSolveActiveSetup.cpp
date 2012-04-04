/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <set>
#include <limits> // Max and min 'double' values needed for lookup table init.
#include "../biophysics/CaConc.h"
#include "../biophysics/HHGate.h"
#include "../biophysics/ChanBase.h"
#include "../biophysics/HHChannel.h"
#include "HSolveUtils.h"
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"

//////////////////////////////////////////////////////////////////////
// Setup of data structures
//////////////////////////////////////////////////////////////////////

void HSolveActive::setup( Id seed, double dt ) {
	cout << ".. HA.setup()" << endl;
	
	this->HSolvePassive::setup( seed, dt );
	
	readHHChannels();
	readGates();
	readCalcium();
	createLookupTables();
	readSynapses();
	readExternalChannels();
	cleanup();
	
	cout << "# of compartments: " << compartmentId_.size() << "." << endl;
	cout << "# of channels: " << channelId_.size() << "." << endl;
	cout << "# of gates: " << gateId_.size() << "." << endl;
	cout << "# of states: " << state_.size() << "." << endl;
	cout << "# of Ca pools: " << caConc_.size() << "." << endl;
	cout << "# of SynChans: " << synchan_.size() << "." << endl;
	cout << "# of SpikeGens: " << spikegen_.size() << "." << endl;
}

void HSolveActive::readHHChannels() {
	vector< Id >::iterator icompt;
	vector< Id >::iterator ichan;
	int nChannel;
	double Gbar, Ek;
	double X, Y, Z;
	double Xpower, Ypower, Zpower;
	int instant;
	
	for ( icompt = compartmentId_.begin(); icompt != compartmentId_.end(); ++icompt )
	{
		nChannel = HSolveUtils::hhchannels( *icompt, channelId_ );
		
		// todo: discard channels with Gbar = 0.0
		channelCount_.push_back( nChannel );
		
		ichan = channelId_.end() - nChannel;
		for ( ; ichan != channelId_.end(); ++ichan ) {
			channel_.resize( channel_.size() + 1 );
			ChannelStruct& channel = channel_.back();
			
			current_.resize( current_.size() + 1 );
			CurrentStruct& current = current_.back();
			
			Gbar    = HSolveUtils::get< ChanBase, double >( *ichan, "Gbar" );
			Ek      = HSolveUtils::get< ChanBase, double >( *ichan, "Ek" );
			X       = HSolveUtils::get< HHChannel, double >( *ichan, "X" );
			Y       = HSolveUtils::get< HHChannel, double >( *ichan, "Y" );
			Z       = HSolveUtils::get< HHChannel, double >( *ichan, "Z" );
			Xpower  = HSolveUtils::get< HHChannel, double >( *ichan, "Xpower" );
			Ypower  = HSolveUtils::get< HHChannel, double >( *ichan, "Ypower" );
			Zpower  = HSolveUtils::get< HHChannel, double >( *ichan, "Zpower" );
			instant = HSolveUtils::get< HHChannel, int >( *ichan, "instant" );
			
			current.Ek = Ek;
			
			channel.Gbar_ = Gbar;
			channel.setPowers( Xpower, Ypower, Zpower );
			channel.instant_ = instant;
			
			/*
			 * Map channel index to state index. This is useful in the
			 * interface to find gate values.
			 */
			chan2state_.push_back( state_.size() );
			
			if ( Xpower > 0.0 )
				state_.push_back( X );
			if ( Ypower > 0.0 )
				state_.push_back( Y );
			if ( Zpower > 0.0 )
				state_.push_back( Z );
			
			/*
			 * Map channel index to compartment index. This is useful in the
			 * interface to generate channel Ik values (since we then need the
			 * compartment Vm).
			 */
			chan2compt_.push_back( icompt - compartmentId_.begin() );
		}
	}
	
	int nCumulative = 0;
	currentBoundary_.resize( nCompt_ );
	for ( unsigned int ic = 0; ic < nCompt_; ++ic ) {
		nCumulative += channelCount_[ ic ];
		currentBoundary_[ ic ] = current_.begin() + nCumulative;
	}
}

void HSolveActive::readGates() {
	vector< Id >::iterator ichan;
	unsigned int nGates;
	int useConcentration;
	for ( ichan = channelId_.begin(); ichan != channelId_.end(); ++ichan ) {
		nGates = HSolveUtils::gates( *ichan, gateId_ );
		gCaDepend_.insert( gCaDepend_.end(), nGates, 0 );
		useConcentration =
			HSolveUtils::get< HHChannel, int >( *ichan, "useConcentration" );
		if ( useConcentration )
			gCaDepend_.back() = 1;
	}
}

void HSolveActive::readCalcium() {
	CaConcStruct caConc;
	double Ca, CaBasal, tau, B;
	vector< Id > caConcId;
	vector< int > caTargetIndex;
	map< Id, int > caConcIndex;
	int nTarget, nDepend;
	vector< Id >::iterator iconc;
	
	caCount_.resize( nCompt_ );
	unsigned int ichan = 0;
	for ( unsigned int ic = 0; ic < nCompt_; ++ic ) {
		unsigned int chanBoundary = ichan + channelCount_[ ic ];
		unsigned int nCa = caConc_.size();
		
		for ( ; ichan < chanBoundary; ++ichan ) {
			caConcId.clear();
			
			nTarget = HSolveUtils::caTarget( channelId_[ ichan ], caConcId );
			if ( nTarget == 0 )
				// No calcium pools fed by this channel.
				caTargetIndex.push_back( -1 );
			
			nDepend = HSolveUtils::caDepend( channelId_[ ichan ], caConcId );
			if ( nDepend == 0 )
				// Channel does not depend on calcium.
				caDependIndex_.push_back( -1 );
			
			for ( iconc = caConcId.begin(); iconc != caConcId.end(); ++iconc )
				if ( caConcIndex.find( *iconc ) == caConcIndex.end() ) {
					caConcIndex[ *iconc ] = caCount_[ ic ];
					++caCount_[ ic ];
					
					Ca =
						HSolveUtils::get< CaConc, double >( *iconc, "Ca" );
					CaBasal =
						HSolveUtils::get< CaConc, double >( *iconc, "CaBasal" );
					tau =
						HSolveUtils::get< CaConc, double >( *iconc, "tau" );
					B =
						HSolveUtils::get< CaConc, double >( *iconc, "B" );
					
					caConc.c_ = Ca - CaBasal;
					caConc.factor1_ = 4.0 / ( 2.0 + dt_ / tau ) - 1.0;
					caConc.factor2_ = 2.0 * B * dt_ / ( 2.0 + dt_ / tau );
					caConc.CaBasal_ = CaBasal;
					
					caConc_.push_back( caConc );
					caConcId_.push_back( *iconc );
				}
			
			if ( nTarget != 0 )
				caTargetIndex.push_back( caConcIndex[ caConcId.front() ] + nCa );
			if ( nDepend != 0 )
				caDependIndex_.push_back( caConcIndex[ caConcId.back() ] );
		}
	}
	
	caTarget_.resize( channel_.size() );
	ca_.resize( caConc_.size() );
	caActivation_.resize( caConc_.size() );
	
	for ( unsigned int ichan = 0; ichan < channel_.size(); ++ichan ) {
		if ( caTargetIndex[ ichan ] == -1 )
			caTarget_[ ichan ] = 0;
		else
			caTarget_[ ichan ] = &caActivation_[ caTargetIndex[ ichan ] ];
	}
}

void HSolveActive::createLookupTables() {
	std::set< Id > caSet;
	std::set< Id > vSet;
	vector< Id > caGate;
	vector< Id > vGate;
	map< Id, unsigned int > gateSpecies;
	
	for ( unsigned int ig = 0; ig < gateId_.size(); ++ig )
		if ( gCaDepend_[ ig ] )
			caSet.insert( gateId_[ ig ] );
		else
			vSet.insert( gateId_[ ig ] );
	
	caGate.insert( caGate.end(), caSet.begin(), caSet.end() );
	vGate.insert( vGate.end(), vSet.begin(), vSet.end() );
	
	for ( unsigned int ig = 0; ig < caGate.size(); ++ig )
		gateSpecies[ caGate[ ig ] ] = ig;
	for ( unsigned int ig = 0; ig < vGate.size(); ++ig )
		gateSpecies[ vGate[ ig ] ] = ig;
	
	/*
	 * Finding the smallest xmin and largest xmax across all gates' lookup
	 * tables.
	 */
	vMin_ = numeric_limits< double >::max();
	vMax_ = numeric_limits< double >::min();
	caMin_ = numeric_limits< double >::max();
	caMax_ = numeric_limits< double >::min();
	
	double min;
	double max;
	
	for ( unsigned int ig = 0; ig < caGate.size(); ++ig ) {
		min = HSolveUtils::get< HHGate, double >( caGate[ ig ], "min" );
		max = HSolveUtils::get< HHGate, double >( caGate[ ig ], "max" );
		if ( min < caMin_ )
			caMin_ = min;
		if ( max > caMax_ )
			caMax_ = max;
	}
	
	for ( unsigned int ig = 0; ig < vGate.size(); ++ig ) {
		min = HSolveUtils::get< HHGate, double >( vGate[ ig ], "min" );
		max = HSolveUtils::get< HHGate, double >( vGate[ ig ], "max" );
		if ( min < vMin_ )
			vMin_ = min;
		if ( max > vMax_ )
			vMax_ = max;
	}
	
	caTable_ = LookupTable( caMin_, caMax_, caDiv_, caGate.size() );
	vTable_ = LookupTable( vMin_, vMax_, vDiv_, vGate.size() );
	
	vector< double > grid;
	vector< double > A, B;
	vector< double >::iterator ia, ib;
	double a, b;
	//~ int AMode, BMode;
	//~ bool interpolate;
	
	// Calcium-dependent lookup tables
	if ( !caGate.empty() ) {
		grid.resize( 1 + caDiv_ );
		double dca = ( caMax_ - caMin_ ) / caDiv_;
		for ( int igrid = 0; igrid <= caDiv_; ++igrid )
			grid[ igrid ] = caMin_ + igrid * dca;
	}
	
	for ( unsigned int ig = 0; ig < caGate.size(); ++ig ) {
		HSolveUtils::rates( caGate[ ig ], grid, A, B );
		//~ HSolveUtils::modes( caGate[ ig ], AMode, BMode );
		//~ interpolate = ( AMode == 1 ) || ( BMode == 1 );
		
		ia = A.begin();
		ib = B.begin();
		for ( unsigned int igrid = 0; igrid < grid.size(); ++igrid ) {
			a = *ia;
			b = *ib;
			
			// *ia = ( 2.0 - dt_ * b ) / ( 2.0 + dt_ * b );
			// *ib = dt_ * a / ( 1.0 + dt_ * b / 2.0 );
			// *ia = dt_ * a;
			// *ib = 1.0 + dt_ * b / 2.0;
			++ia, ++ib;
		}
		
		//~ caTable_.addColumns( ig, A, B, interpolate );
		caTable_.addColumns( ig, A, B );
	}
	
	// Voltage-dependent lookup tables
	if ( !vGate.empty() ) {
		grid.resize( 1 + vDiv_ );
		double dv = ( vMax_ - vMin_ ) / vDiv_;
		for ( int igrid = 0; igrid <= vDiv_; ++igrid )
			grid[ igrid ] = vMin_ + igrid * dv;
	}
	
	for ( unsigned int ig = 0; ig < vGate.size(); ++ig ) {
		HSolveUtils::rates( vGate[ ig ], grid, A, B );
		//~ HSolveUtils::modes( vGate[ ig ], AMode, BMode );
		//~ interpolate = ( AMode == 1 ) || ( BMode == 1 );
		
		ia = A.begin();
		ib = B.begin();
		for ( unsigned int igrid = 0; igrid < grid.size(); ++igrid ) {
			a = *ia;
			b = *ib;
			
			// *ia = ( 2.0 - dt_ * b ) / ( 2.0 + dt_ * b );
			// *ib = dt_ * a / ( 1.0 + dt_ * b / 2.0 );
			// *ia = dt_ * a;
			// *ib = 1.0 + dt_ * b / 2.0;
			++ia, ++ib;
		}
		
		//~ vTable_.addColumns( ig, A, B, interpolate );
		vTable_.addColumns( ig, A, B );
	}
	
	column_.reserve( gateId_.size() );
	for ( unsigned int ig = 0; ig < gateId_.size(); ++ig ) {
		unsigned int species = gateSpecies[ gateId_[ ig ] ];
		
		LookupColumn column;
		if ( gCaDepend_[ ig ] )
			caTable_.column( species, column );
		else
			vTable_.column( species, column );
		
		column_.push_back( column );
	}
	
	///////////////////!!!!!!!!!!
	unsigned int maxN = *( max_element( caCount_.begin(), caCount_.end() ) );
	caRowCompt_.resize( maxN );
	for ( unsigned int ichan = 0; ichan < channel_.size(); ++ichan ) {
		if ( channel_[ ichan ].Zpower_ > 0.0 ) {
			int index = caDependIndex_[ ichan ];
			if ( index == -1 )
				caRow_.push_back( 0 );
			else
				caRow_.push_back( &caRowCompt_[ index ] );
		}
	}
}

void HSolveActive::readSynapses() {
	vector< Id > spikeId;
	vector< Id > synId;
	vector< Id >::iterator syn;
	vector< Id >::iterator spike;
	SpikeGenStruct spikegen;
	SynChanStruct synchan;
	
	for ( unsigned int ic = 0; ic < nCompt_; ++ic ) {
		synId.clear();
		HSolveUtils::synchans( compartmentId_[ ic ], synId );
		for ( syn = synId.begin(); syn != synId.end(); ++syn ) {
			synchan.compt_ = ic;
			synchan.elm_ = *syn;
			synchan_.push_back( synchan );
		}
		
		spikeId.clear();
		HSolveUtils::spikegens( compartmentId_[ ic ], spikeId );
		// Very unlikely that there will be >1 spikegens in a compartment,
		// but lets take care of it anyway.
		for ( spike = spikeId.begin(); spike != spikeId.end(); ++spike ) {
			spikegen.compt_ = ic;
			spikegen.elm_ = *spike;
			spikegen_.push_back( spikegen );
		}
	}
}

void HSolveActive::readExternalChannels() {
	vector< string > filter;
	filter.push_back( "HHChannel" );
	filter.push_back( "SynChan" );
	
	externalChannelId_.resize( compartmentId_.size() );
	externalCurrent_.resize( 2 * compartmentId_.size(), 0.0 );
	
	for ( unsigned int ic = 0; ic < compartmentId_.size(); ++ic )
		HSolveUtils::targets(
			compartmentId_[ ic ],
			"channel",
			externalChannelId_[ ic ],
			filter,
			false    // include = false. That is, use filter to exclude.
		);
}

void HSolveActive::cleanup() {
//	compartmentId_.clear();
	gCaDepend_.clear();
	caDependIndex_.clear();
}
