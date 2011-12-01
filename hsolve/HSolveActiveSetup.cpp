/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <set>
#include <limits>	// Max and min 'double' values needed for lookup table init.
#include "biophysics/BioScan.h"
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"

//////////////////////////////////////////////////////////////////////
// Setup of data structures
//////////////////////////////////////////////////////////////////////

void HSolveActive::setup( Id seed, double dt ) {
	this->HSolvePassive::setup( seed, dt );
	
	readHHChannels( );
	readGates( );
	readCalcium( );
	createLookupTables( );
	readSynapses( );
	readExternalChannels( );
	cleanup( );
}

void HSolveActive::readHHChannels( ) {
	vector< Id >::iterator icompt;
	vector< Id >::iterator ichan;
	int nChannel;
	double Gbar, Ek;
	double X, Y, Z;
	double Xpower, Ypower, Zpower;
	int instant;
	
	for ( icompt = compartmentId_.begin(); icompt != compartmentId_.end(); ++icompt )
	{
		nChannel = BioScan::hhchannels( *icompt, channelId_ );
		
		// todo: discard channels with Gbar = 0.0
		channelCount_.push_back( nChannel );
		
		ichan = channelId_.end() - nChannel;
		for ( ; ichan != channelId_.end(); ++ichan ) {
			channel_.resize( channel_.size() + 1 );
			ChannelStruct& channel = channel_.back();
			
			current_.resize( current_.size() + 1 );
			CurrentStruct& current = current_.back();
			
			Eref elm = ( *ichan )();
			get< double >( elm, "Gbar", Gbar );
			get< double >( elm, "Ek", Ek );
			get< double >( elm, "X", X );
			get< double >( elm, "Y", Y );
			get< double >( elm, "Z", Z );
			get< double >( elm, "Xpower", Xpower );
			get< double >( elm, "Ypower", Ypower );
			get< double >( elm, "Zpower", Zpower );
			get< int >( elm, "instant", instant );
			
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

void HSolveActive::readGates( ) {
	vector< Id >::iterator ichan;
	unsigned int nGates;
	int useConcentration;
	for ( ichan = channelId_.begin(); ichan != channelId_.end(); ++ichan ) {
		nGates = BioScan::gates( *ichan, gateId_ );
		gCaDepend_.insert( gCaDepend_.end(), nGates, 0 );
		Eref elm = ( *ichan )();
		get< int >( elm, "useConcentration", useConcentration );
		if ( useConcentration )
			gCaDepend_.back() = 1;
	}
}

namespace hsolve{
	/*
	 * This struct is used in HSolveActive::readCalcium. Since it is
	 * used as a template argument (vector< CaInfo >), the standard does
	 * not allow the struct to be placed locally inside the function.
	 * Hence placing it outside the function, but in a separate
	 * namespace to avoid polluting the global namespace.
	 * 
	 * The structure holds information about Calcium-channel
	 * interactions in a compartment.
	 */
	struct CaInfo
	{
		vector< unsigned int > sourceChannelIndex;
		vector< Id >           targetCaId;
	};
}

void HSolveActive::readCalcium( ) {
	/* Stage 1 */
	caDependIndex_.resize( channel_.size(), -1 );
	vector< hsolve::CaInfo > caInfo( nCompt_ );
	map< Id, unsigned int > caIndex;
	
	unsigned int ichan = 0;
	for ( unsigned int ic = 0; ic < nCompt_; ++ic ) {
		unsigned int chanBoundary = ichan + channelCount_[ ic ];
		for ( ; ichan < chanBoundary; ++ichan ) {
			vector< Id > caTarget;
			vector< Id > caDepend;
			
			BioScan::caTarget( channelId_[ ichan ], caTarget );
			for ( vector< Id >::iterator
			      ica = caTarget.begin();
			      ica != caTarget.end();
			      ++ica )
			{
				caInfo[ ic ].sourceChannelIndex.push_back( ichan );
				caInfo[ ic ].targetCaId.push_back( *ica );
				
				if ( caIndex.find( *ica ) == caIndex.end() ) {
					caIndex[ *ica ] = caConcId_.size();
					caConcId_.push_back( *ica );
				}
			}
			
			BioScan::caDepend( channelId_[ ichan ], caDepend );
			
			if ( channel_[ ichan ].Zpower_ == 0.0 ) {
				if ( caDepend.size() > 0 ) {
					cerr << "Warning!" << endl;
				}
			} else if ( caDepend.size() > 0 ) {
				if ( caDepend.size() > 1 ) {
					cerr << "Warning!" << endl;
				}
				
				Id& ca0 = caDepend.front();
				if ( caIndex.find( ca0 ) == caIndex.end() ) {
					caIndex[ ca0 ] = caConcId_.size();
					caConcId_.push_back( ca0 );
				}
				
				caDependIndex_[ ichan ] = caIndex[ ca0 ];
			}
		}
	}
	
	/* Stage 2 */
	caActivation_.resize( caConcId_.size() );
	
	for ( vector< hsolve::CaInfo >::iterator
	      icainfo = caInfo.begin();
	      icainfo != caInfo.end();
	      ++icainfo )
	{
		unsigned int nConnections = icainfo->sourceChannelIndex.size();
		
		unsigned int type;
		switch ( nConnections )
		{
			case 0:  type = 0; break;
			case 1:  type = 1; break;
			default: type = 2; break;
		}
		
		if ( caTract_.empty() || caTract_.back().type != type )
			caTract_.push_back( CaTractStruct() );
		CaTractStruct& caTract = caTract_.back();
		
		caTract.type = type;
		caTract.length++;
		
		if ( nConnections == 0 )
			continue;
		
		caTract.nConnections.push_back( nConnections );
		
		for ( unsigned int iconn = 0; iconn < nConnections; ++iconn ) {
			CurrentStruct& sourceChannel =
				current_[ icainfo->sourceChannelIndex[ iconn ] ];
			caSource_.push_back( &sourceChannel );
			
			if ( type == 2 ) {
				Id targetCaId = icainfo->targetCaId[ iconn ];
				caTarget_.push_back(
					&caActivation_[ caIndex[ targetCaId ] ]
				);
			}
		}
	}
	
	/* Stage 3 */
	double Ca;
	double CaBasal;
	double tau;
	double B;
	for ( vector< Id >::iterator
	      ica = caConcId_.begin();
	      ica != caConcId_.end();
	      ++ica )
	{
		Eref elm = ( *ica )();
		get< double >( elm, "Ca", Ca );
		get< double >( elm, "CaBasal", CaBasal );
		get< double >( elm, "tau", tau );
		get< double >( elm, "B", B );
		
		CaConcStruct caConc;
		caConc.c_ = Ca - CaBasal;
		caConc.ca_ = Ca;
		caConc.factor1_ = 4.0 / ( 2.0 + dt_ / tau ) - 1.0;
		caConc.factor2_ = 2.0 * B * dt_ / ( 2.0 + dt_ / tau );
		caConc.CaBasal_ = CaBasal;
		
		caConc_.push_back( caConc );
	}
}

void HSolveActive::createLookupTables( ) {
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
		BioScan::domain( caGate[ ig ], min, max );
		if ( min < caMin_ )
			caMin_ = min;
		if ( max > caMax_ )
			caMax_ = max;
	}
	
	for ( unsigned int ig = 0; ig < vGate.size(); ++ig ) {
		BioScan::domain( vGate[ ig ], min, max );
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
	int AMode, BMode;
	bool interpolate;
	
	// Calcium-dependent lookup tables
	if ( caGate.size() ) {
		grid.resize( 1 + caDiv_ );
		double dca = ( caMax_ - caMin_ ) / caDiv_;
		for ( int igrid = 0; igrid <= caDiv_; ++igrid )
			grid[ igrid ] = caMin_ + igrid * dca;
	}
	
	for ( unsigned int ig = 0; ig < caGate.size(); ++ig ) {
		BioScan::rates( caGate[ ig ], grid, A, B );
		BioScan::modes( caGate[ ig ], AMode, BMode );
		interpolate = ( AMode == 1 ) || ( BMode == 1 );
		
		ia = A.begin();
		ib = B.begin();
		for ( unsigned int igrid = 0; igrid < grid.size(); ++igrid ) {
			a = *ia;
			b = *ib;
			
			//~ *ia = ( 2.0 - dt_ * b ) / ( 2.0 + dt_ * b );
			//~ *ib = dt_ * a / ( 1.0 + dt_ * b / 2.0 );
			//~ *ia = dt_ * a;
			//~ *ib = 1.0 + dt_ * b / 2.0;
			++ia, ++ib;
		}
		
		caTable_.addColumns( ig, A, B, interpolate );
	}
	
	// Voltage-dependent lookup tables
	if ( vGate.size() ) {
		grid.resize( 1 + vDiv_ );
		double dv = ( vMax_ - vMin_ ) / vDiv_;
		for ( int igrid = 0; igrid <= vDiv_; ++igrid )
			grid[ igrid ] = vMin_ + igrid * dv;
	}
	
	for ( unsigned int ig = 0; ig < vGate.size(); ++ig ) {
		BioScan::rates( vGate[ ig ], grid, A, B );
		BioScan::modes( vGate[ ig ], AMode, BMode );
		interpolate = ( AMode == 1 ) || ( BMode == 1 );
		
		ia = A.begin();
		ib = B.begin();
		for ( unsigned int igrid = 0; igrid < grid.size(); ++igrid ) {
			a = *ia;
			b = *ib;
			
			//~ *ia = ( 2.0 - dt_ * b ) / ( 2.0 + dt_ * b );
			//~ *ib = dt_ * a / ( 1.0 + dt_ * b / 2.0 );
			//~ *ia = dt_ * a;
			//~ *ib = 1.0 + dt_ * b / 2.0;
			++ia, ++ib;
		}
		
		vTable_.addColumns( ig, A, B, interpolate );
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
	
	/*
	 * Setting up caRow_: Resizing it appropriately, and initializing
	 * it with appropriate values for the first time-step in HSolve.
	 * 
	 * Later HSolve takes care of updating caRow_ with new values as
	 * Ca values evolve.
	 */
	caRow_.resize( caConc_.size() );
	vector< CaConcStruct >::iterator icaconc = caConc_.begin();
	for ( vector< LookupRow >::iterator
	      icarow = caRow_.begin();
	      icarow != caRow_.end();
	      ++icarow )
	{
		caTable_.row( icaconc->ca_, *icarow );
		
		++icaconc;
	}
	
	for ( unsigned int ichan = 0; ichan < channel_.size(); ++ichan ) {
		if ( channel_[ ichan ].Zpower_ > 0.0 ) {
			int index = caDependIndex_[ ichan ];
			if ( index == -1 )
				caRowChan_.push_back( 0 );
			else
				caRowChan_.push_back( &caRow_[ index ] );
		}
	}
}

void HSolveActive::readSynapses( ) {
	vector< Id > spikeId;
	vector< Id > synId;
	vector< Id >::iterator syn;
	vector< Id >::iterator spike;
	SpikeGenStruct spikegen;
	SynChanStruct synchan;
	
	for ( unsigned int ic = 0; ic < nCompt_; ++ic ) {
		synId.clear( );
		BioScan::synchans( compartmentId_[ ic ], synId );
		for ( syn = synId.begin(); syn != synId.end(); ++syn ) {
			synchan.compt_ = ic;
			synchan.elm_ = ( *syn )();
			synchan_.push_back( synchan );
		}
		
		spikeId.clear( );
		BioScan::spikegens( compartmentId_[ ic ], spikeId );
		// Very unlikely that there will be >1 spikegens in a compartment,
		// but lets take care of it anyway.
		for ( spike = spikeId.begin(); spike != spikeId.end(); ++spike ) {
			spikegen.compt_ = ic;
			spikegen.elm_ = ( *spike )();
			spikegen_.push_back( spikegen );
		}
	}
}

void HSolveActive::readExternalChannels( ) {
	vector< string > include;
	vector< string > exclude;
	exclude.push_back( "HHChannel" );
	exclude.push_back( "SynChan" );
	
	externalChannelId_.resize( compartmentId_.size() );
	externalCurrent_.resize( 2 * compartmentId_.size(), 0.0 );
	
	for ( unsigned int ic = 0; ic < compartmentId_.size(); ++ic )
		BioScan::targets(
			compartmentId_[ ic ],
			"channel",
			externalChannelId_[ ic ],
			include,
			exclude
		);
}

void HSolveActive::cleanup( ) {
//	compartmentId_.clear( );
	gCaDepend_.clear();
	caDependIndex_.clear();
}
