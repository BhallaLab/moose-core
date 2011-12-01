/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "biophysics/Compartment.h"
#include "biophysics/SpikeGen.h"
#include "biophysics/CaConc.h"
#include <queue>
#include "biophysics/SynInfo.h"
#include "biophysics/SynChan.h"
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"

extern ostream& operator <<( ostream& s, const HinesMatrix& m );

static const Finfo* synGkFinfo = initSynChanCinfo()->findFinfo( "Gk" );
static const Finfo* synEkFinfo = initSynChanCinfo()->findFinfo( "Ek" );
static const Finfo* spikeVmFinfo = initSpikeGenCinfo()->findFinfo( "Vm" );

const int HSolveActive::INSTANT_X = 1;
const int HSolveActive::INSTANT_Y = 2;
const int HSolveActive::INSTANT_Z = 4;

HSolveActive::HSolveActive()
{
	caAdvance_ = 1;
	
	// Default lookup table size
	vDiv_ = 3000;    // for voltage
	caDiv_ = 3000;   // for calcium
}

//////////////////////////////////////////////////////////////////////
// Solving differential equations
//////////////////////////////////////////////////////////////////////

void HSolveActive::solve( ProcInfo info ) {
	if ( !current_.size() ) {
		current_.resize( channel_.size() );
	}
	
	advanceChannels( info->dt_ );
	calculateChannelCurrents( );
	updateMatrix( );
	HSolvePassive::forwardEliminate( );
	HSolvePassive::backwardSubstitute( );
	advanceCalcium( );
	advanceSynChans( info );
	
	sendValues( );
	sendSpikes( info );
	
	externalCurrent_.assign( externalCurrent_.size(), 0.0 );
}

void HSolveActive::calculateChannelCurrents( ) {
	vector< ChannelStruct >::iterator ichan;
	vector< CurrentStruct >::iterator icurrent = current_.begin();
	
	if ( state_.size() != 0 ) {
		double* istate = &state_[ 0 ];
		
		for ( ichan = channel_.begin(); ichan != channel_.end(); ++ichan ) {
			ichan->process( istate, *icurrent );
			++icurrent;
		}
	}
}

void HSolveActive::updateMatrix( ) {
	/*
	 * Copy contents of HJCopy_ into HJ_. Cannot do a vector assign() because
	 * iterators to HJ_ get invalidated in MS VC++
	 */
	if ( HJ_.size() != 0 )
		memcpy( &HJ_[ 0 ], &HJCopy_[ 0 ], sizeof( double ) * HJ_.size() );
	
	double GkSum, GkEkSum;
	vector< CurrentStruct >::iterator icurrent = current_.begin();
	vector< currentVecIter >::iterator iboundary = currentBoundary_.begin();
	vector< double >::iterator ihs = HS_.begin();
	vector< double >::iterator iv = V_.begin();
	
	vector< CompartmentStruct >::iterator ic;
	for ( ic = compartment_.begin(); ic != compartment_.end(); ++ic ) {
		GkSum   = 0.0;
		GkEkSum = 0.0;
		for ( ; icurrent < *iboundary; ++icurrent ) {
			GkSum   += icurrent->Gk;
			GkEkSum += icurrent->Gk * icurrent->Ek;
		}
		
		*ihs = *( 2 + ihs ) + GkSum;
		*( 3 + ihs ) = *iv * ic->CmByDt + ic->EmByRm + GkEkSum;
		
		++iboundary, ihs += 4, ++iv;
	}
	
	map< unsigned int, InjectStruct >::iterator inject;
	for ( inject = inject_.begin(); inject != inject_.end(); ++inject ) {
		unsigned int ic = inject->first;
		InjectStruct& value = inject->second;
		
		HS_[ 4 * ic + 3 ] += value.injectVarying + value.injectBasal;
		
		value.injectVarying = 0.0;
	}
	
	double Gk, Ek;
	vector< SynChanStruct >::iterator isyn;
	for ( isyn = synchan_.begin(); isyn != synchan_.end(); ++isyn ) {
		get< double >( isyn->elm_, synGkFinfo, Gk );
		get< double >( isyn->elm_, synEkFinfo, Ek );
		
		unsigned int ic = isyn->compt_;
		HS_[ 4 * ic ] += Gk;
		HS_[ 4 * ic + 3 ] += Gk * Ek;
	}
	
	ihs = HS_.begin();
	for ( vector< double >::iterator
	      iec = externalCurrent_.begin();
	      iec != externalCurrent_.end();
	      iec += 2 )
	{
		*ihs += *iec;
		*( 3 + ihs ) += *( iec + 1 );
		
		ihs += 4;
	}
	
	stage_ = 0;    // Update done.
}

void HSolveActive::advanceCalcium( ) {
	vector< CaTractStruct >::iterator icatract = caTract_.begin();
	vector< CurrentStruct* >::iterator icasource = caSource_.begin();
	vector< double >::iterator icaactivation = caActivation_.begin();
	vector< double* >::iterator icatarget = caTarget_.begin();
	
	caActivation_.assign( caActivation_.size(), 0.0 );
	
	/*
	 * caAdvance_: This flag determines how current flowing into a calcium pool
	 * is computed. A value of 0 means that the membrane potential at the
	 * beginning of the time-step is used for the calculation. This is how
	 * GENESIS does its computations. A value of 1 means the membrane potential
	 * at the middle of the time-step is used. This is the correct way of
	 * integration, and is the default way.
	 */
	vector< double > v0;
	vector< double >::iterator iv;
	if ( caAdvance_ > 0 ) {
		iv = VMid_.begin();
	} else {
		v0.resize( nCompt_ );
		iv = v0.begin();
		
		/*
		 * Reconstructing Vm at the beginning of the time-step, by
		 * extrapolating backwards (using Vm at the middle and end of
		 * time-step).
		 */
		for ( unsigned int ic = 0; ic < nCompt_; ++ic )
			v0[ ic ] = ( 2 * VMid_[ ic ] - V_[ ic ] );
	}
	
	for ( ; icatract != caTract_.end(); ++icatract ) {
		switch( icatract->type )
		{
			case 0:
				iv += icatract->length;
				
				break;
			
			case 1:
				for ( unsigned int ic = 0;
					  ic < icatract->length;
					  ++ic )
				{
					*icaactivation +=
						( *icasource )->Gk *
						( ( *icasource )->Ek - *iv );
					
					++icasource, ++icaactivation, ++iv;
				}
				
				break;
			
			case 2:
				for ( unsigned int ic = 0;
					  ic < icatract->length;
					  ++ic )
				{
					unsigned int nConnections =
						icatract->nConnections[ ic ];
					
					for ( unsigned int iconn = 0;
						  iconn < nConnections;
						  ++iconn )
					{
						**icatarget +=
							( *icasource )->Gk *
							( ( *icasource )->Ek - *iv );
						
						++icasource, ++icatarget, ++iv;
					}
				}
				
				icaactivation += icatract->nPools;
				
				break;
			
			default: assert( 0 );
		}
	}
	
	vector< LookupRow >::iterator icarow = caRow_.begin();
	icaactivation = caActivation_.begin();
	for ( vector< CaConcStruct >::iterator
	      icaconc = caConc_.begin();
	      icaconc != caConc_.end();
	      ++icaconc )
	{
		icaconc->process( *icaactivation );
		
		caTable_.row( icaconc->ca_, *icarow );
		
		++icaactivation, ++icarow;
	}
}

void HSolveActive::advanceChannels( double dt ) {
	vector< double >::iterator iv;
	vector< double >::iterator istate = state_.begin();
	vector< int >::iterator ichannelcount = channelCount_.begin();
	vector< ChannelStruct >::iterator ichan = channel_.begin();
	vector< ChannelStruct >::iterator chanBoundary;
	vector< LookupColumn >::iterator icolumn = column_.begin();
	vector< LookupRow* >::iterator icarowchan = caRowChan_.begin();
	
	/*
	 * \TODO: replace channelCount_ with channelBoundary_.
	 */
	
	LookupRow vRow;
	double C1, C2;
	for ( iv = V_.begin(); iv != V_.end(); ++iv ) {
		vTable_.row( *iv, vRow );
		
		chanBoundary = ichan + *ichannelcount;
		for ( ; ichan < chanBoundary; ++ichan ) {
			if ( ichan->Xpower_ > 0.0 ) {
				vTable_.lookup( *icolumn, vRow, C1, C2 );
				//~ *istate = *istate * C1 + C2;
				//~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
				if ( ichan->instant_ & INSTANT_X )
					*istate = C1 / C2;
				else {
					double temp = 1.0 + dt / 2.0 * C2;
					*istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
				}
				
				++icolumn, ++istate;
			}
			
			if ( ichan->Ypower_ > 0.0 ) {
				vTable_.lookup( *icolumn, vRow, C1, C2 );
				//~ *istate = *istate * C1 + C2;
				//~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
				if ( ichan->instant_ & INSTANT_Y )
					*istate = C1 / C2;
				else {
					double temp = 1.0 + dt / 2.0 * C2;
					*istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
				}
				
				++icolumn, ++istate;
			}
			
			if ( ichan->Zpower_ > 0.0 ) {
				LookupRow* caRowChan = *icarowchan;
				if ( caRowChan ) {
					caTable_.lookup( *icolumn, *caRowChan, C1, C2 );
				} else {
					vTable_.lookup( *icolumn, vRow, C1, C2 );
				}
				
				//~ *istate = *istate * C1 + C2;
				//~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
				if ( ichan->instant_ & INSTANT_Z )
					*istate = C1 / C2;
				else {
					double temp = 1.0 + dt / 2.0 * C2;
					*istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
				}
				
				++icolumn, ++istate, ++icarowchan;
			}
		}
		
		++ichannelcount;
	}
}

/**
 * SynChans are currently not under solver's control
 */
void HSolveActive::advanceSynChans( ProcInfo info ) {
	return;
}

void HSolveActive::sendSpikes( ProcInfo info ) {
	vector< SpikeGenStruct >::iterator ispike;
	for ( ispike = spikegen_.begin(); ispike != spikegen_.end(); ++ispike ) {
		/* Scope resolution used here to resolve ambiguity between the "set"
		 * function (used here for setting element field values) which belongs
		 * in the global namespace, and the STL "set" container, which is in the
		 * std namespace.
		 */
		::set< double >( ispike->elm_, spikeVmFinfo, V_[ ispike->compt_ ] );
	}
}

/**
 * This function dispatches state values via any source messages on biophysical
 * objects which have been taken over.
 */
void HSolveActive::sendValues( ) {
	static const Slot compartmentVmSrcSlot =
		initCompartmentCinfo( )->getSlot( "VmSrc" );
	static const Slot caConcConcSrcSlot =
		initCaConcCinfo( )->getSlot( "concSrc" );
	static const Slot compartmentChannelVmSlot =
		initCompartmentCinfo( )->getSlot( "channel.Vm" );
	static const Slot compartmentImSrcSlot =
		initCompartmentCinfo( )->getSlot( "ImSrc" );
	
	for ( unsigned int i = 0; i < compartmentId_.size( ); ++i ) {
		send1< double > (
			compartmentId_[ i ].eref(),
			compartmentVmSrcSlot,
			V_[ i ]
		);
		send1< double > (
			compartmentId_[ i ].eref(),
			compartmentImSrcSlot,
			getIm( i )
		);
		// An advantage of sending from the compartment here is that we can use
		// as simple 'send' as opposed to 'sendTo'. sendTo requires the conn
		// index for the target, and that will require extra book keeping.
		// Disadvantage is that the message will go out to regular HHChannels,
		// etc. A possibility is to delete those messages.
		send1< double >(
			compartmentId_[ i ].eref(),
			compartmentChannelVmSlot,
			V_[ i ]
		);
	}
	
	/*
	 * Speed up this function by sending only from objects which have targets.
	 */
	for ( unsigned int i = 0; i < caConcId_.size( ); ++i )
		send1< double > (
			caConcId_[ i ].eref(),
			caConcConcSrcSlot,
			caConc_[ i ].ca_
		);
}
