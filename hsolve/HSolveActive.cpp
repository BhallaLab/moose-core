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
	
	// Default lookup table size and boundaries
	vDiv_ = 3000;    // for voltage
	//~ vMin_ = -0.100;
	//~ vMax_ = 0.050;
	caDiv_ = 3000;   // for calcium
	//~ caMin_ = 0.0;
	//~ caMax_ = 1000.0;
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
        sendSpikes( info );
        sendComptVm();
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
	for ( inject = inject_.begin(); inject != inject_.end(); inject++ ) {
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
	
	stage_ = 0;    // Update done.
}

void HSolveActive::advanceCalcium( ) {
	vector< double* >::iterator icatarget = caTarget_.begin();
	vector< double >::iterator ivmid = VMid_.begin();
	vector< CurrentStruct >::iterator icurrent = current_.begin();
	vector< currentVecIter >::iterator iboundary = currentBoundary_.begin();
	
	caActivation_.assign( caActivation_.size(), 0.0 );
	
	/*
	 * caAdvance_: This flag determines how current flowing into a calcium pool
	 * is computed. A value of 0 means that the membrane potential at the
	 * beginning of the time-step is used for the calculation. This is how
	 * GENESIS does its computations. A value of 1 means the membrane potential
	 * at the middle of the time-step is used. This is the correct way of
	 * integration, and is the default way.
	 */	
	if ( caAdvance_ == 1 ) {
		for ( ; iboundary != currentBoundary_.end(); ++iboundary ) {
			for ( ; icurrent < *iboundary; ++icurrent ) {
				if ( *icatarget )
					**icatarget += icurrent->Gk * ( icurrent->Ek - *ivmid );
				
				++icatarget;
			}
			
			++ivmid;
		}
	} else if ( caAdvance_ == 0 ) {
		vector< double >::iterator iv = V_.begin();
		double v0;
		
		for ( ; iboundary != currentBoundary_.end(); ++iboundary ) {
			for ( ; icurrent < *iboundary; ++icurrent ) {
				if ( *icatarget ) {
					v0 = ( 2 * *ivmid - *iv );
					
					**icatarget += icurrent->Gk * ( icurrent->Ek - v0 );
				}
				
				++icatarget;
			}
			
			++ivmid, ++iv;
		}
	}
	
	vector< CaConcStruct >::iterator icaconc;
	vector< double >::iterator icaactivation = caActivation_.begin();
	vector< double >::iterator ica = ca_.begin();
	for ( icaconc = caConc_.begin(); icaconc != caConc_.end(); ++icaconc ) {
		*ica = icaconc->process( *icaactivation );
		++ica, ++icaactivation;
	}
}

void HSolveActive::advanceChannels( double dt ) {
	vector< double >::iterator iv;
	vector< double >::iterator istate = state_.begin();
	vector< int >::iterator ichannelcount = channelCount_.begin();
	vector< ChannelStruct >::iterator ichan = channel_.begin();
	vector< ChannelStruct >::iterator chanBoundary;
	vector< unsigned int >::iterator icacount = caCount_.begin();
	vector< double >::iterator ica = ca_.begin();
	vector< double >::iterator caBoundary;
	vector< LookupColumn >::iterator icolumn = column_.begin();
	vector< LookupRow >::iterator icarowcompt;
	vector< LookupRow* >::iterator icarow = caRow_.begin();
	
	LookupRow vRow;
	double C1, C2;
	for ( iv = V_.begin(); iv != V_.end(); ++iv ) {
		vTable_.row( *iv, vRow );
		icarowcompt = caRowCompt_.begin();
		caBoundary = ica + *icacount;
		for ( ; ica < caBoundary; ++ica ) {
			caTable_.row( *ica, *icarowcompt );
			++icarowcompt;
		}
		
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
				LookupRow* caRow = *icarow;
				if ( caRow ) {
					caTable_.lookup( *icolumn, *caRow, C1, C2 );
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
				
				++icolumn, ++istate, ++icarow;
			}
		}
		
		++ichannelcount, ++icacount;
	}
}

/**
 * SynChans are currently not under solver's control
 */
void HSolveActive::advanceSynChans( ProcInfo info ) {
	return;
}

void HSolveActive::sendComptVm(){
    vector< SpikeGenStruct >::iterator ispike;
    for ( ispike = spikegen_.begin(); ispike != spikegen_.end(); ++ispike ) {
        cout << "%%% Sending VmSrc to " << ispike->elm_.id().path() << endl;
        send1<double>(getCompartments()[ispike->compt_].eref(), initCompartmentCinfo()->getSlot("VmSrc"), V_[ ispike->compt_ ]);
    }
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
