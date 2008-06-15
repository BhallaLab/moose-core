/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <queue>
#include "SynInfo.h"
#include "RateLookup.h"
#include "HSolveStruct.h"
#include "HSolveBase.h"
#include <cmath>

#include "SpikeGen.h"

void HSolveBase::step( ProcInfo info ) {
	if ( !Gk_.size() ) {
		Gk_.resize( channel_.size() );
		GkEk_.resize( channel_.size() );
	}

	//~ advanceChannels( );
	advanceChannels( info->dt_ );
	calculateChannelCurrents( );
	updateMatrix( );
	forwardEliminate( );
	backwardSubstitute( );
	advanceCalcium( );
	advanceSynChans( info );
	sendSpikes( info );
}

void HSolveBase::calculateChannelCurrents( ) {
	vector< ChannelStruct >::iterator ichan;
	vector< double >::iterator igk = Gk_.begin();
	vector< double >::iterator igkek = GkEk_.begin();
	double* istate = &state_[ 0 ];
	
	for ( ichan = channel_.begin(); ichan != channel_.end(); ++ichan ) {
		ichan->process( istate, *igk, *igkek );
		++igk, ++igkek;
	}
}

void HSolveBase::updateMatrix( ) {
	double GkSum, GkEkSum;
	
	unsigned char ichan;
	vector< unsigned char >::iterator icco = channelCount_.begin();
	vector< double >::iterator ia      = M_.begin();
	vector< double >::iterator iv      = V_.begin();
	vector< double >::iterator icmbydt = CmByDt_.begin();
	vector< double >::iterator iembyrm = EmByRm_.begin();
	vector< double >::iterator iinject = inject_.begin();
	vector< double >::iterator igk     = Gk_.begin();
	vector< double >::iterator igkek   = GkEk_.begin();
	for ( unsigned long ic = 0; ic < N_; ++ic ) {
		GkSum   = 0.0;
		GkEkSum = 0.0;
		for ( ichan = 0; ichan < *icco; ++ichan ) {
			GkSum   += *( igk++ );
			GkEkSum += *( igkek++ );
		}
		
		*ia         = *( 3 + ia ) + GkSum;
		*( 4 + ia ) = *iembyrm + *icmbydt * *iv + GkEkSum + *iinject;
		//~ *ia         = *( 3 + ia ) + *icmbydt * GkSum;
		//~ *( 4 + ia ) = *iv + *icmbydt * (*iembyrm + GkEkSum + *iinject);
		++icco, ia += 5, ++icmbydt, ++iv, ++iembyrm, ++iinject;
	}
	
	unsigned int ic;
	vector< SynChanStruct >::iterator isyn;
	for ( isyn = synchan_.begin(); isyn != synchan_.end(); ++isyn ) {
		ic = isyn->compt_;
		M_[ 5 * ic ] += isyn->Gk_;
		M_[ 5 * ic + 4 ] += isyn->Gk_ * isyn->Ek_;
	}
}

void HSolveBase::forwardEliminate( ) {
	vector< double >::iterator ia = M_.begin();
	vector< unsigned long >::iterator icp;
	unsigned long ic = 0;
	for ( icp = checkpoint_.begin(); icp != checkpoint_.end();
	      ++icp, ++ic, ia += 5 ) {
		for ( ; ic < *icp; ++ic, ia += 5 ) {
			*( 5 + ia ) -= *( 2 + ia ) / *ia;
			*( 9 + ia ) -= *( 1 + ia ) * *( 4 + ia ) / *ia;
		}
		M_[ 5 * *++icp ]    -= *( 2 + ia ) / *ia;
		M_[ 4 + 5 * *icp ]  -= *( 1 + ia ) * *( 4 + ia ) / *ia;
	}
	
	for ( ; ic < N_ - 1; ++ic, ia += 5 ) {
		*( 5 + ia ) -= *( 2 + ia ) / *ia;
		*( 9 + ia ) -= *( 1 + ia ) * *( 4 + ia ) / *ia;
	}
}

void HSolveBase::backwardSubstitute( ) {
	long ic = ( long )( N_ ) - 1;
	vector< double >::reverse_iterator ivmid = VMid_.rbegin();
	vector< double >::reverse_iterator iv = V_.rbegin();
	vector< double >::reverse_iterator ia = M_.rbegin();
	vector< unsigned long >::reverse_iterator icp;
	
	*ivmid = *ia / *( 4 + ia );
	*iv    = 2 * *ivmid - *iv;
	--ic, ++ivmid, ++iv, ia += 5;
	
	for ( icp = checkpoint_.rbegin();
	      icp != checkpoint_.rend();
	      icp += 2, --ic, ++ivmid, ++iv, ia += 5 ) {
		for ( ; ic > ( long )( *(1 + icp) );
		      --ic, ++ivmid, ++iv, ia += 5 ) {
			*ivmid = ( *ia - *( 3 + ia ) * *( ivmid - 1 ) )
				 / *( 4 + ia );
			*iv    = 2 * *ivmid - *iv;
		}
		
		*ivmid = ( *ia - *( 3 + ia ) * VMid_[ *icp ] )
			 / *( 4 + ia );
		*iv    = 2 * *ivmid - *iv;
	}
	
	for ( ; ic >= 0; --ic, ++ivmid, ++iv, ia += 5 ) {
		*ivmid = ( *ia - *( 3 + ia ) * *( ivmid - 1 ) )
			 / *( 4 + ia );
		*iv    = 2 * *ivmid - *iv;
	}
}

void HSolveBase::advanceCalcium( ) {
	unsigned char ichan;
	vector< double* >::iterator icatarget = caTarget_.begin();
	vector< double >::iterator igk = Gk_.begin();
	vector< double >::iterator igkek = GkEk_.begin();
	vector< double >::iterator ivmid = VMid_.begin();
	vector< unsigned char >::iterator icco;
	
	caActivation_.assign( caActivation_.size(), 0.0 );
	
double v;
vector< double >::iterator iv = V_.begin();
	for ( icco = channelCount_.begin(); icco != channelCount_.end(); ++icco, ++ivmid,
	++iv )
		for ( ichan = 0; ichan < *icco; ++ichan, ++icatarget, ++igk, ++igkek )
		{
			v = 2 * *ivmid - *iv;
			if ( *icatarget )
				**icatarget += *igkek - *igk * v;
				//~ **icatarget += *igkek - *igk * *ivmid;
		}
	
	vector< CaConcStruct >::iterator icaconc;
	vector< double >::iterator icaactivation = caActivation_.begin();
	vector< double >::iterator ica = ca_.begin();
	for ( icaconc = caConc_.begin(); icaconc != caConc_.end(); ++icaconc )
	{
		*ica = icaconc->process( *icaactivation );
		++ica, ++icaactivation;
	}
}

void HSolveBase::advanceChannels( double dt ) {
	vector< double >::iterator iv;
	vector< double >::iterator istate = state_.begin();
	vector< double* >::iterator icadepend = caDepend_.begin();
	vector< RateLookup >::iterator ilookup = lookup_.begin();
	vector< unsigned char >::iterator icco = channelCount_.begin();
	
	LookupKey key;
	LookupKey keyCa;
	double C1, C2;
	vector< ChannelStruct >::iterator ichan = channel_.begin();
	vector< ChannelStruct >::iterator nextChan;
	for ( iv = V_.begin(); iv != V_.end(); ++iv, ++icco ) {
		if ( *icco == 0 )
			continue;
		
		ilookup->getKey( *iv, key );
		nextChan = ichan + *icco;
		for ( ; ichan < nextChan; ++ichan, ++icadepend ) {
			if ( ichan->Xpower_ ) {
				ilookup->rates( key, C1, C2 );
				//~ *istate = *istate * C1 + C2;
				//~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
				double temp = 1.0 + dt / 2.0 * C2;
				*istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
				
				++ilookup, ++istate;
			}
			
			if ( ichan->Ypower_ ) {
				ilookup->rates( key, C1, C2 );
				//~ *istate = *istate * C1 + C2;
				//~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
				double temp = 1.0 + dt / 2.0 * C2;
				*istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
				
				++ilookup, ++istate;
			}
			
			if ( ichan->Zpower_ ) {
				if ( *icadepend ) {
					ilookup->getKey( **icadepend, keyCa );
					ilookup->rates( keyCa, C1, C2 );
				} else
					ilookup->rates( key, C1, C2 );
				
				//~ *istate = *istate * C1 + C2;
				//~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
				double temp = 1.0 + dt / 2.0 * C2;
				*istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
				
				++ilookup, ++istate;
			}
		}
	}
}

void HSolveBase::advanceSynChans( ProcInfo info ) {
	vector< SynChanStruct >::iterator isyn;
	for ( isyn = synchan_.begin(); isyn != synchan_.end(); ++isyn )
		isyn->process( info );
}

void HSolveBase::sendSpikes( ProcInfo info ) {
/*	vector< SpikeGenStruct >::iterator ispike;
	for ( ispike = spikegen_.begin(); ispike != spikegen_.end(); ++ispike ) {
		set< double >( ispike->elm_, "Vm", V_[ ispike->compt_ ] );
		Conn c( ispike->elm_, 0 );
		SpikeGen::processFunc( c, info );
	}*/
}
