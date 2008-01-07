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
#include "HSolveStruct.h"
#include "HSolveBase.h"
#include <cmath>

#include "SpikeGen.h"

// This macro is used for linear interpolation during table lookup of rate
// constants. Will not be included in coming versions.
#define WT_AVG( A, B, FRACTION ) \
        ( ( A ) * ( 1.0 - ( FRACTION ) ) + ( B ) * ( FRACTION ) )

void HSolveBase::step( ProcInfo info ) {
	updateMatrix( );
	forwardEliminate( );
	backwardSubstitute( );
	advanceChannels( );
	advanceSynChans( info );
	sendSpikes( info );
}

// This function needs cleanup. Many immediate optimizations come in here.
void HSolveBase::updateMatrix( ) {
	double Gk, GkEk,
	       GkSum, GkEkSum;
	double state;
	
	unsigned char ichan;
	unsigned char igate;
	vector< unsigned char >::iterator icco = channelCount_.begin();
	vector< unsigned char >::iterator igco = gateCount1_.begin();
	vector< double >::iterator igbar   = Gbar_.begin();
	vector< double >::iterator igbarek = GbarEk_.begin();
	vector< double >::iterator ipower  = power_.begin();
	vector< double >::iterator istate  = state_.begin();
	vector< double >::iterator ia      = M_.begin();
	vector< double >::iterator iv      = V_.begin();
	vector< double >::iterator ialpha  = CmByDt_.begin();
	vector< double >::iterator iileak  = EmByRm_.begin();
	vector< double >::iterator iinject = inject_.begin();
	for ( unsigned long ic = 0; ic < N_; ++ic ) {
		GkSum   = 0.0;
		GkEkSum = 0.0;
		for ( ichan = 0; ichan < *icco; ++ichan, ++igco ) {
			Gk   = *( igbar++ );
			GkEk = *( igbarek++ );
			for ( igate = 0; igate < *igco; ++igate ) {
				state = pow( *( istate++ ), *( ipower++ ) );
				Gk   *= state;
				GkEk *= state;
			}
			GkSum   += Gk;
			GkEkSum += GkEk;
		}
		
		*ia         = *( 3 + ia ) + GkSum;
		*( 4 + ia ) = *iileak + *ialpha * *iv + GkEkSum + *iinject;
		++icco, ia += 5, ++ialpha, ++iv, ++iileak, ++iinject;
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
	vector< double > VMid( N_ );
	
	long ic = ( long )( N_ ) - 1;
	vector< double >::reverse_iterator ivmid = VMid.rbegin();
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
		
		*ivmid = ( *ia - *( 3 + ia ) * VMid[ *icp ] )
			 / *( 4 + ia );
		*iv    = 2 * *ivmid - *iv;
	}
	
	for ( ; ic >= 0; --ic, ++ivmid, ++iv, ia += 5 ) {
		*ivmid = ( *ia - *( 3 + ia ) * *( ivmid - 1 ) )
			 / *( 4 + ia );
		*iv    = 2 * *ivmid - *iv;
	}
}

void HSolveBase::advanceChannels( ) {
	vector< double >::iterator iv;
	vector< double >::iterator ibase;
	vector< double >::iterator ilookup;
	vector< double >::iterator istate = state_.begin();
	vector< unsigned char >::iterator ifamily = gateFamily_.begin();
	vector< unsigned char >::iterator igco = gateCount_.begin();
	double distance, fraction;
	unsigned char ig;
	for ( iv = V_.begin(); iv != V_.end(); ++iv, ++igco )
		if ( *igco ) {
			distance = ( *iv - VLo_ ) / dV_;
			ibase    = lookup_.begin() + lookupBlocSize_ *
				   ( unsigned long )( distance );
			fraction = distance - floor( distance );
			for ( ig = 0; ig < *igco; ++ig, ++istate ) {
				ilookup = ibase + 2 * *( ifamily++ );
				*istate = *istate * \
					WT_AVG( *ilookup, *( lookupBlocSize_ + ilookup ), fraction ) + \
					WT_AVG( *( 1 + ilookup ), *( ( 1 + lookupBlocSize_ ) + ilookup ), fraction );
			}
		}
}

void HSolveBase::advanceSynChans( ProcInfo info ) {
	vector< SynChanStruct >::iterator isyn;
	for ( isyn = synchan_.begin(); isyn != synchan_.end(); ++isyn )
		isyn->process( info );
}

void HSolveBase::sendSpikes( ProcInfo info ) {
	vector< SpikeGenStruct >::iterator ispike;
	for ( ispike = spikegen_.begin(); ispike != spikegen_.end(); ++ispike ) {
		set< double >( ispike->elm_, "Vm", V_[ ispike->compt_ ] );
		Conn c( ispike->elm_, 0 );
		SpikeGen::processFunc( c, info );
	}
}
