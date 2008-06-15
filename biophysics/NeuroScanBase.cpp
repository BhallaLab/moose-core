/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <queue>
#include "SynInfo.h"
#include "RateLookup.h"
#include "HSolveStruct.h"
#include "NeuroScanBase.h"
#include <stack>
#include <set>

void NeuroScanBase::initialize( Id seed, double dt )
{
	dt_ = dt;
	
	readCompartments( seed );
	readChannels( );
	readGates( );
	readCalcium( );
	readSynapses( );
	createMatrix( );
	createLookupTables( );
	concludeInit( );
}

void NeuroScanBase::readCompartments( Id seed ) {
	Id parent;
	vector< Id > child = neighbours( seed );
	if ( child.size() != 1 )
		while ( !child.empty() ) {
			parent = seed;
			seed   = child[ 0 ];
			child  = children( seed, parent );
		}
	
	unsigned int label = 0;
	stack< CNode > nodeStack;
	CNode currentNode;
	currentNode.self_   = seed;
	currentNode.child_  = neighbours( currentNode.self_ );
	currentNode.state_  = 0;
	currentNode.label_  = label;
	nodeStack.push( currentNode );
	compartmentId_.push_back( currentNode.self_ );
	
	while ( !nodeStack.empty() )
		if ( currentNode.state_ < currentNode.child_.size() ) {
			++label;
			
			if ( currentNode.state_ >= 1 ) {
				checkpoint_.push_back( currentNode.label_ );
				checkpoint_.push_back( label );
			}
			
			currentNode.parent_ = currentNode.self_;
			currentNode.self_   = currentNode.child_[
				currentNode.state_ ];
			currentNode.child_  = children( currentNode.self_,
				currentNode.parent_ );
			currentNode.state_  = 0;
			currentNode.label_  = label;
			nodeStack.push( currentNode );
			compartmentId_.push_back( currentNode.self_ );
		} else {
			nodeStack.pop();
			if( !nodeStack.empty() ) {
				++( nodeStack.top().state_ );
				currentNode = nodeStack.top();
			}
		}
	
	N_ = compartmentId_.size( );
	//~ reverse( compartmentId_.begin(), compartmentId_.end() );
	reverse( checkpoint_.begin(), checkpoint_.end() );
	for ( unsigned long icp = 0; icp < checkpoint_.size(); ++icp )
		checkpoint_[ icp ] = ( N_ - 1 ) - checkpoint_[ icp ];
}

void NeuroScanBase::readChannels( ) {
	vector< Id >::iterator icompt;
	vector< Id >::iterator ichan;
	vector< Id > channelId;
	double Gbar, Ek;
	double X, Y, Z;
	double Xpower, Ypower, Zpower;
	
	for ( icompt = compartmentId_.begin();
	      icompt != compartmentId_.end();
	      ++icompt )
	{
		channelId = channels( *icompt );
reverse( channelId.begin(), channelId.end() );
		// todo: discard channels with Gbar = 0.0
		channelId_.insert( channelId_.end(), channelId.begin(), channelId.end() );
		channelCount_.push_back( ( unsigned char )( channelId.size( ) ) );
		
		for ( ichan = channelId.begin(); ichan != channelId.end(); ++ichan ) {
			channel_.resize( channel_.size() + 1 );
			ChannelStruct& channel = channel_.back();
			
			field( *ichan, "Gbar", Gbar );
			field( *ichan, "Ek", Ek );
			field( *ichan, "X", X );
			field( *ichan, "Y", Y );
			field( *ichan, "Z", Z );
			field( *ichan, "Xpower", Xpower );
			field( *ichan, "Ypower", Ypower );
			field( *ichan, "Zpower", Zpower );
			
			channel.Gbar_ = Gbar;
			channel.GbarEk_ = Gbar * Ek;
			channel.setPowers( Xpower, Ypower, Zpower );
			
			if ( Xpower )
				state_.push_back( X );
			if ( Ypower )
				state_.push_back( Y );
			if ( Zpower )
				state_.push_back( Z );
		}
	}
}

void NeuroScanBase::readGates( ) {
	vector< Id >::iterator ichan;
	unsigned int nGates;
	int useConcentration;
	for ( ichan = channelId_.begin(); ichan != channelId_.end(); ++ichan ) {
		nGates = gates( *ichan, gateId_ );
		gCaDepend_.insert( gCaDepend_.end(), nGates, 0 );
		field( *ichan, "useConcentration", useConcentration );
		if ( useConcentration )
			gCaDepend_.back() = 1;
	}
}

void NeuroScanBase::readCalcium( ) {
	CaConcStruct caConc;
	double Ca, CaBasal, tau, B;
	vector< Id > caConcId;
	vector< int > caTargetIndex;
	vector< int > caDependIndex;
	map< Id, int > caConcIndex;
	int nTarget, nDepend;
	vector< Id >::iterator iconc;
	
	for ( unsigned int ichan = 0; ichan < channel_.size(); ++ichan ) {
		caConcId.resize( 0 );
		
		nTarget = caTarget( channelId_[ ichan ], caConcId );
		if ( nTarget == 0 )
			// No calcium pools fed by this channel.
			// Signal this using bitwise complement of (unsigned int)(0)
			//~ caTargetIndex.push_back( ~( 0U ) );
			caTargetIndex.push_back( -1 );
		
		nDepend = caDepend( channelId_[ ichan ], caConcId );
		if ( nDepend == 0 )
			// Channel does not depend on calcium.
			// Signal this using bitwise complement of (unsigned int)(0)
			caDependIndex.push_back( -1 );
		
		if ( caConcId.size() == 0 )
			continue;
		
		for ( iconc = caConcId.begin(); iconc != caConcId.end(); ++iconc )
			if ( caConcIndex.find( *iconc ) == caConcIndex.end() ) {
				field( *iconc, "Ca", Ca );
				field( *iconc, "CaBasal", CaBasal );
				field( *iconc, "tau", tau );
				field( *iconc, "B", B );
				
				caConc.c_ = Ca - CaBasal;
				caConc.factor1_ = 4.0 / ( 2.0 + dt_ / tau ) - 1.0;
				caConc.factor2_ = 2.0 * B * dt_ / ( 2.0 + dt_ / tau );
				caConc.CaBasal_ = CaBasal;
				
				caConc_.push_back( caConc );
				caConcIndex[ *iconc ] = caConc_.size() - 1;
			}
		
		if ( nTarget != 0 )
			caTargetIndex.push_back( caConcIndex[ caConcId.front() ] );
		if ( nDepend != 0 )
			caDependIndex.push_back( caConcIndex[ caConcId.back() ] );
	}
	
	caTarget_.resize( channel_.size() );
	caDepend_.resize( channel_.size() );
	ca_.resize( caConc_.size() );
	caActivation_.resize( caConc_.size() );
	
	for ( unsigned int ichan = 0; ichan < channel_.size(); ++ichan ) {
		if ( caTargetIndex[ ichan ] == -1 )
			caTarget_[ ichan ] = 0;
		else
			caTarget_[ ichan ] = &caActivation_[ caTargetIndex[ ichan ] ];
		
		if ( caDependIndex[ ichan ] == -1 )
			caDepend_[ ichan ] = 0;
		else
			caDepend_[ ichan ] = &ca_[ caDependIndex[ ichan ] ];
	}
}

void NeuroScanBase::readSynapses( ) {
	Id spike;
	vector< Id > syn;
	vector< Id >::iterator isyn;
	SpikeGenStruct spikegen;
	SynChanStruct synchan;
	
	for ( unsigned int ic = 0; ic < N_; ++ic ) {
		syn = postsyn( compartmentId_[ ic ] );
		for ( isyn = syn.begin(); isyn != syn.end(); ++isyn ) {
			synchan.compt_ = ic;
			synchan.elm_ = ( *isyn )();
			synchanFields( *isyn, synchan );
			synchan_.push_back( synchan );
		}
		
		spike = presyn( compartmentId_[ ic ] );
		if ( spike.bad() )
			continue;
		
		spikegen.compt_ = ic;
		spikegen.elm_ = spike();
		field( spike, "threshold", spikegen.threshold_ );
		field( spike, "refractT", spikegen.refractT_ );
		field( spike, "state", spikegen.state_ );
		spikegen_.push_back( spikegen );
	}
}

void NeuroScanBase::createMatrix( ) {
	M_.resize( 5 * N_, 0.0 );
	VMid_.resize( N_ );
	double  Vm, Em, Cm, Rm,
		Ra, Ra_parent,
		R_inv, CmByDt, inject;
	
	unsigned long ia = 0, ic = 0;
	unsigned long checkpoint, parentIndex;
	for ( unsigned long icp = 0; icp < 1 + checkpoint_.size(); ++icp ) {
		checkpoint = icp < checkpoint_.size() ?
		             checkpoint_[ icp ] :
		             N_ - 1;
		
		for ( ; ic < 1 + checkpoint; ++ic, ia += 5 ) {
			// Read initVm instead of Vm, since we are at reset time
			field( compartmentId_[ ic ], "initVm", Vm );
			field( compartmentId_[ ic ], "Em", Em );
			field( compartmentId_[ ic ], "Cm", Cm );
			field( compartmentId_[ ic ], "Rm", Rm );
			field( compartmentId_[ ic ], "Ra", Ra );
			field( compartmentId_[ ic ], "inject", inject );
			
			if ( ic < N_ - 1 ) {
				parentIndex = ic < checkpoint ?
					      1 + ic :
					      checkpoint_[ ++icp ];
				field( compartmentId_[ parentIndex ],
				       "Ra", Ra_parent );
				R_inv = 2.0 / ( Ra + Ra_parent );
				M_[ 5 * parentIndex + 3 ] += R_inv;
			} else
				R_inv = 0.0;
			
			//~ CmByDt = dt_ / ( 2.0 * Cm );
			CmByDt = 2.0 * Cm / dt_;
			CmByDt_.push_back( CmByDt );
			EmByRm_.push_back( Em / Rm );
			inject_.push_back( inject );
			V_.push_back( Vm );
			
			M_[ 1 + ia ]  = -R_inv;
			M_[ 2 + ia ]  = R_inv * R_inv;
			M_[ 3 + ia ] += R_inv + 1.0 / Rm + CmByDt;
			//~ 
			//~ M_[ 1 + ia ]  = -R_inv / CmByDt;
			//~ M_[ 2 + ia ]  = R_inv * R_inv / CmByDt;
			//~ M_[ 3 + ia ] += R_inv + 1.0 / Rm + 1.0 / CmByDt;
			
			//~ M_[ 1 + ia ]  = -CmByDt * R_inv;
			//~ M_[ 2 + ia ]  = CmByDt * CmByDt * R_inv * R_inv;
			//~ M_[ 3 + ia ] += 1.0 + CmByDt * ( R_inv + 1.0 / Rm );
		}
	}
	
	for ( unsigned long ic = 0, ia = 0; ic < N_; ++ic, ia += 5 ) {
		//~ M_[ 1 + ia ] /= CmByDt_[ ic ];
		//~ M_[ 2 + ia ] /= CmByDt_[ ic ];
		//~ M_[ 3 + ia ] *= CmByDt_[ ic ];
	}
}

void NeuroScanBase::createLookupTables( ) {
	std::set< Id > caSet;
	std::set< Id > vSet;
	vector< Id > caGate;
	vector< Id > vGate;
	map< Id, unsigned int > caType;
	map< Id, unsigned int > vType;
	
	for ( unsigned int ig = 0; ig < gateId_.size(); ++ig )
		if ( gCaDepend_[ ig ] )
			caSet.insert( gateId_[ ig ] );
		else
			vSet.insert( gateId_[ ig ] );
	
	caGate.insert( caGate.end(), caSet.begin(), caSet.end() );
	vGate.insert( vGate.end(), vSet.begin(), vSet.end() );
	
	for ( unsigned int ig = 0; ig < caGate.size(); ++ig )
		caType[ caGate[ ig ] ] = ig;
	for ( unsigned int ig = 0; ig < vGate.size(); ++ig )
		vType[ vGate[ ig ] ] = ig;
	
	lookupGroup_.push_back(
		RateLookupGroup( caMin_, caMax_, caDiv_, caGate.size() ) );
	lookupGroup_.push_back(
		RateLookupGroup( vMin_, vMax_, vDiv_, vGate.size() ) );
	RateLookupGroup& caLookupGroup = lookupGroup_[ 0 ];
	RateLookupGroup& vLookupGroup = lookupGroup_[ 1 ];
	
	vector< double > grid;
	vector< double > A, B;
	vector< double >::iterator ia, ib;
	double a, b;
	
	// Calcium-dependent lookup tables
	if ( caGate.size() ) {
		grid.resize( 1 + caDiv_ );
		double dca = ( caMax_ - caMin_ ) / caDiv_;
		for ( int igrid = 0; igrid <= caDiv_; ++igrid )
			grid[ igrid ] = caMin_ + igrid * dca;
	}
	
	for ( unsigned int ig = 0; ig < caGate.size(); ++ig ) {
		rates( caGate[ ig ], grid, A, B );
		
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
		
		caLookupGroup.addTable( ig, A, B );
	}
	
	// Voltage-dependent lookup tables
	if ( vGate.size() ) {
		grid.resize( 1 + vDiv_ );
		double dv = ( vMax_ - vMin_ ) / vDiv_;
		for ( int igrid = 0; igrid <= vDiv_; ++igrid )
			grid[ igrid ] = vMin_ + igrid * dv;
	}
	
	for ( unsigned int ig = 0; ig < vGate.size(); ++ig ) {
		rates( vGate[ ig ], grid, A, B );
		
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
		
		vLookupGroup.addTable( ig, A, B );
	}
	
	lookup_.reserve( gateId_.size() );
	for ( unsigned int ig = 0; ig < gateId_.size(); ++ig )
		if ( gCaDepend_[ ig ] )
			lookup_.push_back( caLookupGroup.slice( caType[ gateId_[ ig ] ] ) );
		else
			lookup_.push_back( vLookupGroup.slice( vType[ gateId_[ ig ] ] ) );
}

void NeuroScanBase::concludeInit( ) {
//	compartmentId_.clear( );
}
