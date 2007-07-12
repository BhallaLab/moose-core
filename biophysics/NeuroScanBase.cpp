/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "HSolveStructure.h"
#include "NeuroScanBase.h"
#include <stack>
#include <set>

void NeuroScanBase::initialize( unsigned int seed, double dt )
{
	dV_ = ( VHi_ - VLo_ ) / NDiv_;
	dt_ = dt;
	
	constructTree( seed );
	constructMatrix( );
	constructChannelDatabase( );
	constructLookupTables( );
}

void NeuroScanBase::constructTree( unsigned int seed ) {
	unsigned int parent;
	vector< unsigned int > child = neighbours( seed );
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
	compartment_.push_back( currentNode.self_ );
	
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
			compartment_.push_back( currentNode.self_ );
		} else {
			nodeStack.pop();
			if( !nodeStack.empty() ) {
				++( nodeStack.top().state_ );
				currentNode = nodeStack.top();
			}
		}
	
	N_ = compartment_.size( );
	reverse( compartment_.begin(), compartment_.end() );
	reverse( checkpoint_.begin(), checkpoint_.end() );
	for ( unsigned long icp = 0; icp < checkpoint_.size(); ++icp )
		checkpoint_[ icp ] = ( N_ - 1 ) - checkpoint_[ icp ];
}

void NeuroScanBase::constructMatrix( ) {
	M_.resize( 5 * N_ );
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
			field( compartment_[ ic ], "Vm", Vm );
			field( compartment_[ ic ], "Em", Em );
			field( compartment_[ ic ], "Cm", Cm );
			field( compartment_[ ic ], "Rm", Rm );
			field( compartment_[ ic ], "Ra", Ra );
			field( compartment_[ ic ], "inject", inject );
			
			if ( ic < N_ - 1 ) {
				parentIndex = ic < checkpoint ?
					      1 + ic :
					      checkpoint_[ ++icp ];
				field( compartment_[ parentIndex ],
				       "Ra", Ra_parent );
				R_inv = 2.0 / ( Ra + Ra_parent );
				M_[ 5 * parentIndex + 3 ] += R_inv;
			} else
				R_inv = 0.0;
			
			CmByDt_.push_back( CmByDt = 2.0 * Cm / dt_ );
			EmByRm_.push_back( Em / Rm );
			inject_.push_back( inject );
			V_.push_back( Vm );
			
			M_[ 1 + ia ]  = -R_inv;
			M_[ 2 + ia ]  = R_inv * R_inv;
			M_[ 3 + ia ] += R_inv + 1.0 / Rm + CmByDt;
		}
	}
}

void NeuroScanBase::constructChannelDatabase( ) {
	vector< unsigned int >::iterator icompt, ichan, igate;
	vector< unsigned int > channel, gate;
	double Gbar, Ek;
	double state, power;
	
	for ( icompt = compartment_.begin();
	      icompt != compartment_.end();
	      ++icompt ) {
		channel = channels( *icompt );
		channel_.insert( channel_.end(), channel.begin(), channel.end() );
		channelCount_.push_back( ( unsigned char )( channel.size( ) ) );
		gateCount_.push_back( '\0' );
		
		for ( ichan = channel.begin();
		      ichan != channel.end();
		      ++ichan ) {
			gate = gates( *ichan );
			gate_.insert( gate_.end(), gate.begin(), gate.end() );
			gateCount_.back() += gate.size();
			gateCount1_.push_back( gate.size() );
			
			field( *ichan, "Gbar", Gbar );
			field( *ichan, "Ek", Ek );
			Gbar_.push_back( Gbar );
			GbarEk_.push_back( Gbar * Ek );
			
			for ( igate = gate.begin();
			      igate != gate.end(); 
			      ++igate ) {
				field( *igate, "state", state );
				field( *igate, "power", power );
				state_.push_back( state );
				power_.push_back( power );
			}
		}
	}
}

void NeuroScanBase::constructLookupTables( ) {
	std::set< unsigned int >::iterator ifamily;
	std::set< unsigned int > family( gate_.begin(), gate_.end() );
	lookupBlocSize_ = 2 * family.size();
	
	vector< unsigned int >::iterator ig;
	for ( ig = gate_.begin(); ig != gate_.end(); ++ig )
		gateFamily_.push_back (
			static_cast< unsigned char > (
			/*!!!*/	distance( family.begin(),
				family.find( *ig ) ) ) );
	
	double A, B, Vm;
	lookup_.resize( NDiv_ * lookupBlocSize_ );
	vector< double >::iterator il = lookup_.begin();
	for ( int igrid = 0; igrid < NDiv_; ++igrid ) {
		Vm = VLo_ + igrid * dV_;
		for ( ifamily = family.begin(); ifamily != family.end(); ++ifamily ) {
			rates( *ifamily, Vm, A, B );
			
			/* Refine wrt roundoff error, if necessary */
			*( il++ ) = ( 2.0 - dt_ * B ) / ( 2.0 + dt_ * B );
			*( il++ ) = dt_ * A / ( 1.0 + dt_ * B / 2.0 );
		}
	}
}

void NeuroScanBase::concludeInit( ) {
//	compartment_.clear( );
//	gateInfo_.clear( );
}
