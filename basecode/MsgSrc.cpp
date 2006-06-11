/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

void dummyFunc0( Conn* c )
{
	;
}


///////////////////////////////////////////////////////////////////////
// 	NMsgSrc
///////////////////////////////////////////////////////////////////////
bool NMsgSrc::add( RecvFunc func , Conn* target )
{
/*
	vector< RecvFunc >::iterator i = 	
		find( rfuncs_.begin(), rfuncs_.end(), func );
	unsigned long j = static_cast< unsigned long >(
		i - rfuncs_.begin() );
		*/
	unsigned long j = indexOfMatchingFunc( func );

	if (c_->connect( target, j ) ) {
		if ( j == rfuncs_.size() ) // new function
			rfuncs_.push_back( func );
		return 1;
	}
	return 0;
}

// The shared connection has just been set up on a master message.
// Here we assign the subsidiary RecvFuncs.
void NMsgSrc::addRecvFunc( RecvFunc func, unsigned long position )
{
	if ( position < rfuncs_.size() ) {
		if ( rfuncs_[ position ] != func ) {
			cerr << "Error: NMsgSrc::addRecvFunc: Error: rfunc mismatch with existing func on\n";
			cerr << c_->parent()->path() << "\n";
		}
		return;
	}
	if ( position == rfuncs_.size() ) {
		rfuncs_.push_back( func );
	} else {
		cerr << "Error: NMsgSrc::addRecvFunc: position = " << position << " > rfuncs_.size() = " << rfuncs_.size() << " on\n";
		cerr << c_->parent()->path() << "\n";
	}

/*
	vector< RecvFunc >::iterator i = 	
		find( rfuncs_.begin(), rfuncs_.end(), func );
	if ( i == rfuncs_.end() ) // new function 
	{
		if ( rfuncs_.size() > 0 ) {
			cerr << "Error: NMsgSrc::addRecvFunc: Currently do not know how\nto handle different recvFuncs.\n";
			cerr << "i = " << i - rfuncs_.begin() << ", On " << c_->parent()->path() << "\n";
		} else {
			rfuncs_.push_back( func );
		}
	}
	*/
}

// Returns the index of the matching function in the src. If not
// found returns the size of the rfuncs vector.
unsigned long NMsgSrc::indexOfMatchingFunc( RecvFunc func ) const
{
	vector< RecvFunc >::const_iterator i = 	
		find( rfuncs_.begin(), rfuncs_.end(), func );
	return static_cast< unsigned long >( i - rfuncs_.begin() );
}

// Returns the number of matches.
unsigned long NMsgSrc::matchRemoteFunc( RecvFunc func ) const
{
	return 
		static_cast< unsigned long > (
			count( rfuncs_.begin(), rfuncs_.end(), func ) 
		);
}

// Returns target func corresponding to the specified msgno
// Note that this is NOT the index into the rfuncs_array.
RecvFunc NMsgSrc::targetFunc( unsigned long msgno ) const
{
	unsigned long i = c_->index( msgno );
	if ( rfuncs_.size() > i )
		return rfuncs_[ i ];
	cerr << "Error: NMsgSrc::targetFunc: i > size: " << i << " > " <<
		rfuncs_.size() << ", msgno = " << msgno << "\n";
	return 0;
}

void NMsgSrc::dest( vector< Field >& list )
{
	unsigned long i;
	vector< Conn* >::const_iterator j;
	for ( i = 0; i < rfuncs_.size(); i++ ) {
		for ( j = c_->begin( i ); j != c_->end( i ); j++ ) {
			list.push_back( 
				( *j )->parent()->lookupDestField( *j, rfuncs_[ i ] )
			);
		}
	}
}

//////////////////////////////////////////////////////////////////
// SingleMsgSrc stuff here
//////////////////////////////////////////////////////////////////
void SingleMsgSrc::dest( vector< Field >& list )
{
	list.push_back( c_->parent()->lookupDestField( c_, rfunc_ ));
}

void SingleMsgSrc::addRecvFunc( RecvFunc rf, unsigned long position )
{
	if ( position > 0 ) {
		cerr << "Error: SingleMsgSrc::addRecvFunc: position = " << position << " must be zero on\n";
		cerr << c_->parent()->path() << "\n";
		return;
	}
	if ( rfunc_ == 0 || rfunc_ == dummyFunc0 ) {
		rfunc_ = rf;
	} else {
		cerr << "Error: SingleMsgSrc::addRecvFunc: Error: rfunc mismatch with\n";
		cerr << "existing func on " << c_->parent()->path() << "\n";
	}
}
