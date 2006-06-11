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

void cleanRelayFinfos( Element* e , Finfo* f ) {
	/*
	cerr << "Time to clean up empty RelayFinfo0 " <<
		e->name() << "." <<  f->name() << "\n";
		*/
	e->dropRelay( f );
}

//////////////////////////////////////////////////////////////////
// RelayFinfo functions
//////////////////////////////////////////////////////////////////

Finfo* RelayFinfo::respondToAdd( Element* e, const Finfo* sender )
{
	if ( isSameType( sender ) ) {
		return this;
	} else { // Maybe the original field knows what to do
		return f_->respondToAdd( e, sender );
	}
}

bool RelayFinfo::add( Element* e, Field& destfield, bool useSharedConn )
{
	Finfo* f = destfield.respondToAdd( this );
	if ( f ) {
		if ( useSharedConn == 0 ) {
			if ( outConn_.connect(
					f->inConn( destfield.getElement() ), 0
				) ) {
				rfunc_.push_back( f->recvFunc() );
				return 1;
			}
		} else {
			rfunc_.push_back( f->recvFunc() );
			return 1;
		}
	}
	return 0;
}

bool RelayFinfo::dropRelay( unsigned long index,  Conn* src )
{
	if ( src == &inConn_ ) // No function there to deal with.
		return 1;
	if ( rfunc_.size() > index ) {
		rfunc_.erase( rfunc_.begin() + index );
		return 1;
	}
	cerr << "Warning: dropRelay: " << name() <<
		": index out of range\n";
	return 0;
}

// Cleans up if there is nothing to connect to.
void RelayFinfo::handleEmptyConn( Conn* c )
{
	if ( inConn_.nTargets() == 0 && outConn_.nTargets() == 0 ) {
		cleanRelayFinfos( inConn_.parent() , this );
	}
}

void RelayFinfo::dest( vector< Field >& list, Element* e )
{
	vector< Conn* > conns;
	outConn_.listTargets( conns );
	unsigned long i;
	unsigned long max = rfunc_.size();
	if ( max > conns.size() )
		max = conns.size();
	for ( i = 0; i < max; i++ )
		list.push_back(
			conns[i]->parent()->lookupDestField(
			conns[ i ], rfunc_[ i ] )
		);

	innerFinfo()->dest( list, e );
}

// Here we always add it to the end of the list. We don't bother
// with the existing ones.
void RelayFinfo::addRecvFunc( Element* e, RecvFunc rf,
			unsigned long position )
{
	rfunc_.push_back( rf );
/*
	if ( position < rfuncs_.size() ) {
		if ( rfuncs_[ position ] != rf ) {
			cerr << "Error: RelayFinfo::addRecvFunc: Error: rfunc mismatch with existing func on\n"; 
			cerr << e->path() << "\n";
		}
		return;
	}
	if ( position == rfuncs_.size() ) {
		rfuncs_.push_back( rf );
	} else {
		cerr << "Error: RelayFinfo::addRecvFunc: position = " <<
			position <        < " > rfuncs_.size() = " <<
			rfuncs_.size() << " on\n";
		cerr << e->path() << "\n";
	}
	*/
}

//////////////////////////////////////////////////////////////////
// RelayFinfo0 functions
//////////////////////////////////////////////////////////////////

// Used to intercept incoming msg calls.
void RelayFinfo0::relayFunc( Conn* c )
{
	RelayConn* cr = dynamic_cast< RelayConn* >( c );
	if (cr) {
		RelayFinfo0* f = 
			dynamic_cast< RelayFinfo0 *>( cr->finfo() );
		if ( f ) {
			// Do the original recvfunc
			f->innerFinfo()->recvFunc()( c );
			// Then do the outgoing stuff if any
			for (unsigned long i = 0; 
				i < f->outConn( 0 )->nTargets(); i++){
				f->rfunc_[ i ]( f->outConn( 0 )->target( i ) );
			}
		}
	}
}
