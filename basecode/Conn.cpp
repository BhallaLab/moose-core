/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

const unsigned long Conn::MAX = ~0;

// Need to make this thread-safe. Crude way is to put a lock around
// the entire function, but actually only need to protect a specific
// target.
// Connect is always from the source to the target. The source may
// optionally have a nonzero slot argument to specify which
// function is using the connection.
// Added later: the target may also need a slot specification, 
// for the case where it is a shared message.
bool Conn::connect( Conn* target, 
	unsigned long sourceSlot, unsigned long targetSlot )
{
	if (target == this) {
		cerr << "Warning: conn::connect() ignored attempt to send msg to self: " << parent()->name() << "\n";
		return 0;
	}
	if ( this->canConnect(target) ) {
		Conn* tgt = target->respondToConnect( this );
		if ( tgt ) {
			tgt->innerConnect(this, targetSlot);
			this->innerConnect(tgt, sourceSlot);
			return 1;
		}
	}
	cerr << "Warning: conn::connect() failed\n";
	return 0;
}

// Returns the index of the target on the current Conn. Returns
// MAX if not found.
unsigned long Conn::disconnect(Conn* target)
{
	unsigned long i = this->find(target);
	unsigned long j = target->find(this);
	if (i != MAX && j != MAX) {
		this->innerDisconnect(i);
		// later
		// target->cleanMsgSrcs();
		target->innerDisconnect(j);
		return i;
	}
	cerr << "Warning: Conn::disconnect() failed:" << parent()->name() <<
		", " << target->parent()->name() << "\n";
	return MAX;
}

//////////////////////////////////////////////////////////////////
// PlainMultiConn functions
//////////////////////////////////////////////////////////////////

unsigned long PlainMultiConn::find( const Conn* other ) const {
	vector< Conn* >::const_iterator i = 
		std::find( target_.begin(), target_.end(), other );
	if (i != target_.end() )
		return ( i - target_.begin() );
	return MAX;
}

PlainMultiConn::~PlainMultiConn()
{
	disconnectAll();
	/*
	while ( target_.size() > 0 )
		disconnect( target_.back() );
	*/
}

void PlainMultiConn::disconnectAll()
{
	vector< Conn* >::const_iterator i;
	unsigned long j;
	for (i = target_.begin(); i != target_.end(); i++ ) {
		j = (*i)->find( this );
		if ( j != MAX )
			( *i )->innerDisconnect( j );
		else
			cout << "Error: PlainMultiConn::disconnectAll\n";
	}
	target_.resize( 0 );
}


//////////////////////////////////////////////////////////////////
// MultiConn functions
//////////////////////////////////////////////////////////////////

MultiConn::~MultiConn()
{
	disconnectAll();
	/*
	vector< PlainMultiConn* >::const_iterator i;
	for ( i = connVecs_.begin(); i != connVecs_.end(); i++ ) {
		while ( ( *i )->nTargets() > 0 )
			disconnect( ( *i )->target( ( *i )->nTargets() - 1 ) );
		delete *i;
	}
	*/
}

unsigned long MultiConn::find( const Conn* other ) const
{
	unsigned long count = 0;
	vector< PlainMultiConn* >::const_iterator i;
	for ( i = connVecs_.begin(); i != connVecs_.end(); i++ ) {
		unsigned long j = ( *i )->find( other );
		if ( j != MAX )
			return j + count;
		else
			count += ( *i )->nTargets();
	}
	return MAX;
}

Conn* MultiConn::target(unsigned long index) const
{
	long temp = 0;
	vector< PlainMultiConn* >::const_iterator i;

	for ( i = connVecs_.begin(); i != connVecs_.end(); i++ ) {
		temp = index - ( *i )->nTargets();
		if ( temp < 0 )
			return ( *i )->target( index );
		else
			index = temp;
	}
	return 0;
}

unsigned long MultiConn::nTargets() const
{
	unsigned long temp = 0;
	vector< PlainMultiConn* >::const_iterator i;

	for ( i = connVecs_.begin(); i != connVecs_.end(); i++ )
		temp += ( *i )->nTargets();

	return temp;
}

void MultiConn::listTargets( vector< Conn* >& list ) const
{
	vector< PlainMultiConn* >::const_iterator i;
	for ( i = connVecs_.begin(); i != connVecs_.end(); i++ ) {
		list.insert( list.end(), ( *i )->begin(), ( *i )->end() );
	}
}

bool MultiConn::innerConnect( Conn* target, unsigned long slot )
{
	if (connVecs_.size() > slot ) {
		connVecs_[ slot ]->innerConnect( target, slot );
		return 1;
	} else if ( connVecs_.size() == slot ) { // expand
		PlainMultiConn* temp = new PlainMultiConn( parent_ );
		connVecs_.push_back( temp );
		temp->innerConnect( target, 0 );
		return 1;
	}
	return 0;
}

void MultiConn::innerDisconnect(unsigned long index)
{
	unsigned long count = 0;
	vector< PlainMultiConn* >::iterator i;
	for( i = connVecs_.begin(); i != connVecs_.end(); i++ ) {
		count += ( *i )->nTargets();
		if (count > index) {
			( *i )->innerDisconnect(
				( *i )->nTargets() + index - count );
			/*
			// We cannot do this unless the assocated recvfunc is also
			// deleted and the corresponding entry is cleared.
			if ( ( *i )->nTargets() == 0) {
				delete *i;
				connVecs_.erase( i );
			}
			*/
			return;
		}
	}
}

void MultiConn::disconnectAll()
{
	vector< PlainMultiConn* >::iterator i;
	for ( i = connVecs_.begin(); i != connVecs_.end(); i++ ) {
		//  ( *i )->disconnectAll();
		delete *i;
	}
	connVecs_.resize( 0 );
}

void MultiConn::innerDisconnectAll()
{
	vector< PlainMultiConn* >::iterator i;
	for ( i = connVecs_.begin(); i != connVecs_.end(); i++ ) {
		(*i)->innerDisconnectAll();
		delete *i;
	}
	connVecs_.resize( 0 );
}

// Returns the index of the connVec entry matching specified msgno
unsigned long MultiConn::index( unsigned long msgno )
{
	unsigned long count = 0;
	for ( unsigned long i = 0; i < connVecs_.size(); i++) {
		count += connVecs_[i]->nTargets();
		if ( count > msgno )
			return i;
	}
	return MAX;
}

//////////////////////////////////////////////////////////////////
// RelayConn functions
//////////////////////////////////////////////////////////////////

void RelayConn::innerDisconnect(unsigned long index)
{
	PlainMultiConn::innerDisconnect( index );
	if ( nTargets() == 0 )
		f_->handleEmptyConn( this );
}

//////////////////////////////////////////////////////////////////
// MultiReturnConn functions
//////////////////////////////////////////////////////////////////
MultiReturnConn::~MultiReturnConn()
{
	vector< ReturnConn* >::iterator i;
	for ( i = vec_.begin(); i != vec_.end(); i++ )
		delete *i;
}

void MultiReturnConn::listTargets( vector< Conn* >& list ) const 
{
	unsigned long i;
	unsigned long start = list.size();
	unsigned long max = start + vec_.size();
	list.resize( max );
	for (i = 0; i < max; i++ )
		list[ i + start ] = vec_[i]->rawTarget();
}

bool MultiReturnConn::innerConnect( Conn* target, unsigned long slot )
{
	ReturnConn* c = new ReturnConn( parent_ );
	c->innerConnect( target );
	vec_.push_back(c);
	return 1;
}

void MultiReturnConn::innerDisconnect(unsigned long index)
{
	if ( index < vec_.size() ) {
		delete vec_[ index ];
		vec_.erase( vec_.begin() + index );
	}
}

unsigned long MultiReturnConn::find( const Conn* other ) const
{
	unsigned long i;
	for ( i = 0; i < vec_.size(); i++ )
		if ( vec_[i]->rawTarget() == other )
			return i;
	return MAX;
}

void MultiReturnConn::disconnectAll()
{
	vector< ReturnConn* >::iterator i;
	for( i = vec_.begin(); i != vec_.end(); i++) {
		delete *i;
	}
	vec_.resize( 0 );
}

void MultiReturnConn::innerDisconnectAll()
{
	vector< ReturnConn* >::iterator i;
	for( i = vec_.begin(); i != vec_.end(); i++) {
		(*i)->innerDisconnectAll();
		delete *i;
	}
	vec_.resize( 0 );
}

unsigned long MultiReturnConn::indexOfMatchingFunc( RecvFunc rf ) const
{
	vector< ReturnConn* >::const_iterator i;
	for ( i = vec_.begin(); i != vec_.end(); i++ ) {
		if ( (*i)->recvFunc() == rf )
			return static_cast< unsigned long >( i - vec_.begin() );
	}
	return MAX;
}

//returns number of matches
unsigned long MultiReturnConn::matchRemoteFunc( RecvFunc rf ) const
{
	unsigned long count = 0;
	vector< ReturnConn* >::const_iterator i;
	for ( i = vec_.begin(); i != vec_.end(); i++ )
		if ( (*i)->recvFunc() == rf )
			count++;
	return count;
}

// This adds the func to the last ReturnConn
void MultiReturnConn::addRecvFunc( RecvFunc rf )
{
	if ( vec_.size() > 0 )
		vec_.back()->addRecvFunc( rf );
}
