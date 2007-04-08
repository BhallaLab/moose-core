/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/*
#include "header.h"
#include "MsgSrc.h"
#include "MsgDest.h"
#include "DeletionMarkerFinfo.h"
#include "SimpleElement.h"
#include "ThisFinfo.h"
*/

#include "moose.h"
#include "DeletionMarkerFinfo.h"
#include "ThisFinfo.h"

#ifdef DO_UNIT_TESTS
int SimpleElement::numInstances = 0;
#endif

SimpleElement::SimpleElement( const std::string& name )
	: Element(), name_( name ), data_( 0 )
{
#ifdef DO_UNIT_TESTS
		numInstances++;
#endif
		;
}

SimpleElement::SimpleElement(
				const std::string& name, 
				unsigned int srcSize,
				unsigned int destSize,
				void* data
	)
	: Element(), name_( name ), src_( srcSize ), dest_( destSize ), data_( data )
{
#ifdef DO_UNIT_TESTS
		numInstances++;
#endif	
		;
}

SimpleElement::~SimpleElement()
{
#ifndef DO_UNIT_TESTS
	// The unit tests create SimpleElement without any finfos.
	assert( finfo_.size() > 0 );
#endif	
#ifdef DO_UNIT_TESTS
	numInstances--;
#endif

	/**
	 * \todo Lots of cleanup stuff here to implement.
	// Find out what data is, and call its delete command.
	ThisFinfo* tf = dynamic_cast< ThisFinfo* >( finfo_[0] );
	tf->destroy( data() );
	*/	
	if ( data_ ) {
		if ( finfo_.size() > 0 && finfo_[0] != 0 ) {
			ThisFinfo* tf = dynamic_cast< ThisFinfo* >( finfo_[0] );
			if ( tf && tf->noDeleteFlag() == 0 )
				finfo_[0]->ftype()->destroy( data_, 0 );
		}
	}


	// Check if Finfo is one of the transient set, if so, clean it up.
	vector< Finfo* >::iterator i;
	for ( i = finfo_.begin(); i != finfo_.end(); i++ )
		if ( (*i)->isTransient() )
			delete *i;
}

const std::string& SimpleElement::className( ) const
{
	const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( finfo_[0] );
	assert( tf != 0 );
	return tf->cinfo()->name();
}

/**
 * Returns the index of the first matching MsgSrc of the shared set,
 * or if it doesn't exist, it makes a new set of MsgSrcs and returns
 * the starting index of this set.
 * Assumes that the Finfo that called this function has done the
 * necessary type matching to the src.
 * Ensures that all Conns belonging to the linked list of MsgSrcs are
 * contiguous.
 */
unsigned int SimpleElement::insertConnOnSrc(
		unsigned int src, FuncList& rf,
		unsigned int dest, unsigned int nDest )
{
	assert ( rf.size() > 0 );
	assert ( src_.size() > 0 );
	assert ( src_.size() >= src + rf.size() );

	vector< RecvFunc > oldFuncs;
	MsgSrc& s( src_[ src ] );
	unsigned int begin = s.begin();
	unsigned int end = s.end();
	unsigned int next = s.next();

	for (unsigned int i = 0; i < rf.size(); i++ ) {
		// MsgSrc& s( src_[ src + i ] );
		// assert( begin == s.begin() );
		// assert( end == s.end() );
		// assert( next == s.next() );
		oldFuncs.push_back( src_[ src + i ].recvFunc() );
	}

	if ( begin == end ) { // Need to initialize current MsgSrc
		unsigned int i;
		for ( i = 0; i < rf.size(); i++ ) {
			assert( src_[ src + i ].recvFunc() == dummyFunc );
			src_[ src + i ].setFunc( rf[i] );
		}
		return insertConn( src, rf.size(), dest, nDest );
	}

	if ( oldFuncs == rf ) // This src matches, insertion in this src
		return insertConn( src, rf.size(), dest, nDest );

	if ( next != 0 ) // Look in next range for insertion
		return insertConnOnSrc( src + next, rf, dest, nDest );

	// Nothing matches. So we create a new range at the end of the src_.
	unsigned int offset = src_.size() - src;
	for (unsigned int i = 0; i < rf.size(); i++ ) {
		src_[ src + i ].setNext( offset );
		// Note that the new src uses the 'end' from the previous one.
		// This ensures that all Conns within a MsgSrc are contiguous.
		src_.push_back( MsgSrc( end, end, rf[i] ) );
	}
	return insertConn( src_.size() - rf.size(), rf.size(), dest, nDest);
}

/**
 * Inserts a Conn for the desired dest. Returns the index of the
 * new Conn. Assumes that there are no shared Srcs to be dealt with.
 * The nDest is in case this is a shared dest, and we may have to
 * increment multiple dest entries.
 * \todo: Perhaps we don't have to increment multiple entries. If
 * it is a shared dest then a single one will do for all. In that
 * case we should eliminate the nDest argument.
 */
unsigned int SimpleElement::insertConnOnDest(
		unsigned int dest, unsigned int nDest )
{
	assert( dest_.size() >= dest + nDest );

	return insertConn( 0, 0, dest, nDest );
}

vector< Conn >::const_iterator
	SimpleElement::lookupConn( unsigned int i ) const
{
	assert ( i < conn_.size() );
	return conn_.begin() + i;
}

vector< Conn >::iterator
	SimpleElement::lookupVariableConn( unsigned int i )
{
	assert ( i < conn_.size() );
	return conn_.begin() + i;
}

RecvFunc SimpleElement::lookupRecvFunc(
		unsigned int src, unsigned int conn ) const
{
	assert ( src < src_.size() );
	assert ( conn < conn_.size() );
	assert ( src_[src].begin() <= conn );
	vector< MsgSrc >::const_iterator i = src_.begin() + src;
	while ( i->end() < conn ) {
		assert ( i->next() != 0 );
		i += i->next();
	}
	return i->recvFunc();
}

unsigned int SimpleElement::connIndex( const Conn* c ) const
{
	return static_cast< unsigned int >( c - &conn_.front() );
}

/**
 * This finds the relative index of a conn arriving at this element.
 */
unsigned int SimpleElement::connDestRelativeIndex(
				const Conn& c, unsigned int slot ) const
{
	assert ( slot < dest_.size() );
	assert ( conn_.size() >= dest_[ slot ].begin() );
	assert ( c.targetIndex() >= dest_[ slot ].begin() );
	return c.targetIndex() - dest_[ slot ].begin();
}

//////////////////////////////////////////////////////////////////
// Src functions
//////////////////////////////////////////////////////////////////

vector< Conn >::const_iterator
	SimpleElement::connSrcBegin( unsigned int src ) const
{
	assert ( src < src_.size() );
	assert ( conn_.size() >= src_[ src ].begin() );
	return conn_.begin() + src_[ src ].begin();
}

vector< Conn >::const_iterator
	SimpleElement::connSrcEnd( unsigned int src ) const
{
	assert (  src  < src_.size() );
	assert ( conn_.size() >= src_[ src ].end() );
	return conn_.begin() + src_[ src ].end();
}


vector< Conn >::const_iterator
	SimpleElement::connSrcVeryEnd( unsigned int src ) const
{
	assert (  src  < src_.size() );
	assert ( conn_.size() >= src_[ src ].end() );
	unsigned int n = nextSrc( src );
	if ( n != 0 )
		return connSrcVeryEnd( n );

	return conn_.begin() + src_[ src ].end();
}

unsigned int SimpleElement::nextSrc( unsigned int src ) const
{
	assert ( src < src_.size() );
	return ( src_[ src ].next() == 0 ) ? 0 : src + src_[ src ].next();
	// return src_[ src ].next();
}

RecvFunc SimpleElement::srcRecvFunc( unsigned int src ) const
{
	assert ( src < src_.size() );
	return src_[ src ].recvFunc();
}

//////////////////////////////////////////////////////////////////
// Dest functions
//////////////////////////////////////////////////////////////////
vector< Conn >::const_iterator
	SimpleElement::connDestBegin( unsigned int dest ) const
{
	assert ( dest < dest_.size() );
	assert ( conn_.size() >= dest_[ dest ].begin() );
	return conn_.begin() + dest_[ dest ].begin();
}

vector< Conn >::const_iterator
	SimpleElement::connDestEnd( unsigned int dest ) const
{
	assert (  dest  < dest_.size() );
	assert ( conn_.size() >= dest_[ dest ].end() );
	return conn_.begin() + dest_[ dest ].end();
}

//////////////////////////////////////////////////////////////////
// Conn functions
//////////////////////////////////////////////////////////////////

/**
 * Puts a new Conn at the end of the set of srcs and dests specified,
 * and update this set and all later ones.
 * Return the location of the inserted Conn.
 */
unsigned int SimpleElement::insertConn(
				unsigned int src, unsigned int nSrc,
				unsigned int dest, unsigned int nDest )
{
	assert( src + nSrc <= src_.size() );
	assert( dest + nDest <= dest_.size() );
	assert( nSrc + nDest > 0 );
	unsigned int location;
	if ( nSrc > 0 )
			location = src_[ src ].end();
	else if ( nDest > 0 )
			location = dest_[ dest ].end();

	conn_.insert( conn_.begin() + location, Conn() );

	// Update the src_ and dest_ ranges. This is the easy part.
	vector< MsgSrc >::iterator i;
	if ( nSrc > 0 ) {
		vector< MsgSrc >::iterator j = src_.begin() + src + nSrc;
		for ( i = src_.begin() + src; i != j; i++ )
			i->insertWithin( );
		for ( ; i != src_.end(); i++ )
			i->insertBefore( );
	}

	vector< MsgDest >::iterator k;
	for ( k = dest_.begin() + dest; 
					k != dest_.begin() + dest + nDest; k++ )
		k->insertWithin();
	for ( ; k != dest_.end(); k++ )
		k->insertBefore( );

	// Update the Conns. This is the hard part.
	for ( unsigned int j = location + 1; j < conn_.size(); j++ )
		conn_[j].updateIndex( j );

	return location;
}

/**
 * Take two naive Conns, and assign their values so that they point
 * to each other
 */
void SimpleElement::connect( unsigned int myConn, 
				Element* targetElement, unsigned int targetConn)
{
	assert( conn_.size() > myConn );
	assert( targetElement != 0 );
	assert( targetElement->connSize() > targetConn );

	conn_[myConn].set( targetElement, targetConn );
	targetElement->lookupVariableConn( targetConn )->
			set( this, myConn );
}

/**
 * Delete a Conn. Deleting a single conn_ also deletes its partner. 
 * This involves cleaning up the range_ and the conn_
 * vectors locally, as well as at the target.
 */
void SimpleElement::disconnect( unsigned int connIndex )
{
	assert( connIndex < conn_.size() );
	conn_[ connIndex ].targetElement()->
			deleteHalfConn( conn_[ connIndex ].targetIndex() );
	deleteHalfConn( connIndex );
}

/**
 * Do the clean up only from the viewpoint of the conn on the local
 * object. This involves cleaning up the src_ and dest_ and updating
 * the * return index of each of the targets of the conn_ vector.
 * The calling function has to ensure that the target conn is likewise
 * cleaned up.
 * Note that this function may leave orphan src_ ranges that have
 * a function set but are empty. Simplest to ignore, as it is unlikely
 * to be a major source of garbage.
 */
void SimpleElement::deleteHalfConn( unsigned int connIndex )
{
	assert( connIndex < conn_.size() );

	vector< MsgSrc >::iterator i;
	for ( i = src_.begin(); i != src_.end(); i++ )
		i->dropConn( connIndex );

	vector< MsgDest >::iterator j;
	for ( j = dest_.begin(); j != dest_.end(); j++ )
		j->dropConn( connIndex );

	conn_.erase( conn_.begin() + connIndex );
	for ( unsigned int j = connIndex; j < conn_.size(); j++ )
		conn_[j].updateIndex( j );
}

const Finfo* SimpleElement::findFinfo( const string& name )
{
	vector< Finfo* >::reverse_iterator i;
	const Finfo* ret;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	// Reverse iterate because the zeroth finfo is the base,
	// and we want more recent finfos to override old ones.
	for ( i = finfo_.rbegin(); i != finfo_.rend(); i++ )
	{
			ret = (*i)->match( this, name );
			if ( ret )
					return ret;
	}
	return 0;
}

const Finfo* SimpleElement::findFinfo( unsigned int connIndex ) const
{
	vector< Finfo* >::const_reverse_iterator i;
	const Finfo* ret;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	// Reverse iterate because the zeroth finfo is the base,
	// and we want more recent finfos to override old ones.
	for ( i = finfo_.rbegin(); i != finfo_.rend(); i++ )
	{
			ret = (*i)->match( this, connIndex );
			if ( ret )
					return ret;
	}
	return 0;
}

unsigned int SimpleElement::listFinfos( 
				vector< const Finfo* >& flist ) const
{
	vector< Finfo* >::const_iterator i;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	for ( i = finfo_.begin(); i != finfo_.end(); i++ )
	{
		(*i)->listFinfos( flist );
	}

	return flist.size();
}

unsigned int SimpleElement::listLocalFinfos( vector< Finfo* >& flist )
{
	flist.resize( 0 );
	if ( finfo_.size() <= 1 )
		return 0;
	flist.insert( flist.end(), finfo_.begin() + 1, finfo_.end() );
	return flist.size();
}

/**
 * Here we need to put in the new Finfo, and also check if it
 * requires allocation of any MsgSrc or MsgDest slots.
 */
void SimpleElement::addFinfo( Finfo* f )
{
	unsigned int nSrc = src_.size();
	unsigned int nDest = dest_.size();
	f->countMessages( nSrc, nDest );

	if ( nSrc > src_.size() ) {
		unsigned int end = lastSrcConnIndex();
		MsgSrc temp( end, end, dummyFunc );
		src_.push_back( temp );
	}
	if ( nDest > dest_.size() ) {
		unsigned int end = lastDestConnIndex();
		MsgDest temp( end, end );
		dest_.push_back( temp );
	}
	finfo_.push_back( f );
}

bool SimpleElement::isConnOnSrc(
			unsigned int srcIndex, unsigned int connIndex ) const
{
	assert( srcIndex < src_.size() );
	assert( connIndex < conn_.size() );
	vector< MsgSrc >::const_iterator i = src_.begin() + srcIndex;

	while ( i != src_.end() ) {
		if ( i->begin() <= connIndex && i->end() > connIndex )
				return 1;
		if ( i->next() > 0 )
				i += i->next();
		else
			return 0;
	}
	assert( 0 ); // should never get here
	return 0;
}

bool SimpleElement::isConnOnDest(
			unsigned int destIndex, unsigned int connIndex ) const
{
	assert( destIndex < dest_.size() );
	assert( connIndex < conn_.size() );
	return (
			dest_[ destIndex ].begin() <= connIndex &&
			dest_[ destIndex ].end() > connIndex
	);
}

unsigned int SimpleElement::lastSrcConnIndex() const
{
	vector< MsgSrc >::const_iterator i;
	unsigned int ret = 0;
	for (i = src_.begin(); i != src_.end(); i++ )
		if ( ret < i->end() )
				ret = i->end();

	return ret;
}

unsigned int SimpleElement::lastDestConnIndex() const
{
	vector< MsgDest >::const_iterator i;
	unsigned int ret = 0;
	for (i = dest_.begin(); i != dest_.end(); i++ )
		if ( ret < i->end() )
				ret = i->end();

	return ret;
}

bool SimpleElement::isMarkedForDeletion() const
{
	if ( finfo_.size() > 0 )
		return finfo_.back() == DeletionMarkerFinfo::global();
	// This fallback case should only occur during unit testing.
	return 0;
}

void SimpleElement::prepareForDeletion( bool stage )
{
	if ( stage == 0 ) {
		finfo_.push_back( DeletionMarkerFinfo::global() );
	} else {
		vector< Conn >::iterator i;
		for ( i = conn_.begin(); i != conn_.end(); i++ ) {
			if ( !i->targetElement()->isMarkedForDeletion() )
				i->targetElement()->deleteHalfConn( i->targetIndex() );
		}
	}
}
