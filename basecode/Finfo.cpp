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

Conn* Finfo::dummyConn_ = new SynConn< int >( Element::root() );

Field Finfo::match(const string& s)
{
	if (name_ == s)
		return Field( this );
	return Field();
}

void Finfo::src( vector< Field >& list, Element* e )
{
	vector< Conn* > conns;
	inConn( e )->listTargets( conns );
	vector< Conn* >::iterator i;

	for ( i = conns.begin() ; i != conns.end(); i++) {
		list.push_back(
			( *i )->parent()->lookupSrcField( *i, recvFunc() )
		);
	}
}

void Finfo::dest( vector< Field >& list, Element* e )
{
	vector< Conn* > conns;
	outConn( e )->listTargets( conns );
	vector< Conn* >::iterator i;

	for ( i = conns.begin() ; i != conns.end(); i++) {
		list.push_back(
			( *i )->parent()->lookupDestField( *i, targetFunc( e, 0 ) )
		);
	}
}

// Removes a connection. Basically looks up src and dest conns to do so.
// Usually called from the dest with the src as argument,
// because synapses (destinations) have to override this
// function.
bool Finfo::drop( Element* e, Field& srcfield ) {
	// Conn* iconn = destfield->inConn( e );
	Conn* iconn = inConn( e );
	Conn* oconn = srcfield->outConn( srcfield.getElement() );
	return ( oconn->disconnect( iconn ) != Conn::MAX );
}

// Must be called from a dest Finfo. Disconnects all inputs.
bool Finfo::dropAll( Element* e ) {
	Conn* iconn = inConn( e );
	if ( iconn ) {
		iconn->disconnectAll();
		return 1;
	}
	return 0;
}

bool Finfo::strGet( Element* e, string& val )
{
	return this->ftype()->strGet( e, this, val );
}

bool Finfo::strSet( Element* e, const string& val )
{
	return this->ftype()->strSet( e, this, val );
}

bool Finfo::isSameType( const Finfo* other ) const
{
	return ( this->ftype()->isSameType( other->ftype() ) );
}

bool Finfo::isSameType( const Ftype* other ) const
{
	return ( this->ftype()->isSameType( other ) );
}

//////////////////////////////////////////////////////////////////
// DummyFinfo settings.
//////////////////////////////////////////////////////////////////

const Ftype* FinfoDummy::ftype() const
{
	static const Ftype0 myFtype;
	return &myFtype;
}
