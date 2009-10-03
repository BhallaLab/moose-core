/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"


Conn::~Conn()
{;}

void Conn::clearConn()
{
	for( vector< Msg* >::const_iterator i = m_.begin(); 
		i != m_.end(); ++i ) {
		delete *i;
	}
}

void Conn::asend( 
	const Element* e, Qinfo& q, const char* arg ) const
{
	for( vector< Msg* >::const_iterator i = m_.begin(); i != m_.end(); ++i )
		(*i)->addToQ( e, q, arg );
}

// Checks for the correct Msg, and expands the arg to append the
// target index.
void Conn::tsend( 
	const Element* e, Id target, Qinfo& q, const char* arg ) const
{
	assert( q.useSendTo() );
	for( vector< Msg* >::const_iterator i = m_.begin(); i != m_.end(); ++i ) {
		if ( (*i)->e2() == target() || (*i)->e1() == target() ) {
			char* temp = new char[ q.size() + sizeof( unsigned int ) ];
			memcpy( temp, arg, q.size() );
			q.expandSize();
			*reinterpret_cast< unsigned int* >( temp + q.size() ) = 
				target.index();
				(*i)->addToQ( e, q, arg );
			delete[] temp;
			break;
		}
	}
}

/**
 * ClearQ calls clearQ on all Msgs, on e1.
 */
void Conn::clearQ()
{
	for( vector< Msg* >::const_iterator i = m_.begin(); i != m_.end(); ++i )
		(*i)->clearQ();
}

/**
* Add a msg to the list
*/
void Conn::add( Msg* m )
{
	m_.push_back( m );
}

/**
* Drop a msg from the list
*/
void Conn::drop( Msg* m )
{
	// Here we have the spectacularly ugly C++ erase-remove idiot.
	m_.erase( remove( m_.begin(), m_.end(), m ), m_.end() ); 
}
