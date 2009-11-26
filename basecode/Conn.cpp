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
	vector< Msg* > temp = m_;
	// Note that we can't iterate directly over m_, because the deletion
	// operator alters m_ and will invalidate the iterators.
	m_.resize( 0 ); // This avoids the system trying to go through all
	// of the messages on m_. But we still have the disgusting issue of
	// it going through all other conns looking for the msg to be deleted.
	for( vector< Msg* >::const_iterator i = temp.begin(); 
		i != temp.end(); ++i )
	{
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
			*reinterpret_cast< unsigned int* >( temp + q.size() ) = 
				target.index();
			q.expandSize();
			(*i)->addToQ( e, q, temp );
			delete[] temp;
			break;
		}
	}
}

void Conn::tsend( 
	const Element* e, unsigned int targetIndex, Qinfo& q, const char* arg ) const
{
	assert( q.useSendTo() );
	assert( m_.size() == 1 );
	char* temp = new char[ q.size() + sizeof( unsigned int ) ];
	memcpy( temp, arg, q.size() );
	*reinterpret_cast< unsigned int* >( temp + q.size() ) = targetIndex;
	q.expandSize();
	m_[0]->addToQ( e, q, temp );
	delete[] temp;
}

/**
 * process calls process on all Msgs, on e2.
 */
void Conn::process( const ProcInfo* p ) const
{
	for( vector< Msg* >::const_iterator i = m_.begin(); i != m_.end(); ++i )
		(*i)->process( p );
}

/**
 * ClearQ calls clearQ on all Msgs, on e2.
 */
void Conn::clearQ() const
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
void Conn::drop( const Msg* m )
{
	// Here we have the spectacularly ugly C++ erase-remove idiot.
	m_.erase( remove( m_.begin(), m_.end(), m ), m_.end() ); 
}

/**
 * Reassign target. Used typically for once-off calls like 'set'.
 * Creates Msg, if doesn't exist.
 * Releases previous target, if any.
 * Clear later Msgs, if any.
void Conn::setMsgDest( Eref& src, Eref& dest )
{
	clearConn();
	assert ( m_.size() == 0 );
	m_.push_back( new SingleMsg( src, dest ) );
}
 */

unsigned int Conn::numMsg() const 
{
	return m_.size();
}

Element* Conn::getTargetElement( 
	const Element* otherElement, unsigned int index ) const
{
	if ( index < m_.size() ) {
		if ( m_[ index ]->e1() == otherElement )
			return m_[ index ]->e2();
		else if ( m_[ index ]->e2() == otherElement )
			return m_[ index ]->e1();
	}
	return 0;
}
