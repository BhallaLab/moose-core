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
	vector< MsgId > temp = m_;
	// Note that we can't iterate directly over m_, because the deletion
	// operator alters m_ and will invalidate the iterators.
	m_.resize( 0 ); // This avoids the system trying to go through all
	// of the messages on m_. But we still have the issue of
	// it going through all other conns looking for the msg to be deleted.
	for( vector< MsgId >::const_iterator i = temp.begin(); 
		i != temp.end(); ++i )
	{
		Msg::deleteMsg( *i );
	}
}

void Conn::asend( 
	const Element* e, Qinfo& q, const char* arg ) const
{
	for( vector< MsgId >::const_iterator i = m_.begin(); i != m_.end(); ++i)
		Msg::getMsg( *i )->addToQ( e, q, arg );
}

// Checks for the correct Msg, and expands the arg to append the
// target index.
void Conn::tsend( 
	const Element* e, DataId target, Qinfo& q, const char* arg ) const
{
	assert( q.useSendTo() );
	for( vector< MsgId >::const_iterator i = m_.begin(); i != m_.end(); ++i)
	{
		char* temp = new char[ q.size() + sizeof( DataId ) ];
		memcpy( temp, arg, q.size() );
		*reinterpret_cast< DataId* >( temp + q.size() ) = target;
		q.expandSize();
		Msg::getMsg( *i )->addToQ( e, q, temp );
		delete[] temp;
		break;
	}
}

/*
void Conn::tsend( 
	const Element* e, unsigned int targetIndex, Qinfo& q, const char* arg ) const
{
	assert( q.useSendTo() );
	assert( m_.size() == 1 );
	char* temp = new char[ q.size() + sizeof( unsigned int ) ];
	memcpy( temp, arg, q.size() );
	*reinterpret_cast< unsigned int* >( temp + q.size() ) = targetIndex;
	q.expandSize();
	Msg::getMsg( m_[0] )->addToQ( e, q, temp );
	delete[] temp;
}
*/

/**
 * process calls process on all Msgs, on e2.
 */
void Conn::process( const ProcInfo* p ) const
{
	for( vector< MsgId >::const_iterator i = m_.begin(); i != m_.end(); ++i)
		Msg::getMsg( *i )->process( p );
}

/**
 * ClearQ calls clearQ on all Msgs, on e2.
void Conn::clearQ() const
{
	for( vector< Msg* >::const_iterator i = m_.begin(); i != m_.end(); ++i )
		(*i)->clearQ();
}
 */

/**
* Add a msg to the list
*/
void Conn::add( MsgId m )
{
	m_.push_back( m );
}

/**
* Drop a msg from the list
*/
void Conn::drop( MsgId mid )
{
	// Here we have the spectacularly ugly C++ erase-remove idiot.
	m_.erase( remove( m_.begin(), m_.end(), mid ), m_.end() ); 
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
		if ( Msg::getMsg( m_[ index ] )->e1() == otherElement )
			return Msg::getMsg( m_[ index ] )->e2();
		else if ( Msg::getMsg( m_[ index ] )->e2() == otherElement )
			return Msg::getMsg( m_[ index ] )->e1();
	}
	return 0;
}
