/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SEND_H
#define _SEND_H

//////////////////////////////////////////////////////////////////////////
//                      Zero argument section
//////////////////////////////////////////////////////////////////////////
/**
 * This function sends zero-argument messages.
 */
extern void send0( Eref e, Slot src );

/**
 * This function sends zero-argument messages to a specific target.
 * The index is the index in the local conn_ vector of the source Msg.
 */
extern void sendTo0( Eref e, Slot src, unsigned int index);

/**
 * sendBack returns zero-argument messages to sender. 
 * All return info is already there in the Conn.
 */
void sendBack0( const Conn* c, Slot src );

//////////////////////////////////////////////////////////////////////////
//                      One argument section
//////////////////////////////////////////////////////////////////////////

/**
 * This templated function sends single-argument messages.
 */
template < class T > void send1( Eref e, Slot src, T val )
{
	const Msg* m = e.e->msg( src.msg() );
	if ( m->funcId() == 0 ) return;
	do {
		void( *rf )( const Conn*, T ) = 
			reinterpret_cast< void ( * )( const Conn*, T ) >(
				m->func( src.func() )
			);
		vector< ConnTainer* >::const_iterator i;
		for ( i = m->begin(); i != m->end(); i++ ) {
			Conn* j = ( *i )->conn( e, src.func() );
			for ( ; j->good(); j->increment() ){
				rf( j, val );
			}
			delete j;
		}
	// Yes, it is an assignment, not a comparison
	} while ( ( m = m->next( e.e ) ) ); 
}

/**
 * This templated function sends a single-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T > void sendTo1( Eref e,
		Slot src, unsigned int tgt, T val )
{
	const Msg* m = e.e->msg( src.msg() );
	void( *rf )( const Conn*, T ) = 
			reinterpret_cast< void ( * )( const Conn*, T ) >(
			m->func( src.func() )
		);
	const Conn* j = m->findConn( e, tgt, src.func() );
	rf( j,  val );
	delete j;
}

template< class T > void sendBack1( const Conn* c, Slot src, T val )
{
	const Msg* m = c->target().e->msg( src.msg() );
	void( *rf )( const Conn*, T ) = 
			reinterpret_cast< void ( * )( const Conn*, T ) >(
			m->func( src.func() )
		);
	const Conn* flip = c->flip( src.func() );
	rf( flip, val );
	delete flip;
}

//////////////////////////////////////////////////////////////////////////
//                      Two argument section
//////////////////////////////////////////////////////////////////////////
/**
 * This templated function sends two-argument messages.
 */
template < class T1, class T2 > void send2( 
		Eref e, Slot src, T1 v1, T2 v2 )
{
	const Msg* m = e.e->msg( src.msg() );
	if ( m->funcId() == 0 ) return;
	do {
		void( *rf )( const Conn*, T1, T2 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2 ) >(
				m->func( src.func() )
			);
		vector< ConnTainer* >::const_iterator i;
		for ( i = m->begin(); i != m->end(); i++ ) {
			Conn* j = ( *i )->conn( e, src.func() );
			for ( ; j->good(); j->increment() )
				rf( j, v1, v2 );
			delete j;
		}
	// Yes, it is an assignment, not a comparison
	} while ( ( m = m->next( e.e ) ) ); 
}

/**
 * This templated function sends a single-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T1, class T2 > void sendTo2( Eref e,
		Slot src, unsigned int tgt, T1 v1, T2 v2 )
{
	const Msg* m = e.e->msg( src.msg() );
	void( *rf )( const Conn*, T1, T2 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2 ) >(
			m->func( src.func() )
		);
	const Conn* j = m->findConn( e, tgt, src.func() );
	rf( j,  v1, v2 );
	delete j;
}

template< class T1, class T2 > void sendBack2( const Conn* c, Slot src, 
	T1 v1, T2 v2 )
{
	const Msg* m = c->target().e->msg( src.msg() );
	void( *rf )( const Conn*, T1, T2 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2 ) >(
			m->func( src.func() )
		);
	const Conn* flip = c->flip( src.func() );
	rf( flip, v1, v2 );
	delete flip;
}

//////////////////////////////////////////////////////////////////////////
//                      Three argument section
//////////////////////////////////////////////////////////////////////////

/**
 * This templated function sends three-argument messages.
 */
template < class T1, class T2, class T3 > void send3(
	Eref e, Slot src, T1 v1, T2 v2, T3 v3 )
{
	const Msg* m = e.e->msg( src.msg() );
	if ( m->funcId() == 0 ) return;
	do {
		void( *rf )( const Conn*, T1, T2, T3 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3 ) >(
				m->func( src.func() )
			);
		vector< ConnTainer* >::const_iterator i;
		for ( i = m->begin(); i != m->end(); i++ ) {
			Conn* j = ( *i )->conn( e, src.func() );
			for ( ; j->good(); j->increment() )
				rf( j, v1, v2, v3 );
			delete j;
		}
	// Yes, it is an assignment, not a comparison
	} while ( ( m = m->next( e.e ) ) ); 
}

/**
 * This templated function sends a single-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T1, class T2, class T3 > void sendTo3(
		Eref e, Slot src, unsigned int tgt, T1 v1, T2 v2, T3 v3 )
{
	const Msg* m = e.e->msg( src.msg() );
	void( *rf )( const Conn*, T1, T2, T3 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3 ) >(
			m->func( src.func() )
		);
	const Conn* j = m->findConn( e, tgt, src.func() );
	rf( j,  v1, v2, v3 );
	delete j;
}

template< class T1, class T2, class T3 > 
	void sendBack3( const Conn* c, Slot src, T1 v1, T2 v2, T3 v3 )
{
	const Msg* m = c->target().e->msg( src.msg() );
	void( *rf )( const Conn*, T1, T2, T3 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3 ) >(
			m->func( src.func() )
		);
	const Conn* flip = c->flip( src.func() );
	rf( flip, v1, v2, v3 );
	delete flip;
}

//////////////////////////////////////////////////////////////////////////
//                      Four argument section
//////////////////////////////////////////////////////////////////////////
/**
 * This templated function sends four-argument messages.
 */
template < class T1, class T2, class T3, class T4 > void send4(
	Eref e, Slot src, T1 v1, T2 v2, T3 v3, T4 v4 )
{
	const Msg* m = e.e->msg( src.msg() );
	if ( m->funcId() == 0 ) return;
	do {
		void( *rf )( const Conn*, T1, T2, T3, T4 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3, T4 ) >(
				m->func( src.func() )
			);
		vector< ConnTainer* >::const_iterator i;
		for ( i = m->begin(); i != m->end(); i++ ) {
			Conn* j = ( *i )->conn( e, src.func() );
			for ( ; j->good(); j->increment() )
				rf( j, v1, v2, v3, v4 );
			delete j;
		}
	// Yes, it is an assignment, not a comparison
	} while ( ( m = m->next( e.e ) ) );
}

/**
 * This templated function sends a single-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T1, class T2, class T3, class T4 > void sendTo4(
	Eref e, Slot src, unsigned int tgt, T1 v1, T2 v2, T3 v3, T4 v4 )
{
	const Msg* m = e.e->msg( src.msg() );
	void( *rf )( const Conn*, T1, T2, T3, T4 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3, T4 ) >(
			m->func( src.func() )
		);
	const Conn* j = m->findConn( e, tgt, src.func() );
	rf( j,  v1, v2, v3, v4 );
	delete j;
}

template< class T1, class T2, class T3, class T4 > 
	void sendBack4( const Conn* c, Slot src, T1 v1, T2 v2, T3 v3, T4 v4 )
{
	const Msg* m = c->target().e->msg( src.msg() );
	void( *rf )( const Conn*, T1, T2, T3, T4 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3, T4 ) >(
			m->func( src.func() )
		);
	const Conn* flip = c->flip( src.func() );
	rf( flip, v1, v2, v3, v4 );
	delete flip;
}

//////////////////////////////////////////////////////////////////////////
//                      Five argument section
//////////////////////////////////////////////////////////////////////////
/**
 * This templated function sends four-argument messages.
 */
template < class T1, class T2, class T3, class T4, class T5 > void send5(
	Eref e, Slot src, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5 )
{
	const Msg* m = e.e->msg( src.msg() );
	if ( m->funcId() == 0 ) return;
	do {
		void( *rf )( const Conn*, T1, T2, T3, T4, T5 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3, T4, T5 ) >(
				m->func( src.func() )
			);
		vector< ConnTainer* >::const_iterator i;
		for ( i = m->begin(); i != m->end(); i++ ) {
			Conn* j = ( *i )->conn( e, src.func() );
			for ( ; j->good(); j->increment() )
				rf( j, v1, v2, v3, v4, v5 );
			delete j;
		}
	// Yes, it is an assignment, not a comparison
	} while ( ( m = m->next( e.e ) ) );
}

/**
 * This templated function sends a single-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T1, class T2, class T3, class T4, class T5 > void sendTo5(
	Eref e, Slot src, unsigned int tgt, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5 )
{
	const Msg* m = e.e->msg( src.msg() );
	void( *rf )( const Conn*, T1, T2, T3, T4, T5 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3, T4, T5 ) >(
			m->func( src.func() )
		);
	const Conn* j = m->findConn( e, tgt, src.func() );
	rf( j,  v1, v2, v3, v4, v5 );
	delete j;
}

template< class T1, class T2, class T3, class T4, class T5 > 
	void sendBack5( const Conn* c, Slot src, 
	T1 v1, T2 v2, T3 v3, T4 v4, T5 v5 )
{
	const Msg* m = c->target().e->msg( src.msg() );
	void( *rf )( const Conn*, T1, T2, T3, T4, T5 ) = 
			reinterpret_cast< void ( * )( const Conn*, T1, T2, T3, T4, T5 ) >(
			m->func( src.func() )
		);
	const Conn* flip = c->flip( src.func() );
	rf( flip, v1, v2, v3, v4, v5 );
	delete flip;
}

#endif // _SEND_H
