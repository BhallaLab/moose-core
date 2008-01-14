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
#include <vector>
#include "Element.h"
#include "SimpleElement.h"
#include "Conn.h"

/**
 * This function sends zero-argument messages.
 */
extern void send0( const Element* eIn, unsigned int src );

/**
 * This function sends zero-argument messages to a specific target.
 * The target index is given by the absolute index in the local conn_
 * vector.
 */
extern void sendTo0( const Element* eIn, unsigned int src, 
				unsigned int index);

/**
 * This templated function sends single-argument messages.
 */
template < class T > void send1(
				const Element* eIn, unsigned int src, T val )
{
	const SimpleElement* e = static_cast< const SimpleElement* >( eIn );

	do {
		void( *rf )( const Conn&, T ) = 
				reinterpret_cast< void ( * )( const Conn&, T ) >(
				e->srcRecvFunc( src ) ); 
		/// \todo This should be replaced by the STL binder and foreach
		vector< Conn >::const_iterator j;
		for ( j = e->connSrcBegin( src );
						j != e->connSrcEnd( src ); j++ )
				rf( *j, val );
		src = e->nextSrc( src );
	} while ( src != 0 );
}

/**
 * This templated function sends a single-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T > void sendTo1(
		const Element* eIn, unsigned int src, unsigned int conn, T val )
{
	const SimpleElement* e = static_cast< const SimpleElement* >( eIn );
	void( *rf )( const Conn&, T ) = 
			reinterpret_cast< void ( * )( const Conn&, T ) >(
			e->lookupRecvFunc( src, conn ) );
		rf( *( e->lookupConn( conn ) ),  val );
}

/**
 * This templated function sends two-argument messages.
 */
template < class T1, class T2 > void send2(
			const Element* eIn, unsigned int src, T1 val1, T2 val2 )
{
	const SimpleElement* e = static_cast< const SimpleElement* >( eIn );

	do {
		void( *rf )( const Conn&, T1, T2 ) = 
				reinterpret_cast< void ( * )( const Conn&, T1, T2 ) >(
				e->srcRecvFunc( src ) ); 
		/// \todo Are there STL binders for 2 values?
		vector< Conn >::const_iterator j;
		for ( j = e->connSrcBegin( src );
						j != e->connSrcEnd( src ); j++ )
				rf( *j, val1, val2 );
		src = e->nextSrc( src );
	} while ( src != 0 );
}

/**
 * This templated function sends a two-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T1, class T2 > void sendTo2(
		const Element* eIn, unsigned int src, unsigned int conn,
		T1 val1, T2 val2 )
{
	const SimpleElement* e = static_cast< const SimpleElement* >( eIn );
	void( *rf )( const Conn&, T1, T2 ) = 
			reinterpret_cast< void ( * )( const Conn&, T1, T2 ) >(
			e->lookupRecvFunc( src, conn ) );
		rf( *( e->lookupConn( conn ) ),  val1, val2 );
}

/**
 * This templated function sends three-argument messages.
 */
template < class T1, class T2, class T3 > void send3(
			const Element* eIn, unsigned int src,
			T1 val1, T2 val2, T3 val3 )
{
	const SimpleElement* e = static_cast< const SimpleElement* >( eIn );

	do {
		void( *rf )( const Conn&, T1, T2, T3 ) = 
			reinterpret_cast< void ( * )( const Conn&, T1, T2, T3 ) >(
				e->srcRecvFunc( src ) ); 
		/// \todo Are there STL binders for 2 values?
		vector< Conn >::const_iterator j;
		for ( j = e->connSrcBegin( src );
						j != e->connSrcEnd( src ); j++ )
				rf( *j, val1, val2, val3 );
		src = e->nextSrc( src );
	} while ( src != 0 );
}

/**
 * This templated function sends a three-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T1, class T2, class T3 > void sendTo3(
		const Element* eIn, unsigned int src, unsigned int conn,
		T1 val1, T2 val2, T3 val3 )
{
	const SimpleElement* e = static_cast< const SimpleElement* >( eIn );
	void( *rf )( const Conn&, T1, T2, T3 ) = 
			reinterpret_cast< void ( * )( const Conn&, T1, T2, T3 ) >(
			e->lookupRecvFunc( src, conn ) );
		rf( *( e->lookupConn( conn ) ),  val1, val2, val3 );
}

/**
 * This templated function sends four-argument messages.
 */
template < class T1, class T2, class T3, class T4 > void send4(
			const Element* eIn, unsigned int src,
			T1 val1, T2 val2, T3 val3, T4 val4 )
{
	const SimpleElement* e = static_cast< const SimpleElement* >( eIn );

	do {
		void( *rf )( const Conn&, T1, T2, T3, T4 ) = 
			reinterpret_cast< void ( * )( const Conn&, T1, T2, T3, T4 ) >(
				e->srcRecvFunc( src ) ); 
		/// \todo Are there STL binders for 2 values?
		vector< Conn >::const_iterator j;
		for ( j = e->connSrcBegin( src );
						j != e->connSrcEnd( src ); j++ )
				rf( *j, val1, val2, val3, val4 );
		src = e->nextSrc( src );
	} while ( src != 0 );
}

/**
 * This templated function sends a four-argument message to the
 * target specified by the conn argument. Note that this refers
 * to the index in the local conn_ vector.
 */
template< class T1, class T2, class T3, class T4 > void sendTo4(
		const Element* eIn, unsigned int src, unsigned int conn,
		T1 val1, T2 val2, T3 val3, T4 val4 )
{
	const SimpleElement* e = static_cast< const SimpleElement* >( eIn );
	void( *rf )( const Conn&, T1, T2, T3, T4 ) = 
			reinterpret_cast< void ( * )( const Conn&, T1, T2, T3, T4 ) >(
			e->lookupRecvFunc( src, conn ) );
		rf( *( e->lookupConn( conn ) ),  val1, val2, val3, val4 );
}

#endif // _SEND_H
