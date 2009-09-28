/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SLOT_H
#define _SLOT_H

/**
 * Slots can be statically defined, for cases where the ConnId is 
 * hard-coded. For other messages, the Slot has to be set up on the
 * fly.
 */

class Slot
{
	public:
		Slot( ConnId conn, const Cinfo* c, const string& funcName );

	protected:
		FuncId func_;
		ConnId conn_;
};

class Slot0: public Slot
{
	public:
		Slot0( ConnId conn, const Cinfo* c, const string& funcName );
		void send( Eref e );
};

template< class T > class Slot1: public Slot
{
	public:
		Slot1( ConnId conn, const Cinfo* c, const string& funcName )
			: Slot( conn, c, funcName )
		{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const T& arg ) {
			e.asend( conn_, func_, reinterpret_cast< const char* >( &arg ), sizeof( T ) );
		}
};


template< class T1, class T2 > class Slot2: public Slot
{
	public:
		Slot2( ConnId conn, const Cinfo* c, const string& funcName )
			: Slot( conn, c, funcName )
		{ ; }

		void send( Eref e, const T1& arg1, T2& arg2 ) {
			char temp[ sizeof( T1 ) + sizeof( T2 ) ];
			*reinterpret_cast< T1* >( temp ) = arg1;
			*reinterpret_cast< T2* >( temp + sizeof( T1 ) ) = arg1;
			e.asend( conn_, func_, temp, sizeof( T1 ) + sizeof( T2 ) );
		}
};

#endif // _SLOT_H
