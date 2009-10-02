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
		Slot( ConnId conn, FuncId func );
		virtual unsigned int numArgs() const
			{ return 0; }

	protected:
		ConnId conn_;
		FuncId func_;
};

class Slot0: public Slot
{
	public:
		Slot0( ConnId conn, const Cinfo* c, const string& funcName );
		void send( Eref e );
		void sendTo( Eref e, Id target);
};

template< class T > class Slot1: public Slot
{
	public:
		Slot1( ConnId conn, const Cinfo* c, const string& funcName )
			: Slot( conn, c, funcName )
		{ ; }

		Slot1( ConnId conn, FuncId func )
			: Slot( conn, func )
		{ ; }


		// Will need to specialize for strings etc.
		void send( Eref e, const T& arg ) {
			e.asend( conn_, func_, reinterpret_cast< const char* >( &arg ), sizeof( T ) );
		}

		void sendTo( Eref e, Id target, const T& arg ) {
			char temp[ sizeof( T ) + sizeof( unsigned int ) ];
			*reinterpret_cast< T* >( temp ) = arg;
			*reinterpret_cast< unsigned int* >( temp + sizeof( T ) ) = target.index();
			e.tsend( conn_, func_, target, reinterpret_cast< const char* >( &arg ), sizeof( T ) );
		}
};


template< class T1, class T2 > class Slot2: public Slot
{
	public:
		Slot2( ConnId conn, const Cinfo* c, const string& funcName )
			: Slot( conn, c, funcName )
		{ ; }

		Slot2( ConnId conn, FuncId func )
			: Slot( conn, func )
		{ ; }

		void send( Eref e, const T1& arg1, const T2& arg2 ) {
			char temp[ sizeof( T1 ) + sizeof( T2 ) ];
			*reinterpret_cast< T1* >( temp ) = arg1;
			*reinterpret_cast< T2* >( temp + sizeof( T1 ) ) = arg2;
			e.asend( conn_, func_, temp, sizeof( T1 ) + sizeof( T2 ) );
		}

		void sendTo( Eref e, Id target, const T1& arg1, const T2& arg2 )
		{;}
};


template< class T1, class T2, class T3 > class Slot3: public Slot
{
	public:
		Slot3( ConnId conn, const Cinfo* c, const string& funcName )
			: Slot( conn, c, funcName )
		{ ; }
		Slot3( ConnId conn, FuncId func )
			: Slot( conn, func )
		{ ; }

		void send( Eref e, const T1& arg1, T2& arg2, T3& arg3 ) {
			char temp[ sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 ) ];
			*reinterpret_cast< T1* >( temp ) = arg1;
			*reinterpret_cast< T2* >( temp + sizeof( T1 ) ) = arg2;
			*reinterpret_cast< T3* >( temp + sizeof( T1 ) + sizeof( T2) ) =
				arg3;
			e.asend( conn_, func_, temp, sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 ) );
		}
		void sendTo( Eref e, Id target, 
			const T1& arg1, const T2& arg2, const T3& arg3 )
		{;}
};


#endif // _SLOT_H
