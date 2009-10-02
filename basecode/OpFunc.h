/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class OpFunc
{
	public:
		virtual ~OpFunc()
		{;}
		virtual bool checkSlot( const Slot* s) const = 0;
		virtual void op( Eref e, const char* buf ) const = 0;
};

template< class T > class OpFunc0: public OpFunc
{
	public:
		OpFunc0( void ( T::*func )( ) )
			: func_( func )
			{;}

		bool checkSlot( const Slot* s ) const {
			return dynamic_cast< const Slot0* >( s );
		}

		void op( Eref e, const char* buf ) const {
			(static_cast< T* >( e.data() )->*func_)( );
		}

	private:
		void ( T::*func_ )( ); 
};

template< class T, class A > class OpFunc1: public OpFunc
{
	public:
		OpFunc1( void ( T::*func )( const A& ) )
			: func_( func )
			{;}

		bool checkSlot( const Slot* s ) const {
			return dynamic_cast< const Slot1< A >* >( s );
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( Eref e, const char* buf ) const {
			A val;
			Conv< A >::buf2val( val, buf );
			(static_cast< T* >( e.data() )->*func_)( val ) ;
		}

	private:
		void ( T::*func_ )( const A& ); 
};

template< class T, class A1, class A2 > class OpFunc2: public OpFunc
{
	public:
		OpFunc2( void ( T::*func )( A1, A2 ) )
			: func_( func )
			{;}

		bool checkSlot( const Slot* s ) const {
			return dynamic_cast< const Slot2< A1, A2 >* >( s );
		}

		void op( Eref e, const char* buf ) const {
			const char* buf2 = buf + sizeof( A1 );
			(static_cast< T* >( e.data() )->*func_)( 
				*reinterpret_cast< const A1* >( buf ),
				*reinterpret_cast< const A2* >( buf2 )
			);
		}

	private:
		void ( T::*func_ )( A1, A2 ); 
};

template< class T, class A1, class A2, class A3 > class OpFunc3: 
	public OpFunc
{
	public:
		OpFunc3( void ( T::*func )( A1, A2, A3 ) )
			: func_( func )
			{;}

		bool checkSlot( const Slot* s ) const {
			return dynamic_cast< const Slot3< A1, A2, A3 >* >( s );
		}

		void op( Eref e, const char* buf ) const {
			const char* buf2 = buf + sizeof( A1 );
			const char* buf3 = buf2 + sizeof( A2 );
			(static_cast< T* >( e.data() )->*func_)( 
				*reinterpret_cast< const A1* >( buf ),
				*reinterpret_cast< const A2* >( buf2 ),
				*reinterpret_cast< const A3* >( buf3 )
			);
		}

	private:
		void ( T::*func_ )( A1, A2, A3 ); 
};


/**
 * This specialized OpFunc is for returning a single field value
 * It generates an opFunc that takes three arguments:
 * Id, MsgId and FuncId of the function on the object that requested the
 * value. The OpFunc then sends back a message with the info.
 */
template< class T, class A > class GetOpFunc: public OpFunc
{
	public:
		GetOpFunc( const A& ( T::*func )() const )
			: func_( func )
			{;}

		bool checkSlot( const Slot* s ) const {
			return dynamic_cast< const Slot1< A >* >( s );
		}

		void op( Eref e, const char* buf ) const {
			A ret = (( static_cast< T* >( e.data() ) )->*func_)();
		    Id src = *reinterpret_cast< const Id* >( buf );
		    buf += sizeof( Id );
		    MsgId srcMsg = *reinterpret_cast< const MsgId* >( buf );
		    buf += sizeof( MsgId );
		    FuncId srcFunc = *reinterpret_cast< const FuncId* >( buf );
		    Slot1< A > s( srcMsg, srcFunc );
		    s.sendTo( e, src, ret );
		}

	private:
		const A& ( T::*func_ )() const;
};
