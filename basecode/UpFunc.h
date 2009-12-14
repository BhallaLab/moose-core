/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _UPFUNC_H
#define _UPFUNC_H

/**
 * This set of OpFuncs don't operate directly on the Eref::data(), but
 * one level up, on Eref::data1(). This happens specially in synaptic
 * messages, where there is an array of synapses on each target neuron
 * or channel. In these cases we want to do the operation on Eref::data1(),
 * and pass in the index so that the function can decide which synapse to
 * work on.
 */

template< class T > class UpFunc0: public OpFunc
{
	public:
		UpFunc0( void ( T::*func )( DataId index ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo0* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet0* >( s );
		}

		/**
		 * Calls function on first dimension of data, using index as
		 * argument.
		 */
		void op( Eref e, const char* buf ) const {
			(reinterpret_cast< T* >( e.data1() )->*func_)( e.index() );
		}
	private:
		void ( T::*func_ )( DataId index ); 
};

template< class T, class A > class UpFunc1: public OpFunc
{
	public:
		UpFunc1( void ( T::*func )( DataId, A ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo1< A >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet1< A >* >( s );
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		// buf is organized as Qinfo, args, optionally srcIndex.
		void op( Eref e, const char* buf ) const {
			A val;
			Conv< A >::buf2val( val, buf + sizeof( Qinfo ) );
			(reinterpret_cast< T* >( e.data1() )->*func_)( e.index(), val );
		}

	private:
		void ( T::*func_ )( DataId index, const A ); 
};

template< class T, class A1, class A2 > class UpFunc2: public OpFunc
{
	public:
		UpFunc2( void ( T::*func )( DataId, A1, A2 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo2< A1, A2 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet2< A1, A2 >* >( s );
		}

		void op( Eref e, const char* buf ) const {
			buf += sizeof( Qinfo );
			const char* buf2 = buf + sizeof( A1 );
			(reinterpret_cast< T* >( e.data1() )->*func_)( 
				e.index(),
				*reinterpret_cast< const A1* >( buf ),
				*reinterpret_cast< const A2* >( buf2 )
			);
		}

	private:
		void ( T::*func_ )( DataId, A1, A2 ); 
};

template< class T, class A1, class A2, class A3 > class UpFunc3: 
	public OpFunc
{
	public:
		UpFunc3( void ( T::*func )( DataId index, A1, A2, A3 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo3< A1, A2, A3 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet3< A1, A2, A3 >* >( s );
		}

		void op( Eref e, const char* buf ) const {
			buf += sizeof( Qinfo );
			const char* buf2 = buf + sizeof( A1 );
			const char* buf3 = buf2 + sizeof( A2 );
			(reinterpret_cast< T* >( e.data1() )->*func_)( 
				e.index(),
				*reinterpret_cast< const A1* >( buf ),
				*reinterpret_cast< const A2* >( buf2 ),
				*reinterpret_cast< const A3* >( buf3 )
			);
		}

	private:
		void ( T::*func_ )( DataId, A1, A2, A3 ); 
};

/**
 * This specialized UpFunc is for returning a single field value
 * for an array field.
 * It generates an UpFunc that takes an extra argument:
 * FuncId of the function on the object that requested the
 * value. The UpFunc then sends back a message with the info.
 */
template< class T, class A > class GetUpFunc: public OpFunc
{
	public:
		GetUpFunc( A ( T::*func )( DataId ) const )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo1< A >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet1< FuncId >* >( s );
		}

		/**
		 * The buf just contains the funcid on the src element that is
		 * ready to receive the returned data.
		 * In this special case we do not do typechecking, since the
		 * constructors for the get command should have done so already.
		 * So we bypass the usual SrcFinfo::sendTo, and instead go
		 * right to the Conn to send the data.
		 */
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
		    FuncId retFunc = *reinterpret_cast< const FuncId* >( buf );
			const A& ret = (( reinterpret_cast< T* >( e.data1() ) )->*func_)( e.index() );

			Qinfo retq( retFunc, e.index(), Conv< A >::size( ret ), 
				1, !q->isForward() );
			char* temp = new char[ retq.size() ];
			Conv<A>::val2buf( temp, ret );
			Conn c;
			// c.add( const_cast< Msg* >( e.element()->getMsg( q->mid() ) ) );
			c.add( q->mid() ); 
			c.tsend( e.element(), q->srcIndex(), retq, 
				Shell::procInfo(), temp );
			delete[] temp;
		}

	private:
		A ( T::*func_ )( DataId ) const;
};


#endif // _UPFUNC_H
