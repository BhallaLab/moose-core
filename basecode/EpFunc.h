/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _EPFUNC_H
#define _EPFUNC_H
/**
 * This set of classes is derived from OpFunc, and take extra args
 * for the qinfo and Eref.
 */

template< class T > class EpFunc0: public OpFunc
{
	public:
		EpFunc0( void ( T::*func )( Eref e, const Qinfo* q ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo0* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet0* >( s );
		}

		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q ); 
		}

	private:
		void ( T::*func_ )( Eref e, const Qinfo* q ); 
};

template< class T, class A > class EpFunc1: public OpFunc
{
	public:
		EpFunc1( void ( T::*func )( Eref e, const Qinfo* q, A ) )
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
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			Conv< A > arg1( buf + sizeof( Qinfo ) );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1 ) ;
		}

	private:
		void ( T::*func_ )( Eref e, const Qinfo* q, A ); 
};

/*
template< class T > class RetFunc: public OpFunc
{
	public:
		RetFunc( void ( T::*func )( Eref e, const Qinfo* q, const char* arg ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return 1;
		}

		bool checkSet( const SetGet* s ) const {
			return 1;
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q, buf + sizeof( Qinfo ) ) ;
		}

	private:
		void ( T::*func_ )( Eref e, const Qinfo* q, const char* arg ); 
};
*/


template< class T, class A1, class A2 > class EpFunc2: public OpFunc
{
	public:
		EpFunc2( void ( T::*func )( Eref e, const Qinfo* q, A1, A2 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo2< A1, A2 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet2< A1, A2 >* >( s );
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2 ) ;
		}

	private:
		void ( T::*func_ )( Eref e, const Qinfo* q, A1, A2 ); 
};

template< class T, class A1, class A2, class A3 > class EpFunc3:
	public OpFunc
{
	public:
		EpFunc3( void ( T::*func )( Eref e, const Qinfo* q, A1, A2, A3 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo3< A1, A2, A3 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet3< A1, A2, A3 >* >( s );
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			Conv< A3 > arg3( buf + arg1.size() + arg2.size() );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q, 
				*arg1, *arg2, *arg3 ) ;
		}

	private:
		void ( T::*func_ )( Eref e, const Qinfo* q, A1, A2, A3 ); 
};

template< class T, class A1, class A2, class A3, class A4 > class EpFunc4:
	public OpFunc
{
	public:
		EpFunc4( void ( T::*func )( Eref e, const Qinfo* q, A1, A2, A3, A4 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo4< A1, A2, A3, A4 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet4< A1, A2, A3, A4 >* >( s );
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			Conv< A3 > arg3( buf + arg1.size() + arg2.size() );
			Conv< A4 > arg4( buf + arg1.size() + arg2.size() + arg3.size());
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q, 
				*arg1, *arg2, *arg3, *arg4 ) ;
		}

	private:
		void ( T::*func_ )( Eref e, const Qinfo* q, A1, A2, A3, A4 ); 
};

template< class T, class A1, class A2, class A3, class A4, class A5 > class EpFunc5:
	public OpFunc
{
	public:
		EpFunc5( void ( T::*func )( Eref e, const Qinfo* q, A1, A2, A3, A4, A5 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo5< A1, A2, A3, A4, A5 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet5< A1, A2, A3, A4, A5 >* >( s );
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			buf += arg1.size();
			Conv< A2 > arg2( buf );
			buf += arg2.size();
			Conv< A3 > arg3( buf );
			buf += arg3.size();
			Conv< A4 > arg4( buf );
			buf += arg4.size();
			Conv< A5 > arg5( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q, 
				*arg1, *arg2, *arg3, *arg4, *arg5 ) ;
		}

	private:
		void ( T::*func_ )( Eref e, const Qinfo* q, A1, A2, A3, A4, A5 ); 
};

/**
 * This specialized EpFunc is for returning a single field value.
 * Unlike the regular GetOpFunc, this variant takes the Eref
 * and Qinfo.
 * It generates an opFunc that takes a single argument:
 * FuncId of the function on the object that requested the
 * value. The EpFunc then sends back a message with the info.
 */
template< class T, class A > class GetEpFunc: public OpFunc
{
	public:
		GetEpFunc( A ( T::*func )( Eref e, const Qinfo* q ) const )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo1< A >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet1< A >* >( s );
		}

		/**
		 * The buf just contains the funcid on the src element that is
		 * ready to receive the returned data.
		 * Also we are returning the data along the Msg that brought in
		 * the request, so we don't need to scan through all Msgs in
		 * the Element to find the right one.
		 * So we bypass the usual SrcFinfo::sendTo, and instead go
		 * right to the Qinfo::addToQ to send off data.
		 * Finally, the data is copied back-and-forth about 3 times.
		 * Wasteful, but the 'get' function is not to be heavily used.
		 */
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			const A& ret = 
				(( reinterpret_cast< T* >( e.data() ) )->*func_)( e, q );
			Conv<A> conv0( ret );
			char* temp0 = new char[ conv0.size() ];
			conv0.val2buf( temp0 );
			fieldOp( e, buf, temp0, conv0.size() );
			delete[] temp0;
		}

	private:
		A ( T::*func_ )( Eref e, const Qinfo* q ) const;
};

/**
 * This specialized EpFunc is for looking up a single field value using
 * an argument such as a name or an index.
 * It generates an opFunc that takes a single argument:
 * FuncId of the function on the object that requested the
 * value. The EpFunc then sends back a message with the info.
 * After analyzing the call structure it seems like we can't get the
 * argument in to this through the Set/Get format. The Shell, which
 * dispatches the request, doesn't have the hooks to do this.
 * So we roll this back for now.
 *
template< class T, class A, class L > class LookupEpFunc: public OpFunc
{
	public:
		LookupEpFunc( A ( T::*func )( Eref e, const Qinfo* q, L index ) const )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo1< A >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet1< A >* >( s );
		}

		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			Conv< L > index( buf );

			const A& ret = 
				(( reinterpret_cast< T* >( e.data() ) )->*func_)( e, q,
					*index );
			Conv<A> conv0( ret );
			char* temp0 = new char[ conv0.size() ];
			conv0.val2buf( temp0 );
			fieldOp( e, buf, temp0, conv0.size() );
			delete[] temp0;
		}

	private:
		A ( T::*func_ )( Eref e, const Qinfo* q, L index ) const;
};
*/

#endif //_EPFUNC_H
