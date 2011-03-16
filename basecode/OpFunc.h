/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _OPFUNC_H
#define _OPFUNC_H

extern void fieldOp( const Eref& e, const Qinfo* q, const char* buf, 
	const char* data, unsigned int size );
extern bool skipWorkerNodeGlobal( const Eref& e );

class OpFunc
{
	public:
		virtual ~OpFunc()
		{;}
		virtual bool checkFinfo( const Finfo* s) const = 0;
		virtual bool checkSet( const SetGet* s) const = 0;

		/**
		 * Helper function for finding the correct type of SetGet template
		 * in order to do the assignment.
		 */
		virtual bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const = 0;

		virtual void op( const Eref& e, const char* buf ) const = 0;

		virtual void op( const Eref& e, const Qinfo* q, const char* buf ) const = 0;
};

template< class A > class GetOpFuncBase: public OpFunc
{
	public: 
		virtual A reduceOp( const Eref& e ) const = 0;
		
};

// Should I template these off an integer for generating a family?
class OpFuncDummy: public OpFunc
{
	public:
		OpFuncDummy();
		bool checkFinfo( const Finfo* s) const;
		bool checkSet( const SetGet* s) const;

		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const;

		void op( const Eref& e, const char* buf ) const;
		void op( const Eref& e, const Qinfo* q, const char* buf ) const;
};

template< class T > class OpFunc0: public OpFunc
{
	public:
		OpFunc0( void ( T::*func )( ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo0* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet0* >( s );
		}

		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet0::innerStrSet( tgt.objId(), field, arg );
		}

		/**
		 * Call function on T located at e.data(), which is a simple 
		 * array lookup of the data_ vector using the Eref index.
		 */
		void op( const Eref& e, const char* buf ) const {
			(reinterpret_cast< T* >( e.data() )->*func_)( );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			(reinterpret_cast< T* >( e.data() )->*func_)();
		}

	private:
		void ( T::*func_ )( ); 
};

template< class T, class A > class OpFunc1: public OpFunc
{
	public:
		OpFunc1( void ( T::*func )( const A ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo1< A >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet1< A >* >( s );
		}
		
		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet1< A >::innerStrSet( tgt.objId(), field, arg );
		}

		void op( const Eref& e, const char* buf ) const {
			Conv< A > arg1( buf + sizeof( Qinfo ) );
			(reinterpret_cast< T* >( e.data() )->*func_)( *arg1 );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A > arg1( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( *arg1 );
		}

	private:
		void ( T::*func_ )( A ); 
};

template< class T, class A1, class A2 > class OpFunc2: public OpFunc
{
	public:
		OpFunc2( void ( T::*func )( A1, A2 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo2< A1, A2 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet2< A1, A2 >* >( s );
		}
		
		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet2< A1, A2 >::innerStrSet( tgt.objId(), field, arg );
		}

		void op( const Eref& e, const char* buf ) const {
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			(reinterpret_cast< T* >( e.data() )->*func_)( *arg1, *arg2 );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			(reinterpret_cast< T* >( e.data() )->*func_)( *arg1, *arg2 );
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

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo3< A1, A2, A3 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet3< A1, A2, A3 >* >( s );
		}
		
		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet3< A1, A2, A3 >::innerStrSet( tgt.objId(), field, arg );
		}

		void op( const Eref& e, const char* buf ) const {
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			buf += arg1.size();
			Conv< A2 > arg2( buf );
			buf += arg2.size();
			Conv< A3 > arg3( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*arg1, *arg2, *arg3 );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A1 > arg1( buf );
			buf += arg1.size();
			Conv< A2 > arg2( buf );
			buf += arg2.size();
			Conv< A3 > arg3( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*arg1, *arg2, *arg3 );
		}

	private:
		void ( T::*func_ )( A1, A2, A3 ); 
};

template< class T, class A1, class A2, class A3, class A4 > class OpFunc4: 
	public OpFunc
{
	public:
		OpFunc4( void ( T::*func )( A1, A2, A3, A4 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo4< A1, A2, A3, A4 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet4< A1, A2, A3, A4 >* >( s );
		}
		
		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet4< A1, A2, A3, A4 >::innerStrSet( tgt.objId(), field, arg );
		}

		void op( const Eref& e, const char* buf ) const {
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			buf += arg1.size();
			Conv< A2 > arg2( buf );
			buf += arg2.size();
			Conv< A3 > arg3( buf );
			buf += arg3.size();
			Conv< A4 > arg4( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*arg1, *arg2, *arg3, *arg4 );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A1 > arg1( buf );
			buf += arg1.size();
			Conv< A2 > arg2( buf );
			buf += arg2.size();
			Conv< A3 > arg3( buf );
			buf += arg3.size();
			Conv< A4 > arg4( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*arg1, *arg2, *arg3, *arg4 );
		}

	private:
		void ( T::*func_ )( A1, A2, A3, A4 ); 
};

template< class T, class A1, class A2, class A3, class A4, class A5 > class OpFunc5: 
	public OpFunc
{
	public:
		OpFunc5( void ( T::*func )( A1, A2, A3, A4, A5 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo5< A1, A2, A3, A4, A5 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet5< A1, A2, A3, A4, A5 >* >( s );
		}
		
		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet5< A1, A2, A3, A4, A5 >::innerStrSet( tgt.objId(), field, arg );
		}

		void op( const Eref& e, const char* buf ) const {
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
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*arg1, *arg2, *arg3, *arg4, *arg5 );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A1 > arg1( buf );
			buf += arg1.size();
			Conv< A2 > arg2( buf );
			buf += arg2.size();
			Conv< A3 > arg3( buf );
			buf += arg3.size();
			Conv< A4 > arg4( buf );
			buf += arg4.size();
			Conv< A5 > arg5( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*arg1, *arg2, *arg3, *arg4, *arg5 );
		}

	private:
		void ( T::*func_ )( A1, A2, A3, A4, A5 ); 
};

template< class T, class A1, class A2, class A3, class A4, class A5, class A6 > class OpFunc6: 
	public OpFunc
{
	public:
		OpFunc6( void ( T::*func )( A1, A2, A3, A4, A5, A6 ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo6< A1, A2, A3, A4, A5, A6 >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet6< A1, A2, A3, A4, A5, A6 >* >( s );
		}
		
		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet6< A1, A2, A3, A4, A5, A6 >::innerStrSet( tgt.objId(), field, arg );
		}

		void op( const Eref& e, const char* buf ) const {
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
			buf += arg5.size();
			Conv< A6 > arg6( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*arg1, *arg2, *arg3, *arg4, *arg5, *arg6 );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A1 > arg1( buf );
			buf += arg1.size();
			Conv< A2 > arg2( buf );
			buf += arg2.size();
			Conv< A3 > arg3( buf );
			buf += arg3.size();
			Conv< A4 > arg4( buf );
			buf += arg4.size();
			Conv< A5 > arg5( buf );
			buf += arg5.size();
			Conv< A6 > arg6( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*arg1, *arg2, *arg3, *arg4, *arg5, *arg6 );
		}

	private:
		void ( T::*func_ )( A1, A2, A3, A4, A5, A6 ); 
};

/**
 * This specialized OpFunc is for returning a single field value
 * It generates an opFunc that takes a single argument:
 * FuncId of the function on the object that requested the
 * value. The OpFunc then sends back a message with the info.
 */
template< class T, class A > class GetOpFunc: public GetOpFuncBase< A >
{
	public:
		GetOpFunc( A ( T::*func )() const )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return ( dynamic_cast< const SrcFinfo1< A >* >( s )
			|| dynamic_cast< const SrcFinfo1< FuncId >* >( s ) );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const Field< A >* >( s );
		}

		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet1< A >::innerStrSet( tgt.objId(), field, arg );
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
		void op( const Eref& e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			this->op( e, q, buf );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			if ( skipWorkerNodeGlobal( e ) )
				return;
			const A& ret = 
				(( reinterpret_cast< T* >( e.data() ) )->*func_)();
			Conv<A> conv0( ret );
			char* temp0 = new char[ conv0.size() ];
			conv0.val2buf( temp0 );
			fieldOp( e, q, buf, temp0, conv0.size() );
			delete[] temp0;
		}

		A reduceOp( const Eref& e ) const {
			return ( reinterpret_cast< T* >( e.data() )->*func_)();
		}

	private:
		A ( T::*func_ )() const;
};

/**
 * This specialized OpFunc is for looking up a single field value
 * using a single argument.
 * It generates an opFunc that takes two arguments:
 * 1. FuncId of the function on the object that requested the value. 
 * 2. Index or other identifier to do the look up.
 * The OpFunc then sends back a message with the info.
 * Here T is the class that owns the function.
 * A is the return type
 * L is the lookup index.
 */
template< class T, class A, class L > class GetOpFunc1: public GetOpFuncBase< A >
{
	public:
		GetOpFunc1( A ( T::*func )( L ) const )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return ( dynamic_cast< const SrcFinfo1< A >* >( s )
			|| dynamic_cast< const SrcFinfo2< FuncId, L >* >( s ) );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const LookupField< L, A >* >( s );
			// Need to modify in case a message is coming in.
		}

		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const {
			return SetGet1< A >::innerStrSet( tgt.objId(), field, arg );
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
		void op( const Eref& e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			this->op( e, q, buf );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			if ( skipWorkerNodeGlobal( e ) )
				return;
			Conv< L > conv1( buf + sizeof( FuncId ) );
			const A& ret = 
				(( reinterpret_cast< T* >( e.data() ) )->*func_)( *conv1 );
			Conv<A> conv0( ret );
			char* temp0 = new char[ conv0.size() ];
			conv0.val2buf( temp0 );
			fieldOp( e, q, buf, temp0, conv0.size() );
			delete[] temp0;
		}

		/// ReduceOp is not really permissible for this class.
		A reduceOp( const Eref& e ) const {
			L dummy;
			return ( reinterpret_cast< T* >( e.data() )->*func_)( dummy );
		}

	private:
		A ( T::*func_ )( L ) const;
};

#endif // _OPFUNC_H
