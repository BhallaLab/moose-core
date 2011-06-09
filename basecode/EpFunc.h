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
 * Utility function to return a pointer of the desired type from the 
 * Eref data. Mainly here because it lets us specialize for Neutrals,
 * below.
 */
template< class T > T* getEpFuncData( const Eref& e )
{
	return reinterpret_cast< T* >( e.data() ) ;
}

/**
 * This is a template specialization for GetEpFunc applied to Neutrals.
 * This is necessary in order to access Element fields of objects that 
 * may not have been allocated (such as synapses), even though their Element
 * has been created and needs to be manipulated.
 * Apparently regular functions with same args will be preferred
 * over templates, let's see if this works. Nope, it doesn't. Good try.
	extern Neutral* getEpFuncData( const Eref& e );
 * Try this form: it doesn't work either.
extern Neutral* dummyNeutral();
template<> Neutral* getEpFuncData< Neutral >( const Eref& e )
{
	return dummyNeutral();
}
 * Try externing the template itself.
 */
template<> Neutral* getEpFuncData< Neutral >( const Eref& e );



/**
 * This set of classes is derived from OpFunc, and take extra args
 * for the qinfo and Eref.
 */

template< class T > class EpFunc0: public OpFunc
{
	public:
		EpFunc0( void ( T::*func )( const Eref& e, const Qinfo* q ) )
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

		void op( const Eref& e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			( getEpFuncData< T >( e )->*func_ )( e, q );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q ); 
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			( getEpFuncData< T >( e )->*func_ )( e, q );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q ); 
		}

		string rttiType() const {
			return "void";
		}

	private:
		void ( T::*func_ )( const Eref& e, const Qinfo* q ); 
};

template< class T, class A > class EpFunc1: public OpFunc
{
	public:
		EpFunc1( void ( T::*func )( const Eref& e, const Qinfo* q, A ) )
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

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( const Eref& e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			Conv< A > arg1( buf + sizeof( Qinfo ) );
			( getEpFuncData< T >( e )->*func_ )( e, q, *arg1 );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1 ) ;
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A > arg1( buf );
			( getEpFuncData< T >( e )->*func_ )( e, q, *arg1 );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1 ) ;
		}

		string rttiType() const {
			return Conv< A >::rttiType();
		}

	private:
		void ( T::*func_ )( const Eref& e, const Qinfo* q, A ); 
};

template< class T, class A1, class A2 > class EpFunc2: public OpFunc
{
	public:
		EpFunc2( void ( T::*func )( const Eref& e, const Qinfo* q, A1, A2 ) )
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

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( const Eref& e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			( getEpFuncData< T >( e )->*func_ )( e, q, *arg1, *arg2 );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2 ) ;
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			( getEpFuncData< T >( e )->*func_ )( e, q, *arg1, *arg2 );
		// 	(reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2 ) ;
		}

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType(); 
		}

	private:
		void ( T::*func_ )( const Eref& e, const Qinfo* q, A1, A2 ); 
};

template< class T, class A1, class A2, class A3 > class EpFunc3:
	public OpFunc
{
	public:
		EpFunc3( void ( T::*func )( const Eref& e, const Qinfo* q, A1, A2, A3 ) )
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

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( const Eref& e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			Conv< A3 > arg3( buf + arg1.size() + arg2.size() );
			( getEpFuncData< T >( e )->*func_ )( e, q, *arg1, *arg2, *arg3);
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2, *arg3 ) ;
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			Conv< A3 > arg3( buf + arg1.size() + arg2.size() );
			( getEpFuncData< T >( e )->*func_ )( e, q, *arg1, *arg2, *arg3);
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2, *arg3 ) ;
		}

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType() +
				"," + Conv< A3 >::rttiType();
		}

	private:
		void ( T::*func_ )( const Eref& e, const Qinfo* q, A1, A2, A3 ); 
};

template< class T, class A1, class A2, class A3, class A4 > class EpFunc4:
	public OpFunc
{
	public:
		EpFunc4( void ( T::*func )( const Eref& e, const Qinfo* q, A1, A2, A3, A4 ) )
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

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( const Eref& e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			Conv< A3 > arg3( buf + arg1.size() + arg2.size() );
			Conv< A4 > arg4( buf + arg1.size() + arg2.size() + arg3.size());
			( getEpFuncData< T >( e )->*func_ )( e, q, 
				*arg1, *arg2, *arg3, *arg4);
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2, *arg3, *arg4 ) ;
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			Conv< A1 > arg1( buf );
			Conv< A2 > arg2( buf + arg1.size() );
			Conv< A3 > arg3( buf + arg1.size() + arg2.size() );
			Conv< A4 > arg4( buf + arg1.size() + arg2.size() + arg3.size());
			( getEpFuncData< T >( e )->*func_ )( e, q, 
				*arg1, *arg2, *arg3, *arg4);
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2, *arg3, *arg4 ) ;
		}

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType() +
				"," + Conv<A3>::rttiType() + "," + Conv<A4>::rttiType();
		}

	private:
		void ( T::*func_ )( const Eref& e, const Qinfo* q, A1, A2, A3, A4 ); 
};

template< class T, class A1, class A2, class A3, class A4, class A5 > class EpFunc5:
	public OpFunc
{
	public:
		EpFunc5( void ( T::*func )( const Eref& e, const Qinfo* q, A1, A2, A3, A4, A5 ) )
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

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( const Eref& e, const char* buf ) const {
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
			( getEpFuncData< T >( e )->*func_ )( e, q, 
				*arg1, *arg2, *arg3, *arg4, *arg5 );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2, *arg3, *arg4, *arg5 ) ;
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
			( getEpFuncData< T >( e )->*func_ )( e, q, 
				*arg1, *arg2, *arg3, *arg4, *arg5 );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2, *arg3, *arg4, *arg5 ) ;
		}

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType() +
				"," + Conv<A3>::rttiType() + "," + Conv<A4>::rttiType() +
				"," + Conv<A5>::rttiType();
		}

	private:
		void ( T::*func_ )( const Eref& e, const Qinfo* q, A1, A2, A3, A4, A5 ); 
};

template< class T, class A1, class A2, class A3, class A4, class A5, class A6 > class EpFunc6:
	public OpFunc
{
	public:
		EpFunc6( void ( T::*func )( const Eref& e, const Qinfo* q, A1, A2, A3, A4, A5, A6 ) )
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

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( const Eref& e, const char* buf ) const {
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
			buf += arg5.size();
			Conv< A6 > arg6( buf );
			( getEpFuncData< T >( e )->*func_ )( e, q, 
				*arg1, *arg2, *arg3, *arg4, *arg5, *arg6 );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2, *arg3, *arg4, *arg5, *arg6 ) ;
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
			( getEpFuncData< T >( e )->*func_ )( e, q, 
				*arg1, *arg2, *arg3, *arg4, *arg5, *arg6 );
			// (reinterpret_cast< T* >( e.data() )->*func_)( e, q, *arg1, *arg2, *arg3, *arg4, *arg5, *arg6 ) ;
		}

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType() +
				"," + Conv<A3>::rttiType() + "," + Conv<A4>::rttiType() +
				"," + Conv<A5>::rttiType() + "," + Conv<A6>::rttiType();
		}

	private:
		void ( T::*func_ )( const Eref& e, const Qinfo* q, 
			A1, A2, A3, A4, A5, A6 ); 
};

/**
 * This specialized EpFunc is for returning a single field value.
 * Unlike the regular GetOpFunc, this variant takes the Eref
 * and Qinfo.
 * It generates an opFunc that takes a single argument:
 * FuncId of the function on the object that requested the
 * value. The EpFunc then sends back a message with the info.
 */
template< class T, class A > class GetEpFunc: public GetOpFuncBase< A >
{
	public:
		GetEpFunc( A ( T::*func )( const Eref& e, const Qinfo* q ) const )
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
			op( e, q, buf );
		}

		void op( const Eref& e, const Qinfo* q, const char* buf ) const {
			if ( skipWorkerNodeGlobal( e ) )
				return;
			const A& ret = ( getEpFuncData< T >( e )->*func_ )( e, q );
				// (( reinterpret_cast< T* >( e.data() ) )->*func_)( e, q );
			Conv<A> conv0( ret );
			char* temp0 = new char[ conv0.size() ];
			conv0.val2buf( temp0 );
			fieldOp( e, q, buf, temp0, conv0.size() );
			delete[] temp0;
		}

		A reduceOp( const Eref& e ) const {
			Qinfo q; // Dummy.
			return ( getEpFuncData< T >( e )->*func_ )( e, &q );
			// return ( reinterpret_cast< T* >( e.data() )->*func_)( e, &q );
		}

	private:
		A ( T::*func_ )( const Eref& e, const Qinfo* q ) const;
};


/**
 * This specialized EpFunc is for returning a single field value,
 * but the field lookup requires an index argument as well.
 * Unlike the regular GetOpFunc, this variant takes the Eref
 * and Qinfo.
 * It generates an opFunc that takes a single argument:
 * FuncId of the function on the object that requested the
 * value. The EpFunc then sends back a message with the info.
 */
template< class T, class L, class A > class GetEpFunc1: public GetOpFuncBase< A >
{
	public:
		GetEpFunc1( A ( T::*func )( const Eref& e, const Qinfo* q, L ) const )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo1< A >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const LookupField< L, A >* >( s );
		}

		bool strSet( const Eref& tgt,
			const string& field, const string& arg ) const {
			return SetGet1< A >::innerStrSet( tgt.objId(), field, arg );
		}

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
				( getEpFuncData< T >( e )->*func_ )( e, q, *conv1 );

			// const A& ret = (( reinterpret_cast< T* >( e.data() ) )->*func_)( e, q, *conv1 );
			Conv<A> conv0( ret );
			char* temp0 = new char[ conv0.size() ];
			conv0.val2buf( temp0 );
			fieldOp( e, q, buf, temp0, conv0.size() );
			delete[] temp0;
		}

		/// ReduceOp not permissible.
		A reduceOp( const Eref& e ) const {
			static A ret;
			return ret;
		}

	private:
		A ( T::*func_ )( const Eref& e, const Qinfo* q, L ) const;
};


#endif //_EPFUNC_H
