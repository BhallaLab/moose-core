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

class OpFunc
{
	public:
		virtual ~OpFunc()
		{;}
		virtual bool checkFinfo( const Finfo* s) const = 0;
		virtual bool checkSet( const SetGet* s) const = 0;
		virtual void op( Eref e, const char* buf ) const = 0;
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

		/**
		 * Call function on T located at e.data(), which is a simple 
		 * array lookup of the data_ vector using the Eref index.
		 */
		void op( Eref e, const char* buf ) const {
			(reinterpret_cast< T* >( e.data() )->*func_)( );
		}

		/**
		 * Calls function on first dimension of data, using index as
		 * argument.
		void opUp1( Eref e, const char* buf ) const {
			(reinterpret_cast< T* >( e.data1() )->*func_)( e.index() );
		}
		 */

		/**
		 * Call function on T located at e.aData(). This comes from
		 * separating the Eref index into data and field parts. The
		 * data is looked up using the data part of the index, and then
		 * Data::field looks up a void pointer to the field. Assumes a
		 * single such field, it would seem.
		void arrayOp( Eref e, const char* buf ) const {
			(reinterpret_cast< T* >( e.aData() )->*func_)( );
		}

		void parentOp( Eref e, const char* buf ) const {
			(reinterpret_cast< T* >( e.aData() )->*func_)( );
		}
		 */

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

		void op( Eref e, const char* buf ) const {
			Conv< A > arg1( buf + sizeof( Qinfo ) );
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

		/*
		void op( Eref e, const char* buf ) const {
			buf += sizeof( Qinfo );
			const char* buf2 = buf + sizeof( A1 );
			(reinterpret_cast< T* >( e.data() )->*func_)( 
				*reinterpret_cast< const A1* >( buf ),
				*reinterpret_cast< const A2* >( buf2 )
			);
		}
		*/
		void op( Eref e, const char* buf ) const {
			buf += sizeof( Qinfo );
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

		void op( Eref e, const char* buf ) const {
			buf += sizeof( Qinfo );
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

		void op( Eref e, const char* buf ) const {
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

	private:
		void ( T::*func_ )( A1, A2, A3, A4 ); 
};


/**
 * This specialized OpFunc is for returning a single field value
 * It generates an opFunc that takes a single argument:
 * FuncId of the function on the object that requested the
 * value. The OpFunc then sends back a message with the info.
 */
template< class T, class A > class GetOpFunc: public OpFunc
{
	public:
		GetOpFunc( A ( T::*func )() const )
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
		 * Also we are returning the data along the Msg that brought in
		 * the request, so we don't need to scan through all Msgs in
		 * the Element to find the right one.
		 * So we bypass the usual SrcFinfo::sendTo, and instead go
		 * right to the Qinfo::addToQ to send off data.
		 */
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			buf += sizeof( Qinfo );
		    FuncId retFunc = *reinterpret_cast< const FuncId* >( buf );
			const A& ret = 
				(( reinterpret_cast< T* >( e.data() ) )->*func_)();

			// Flag arguments: useSendTo = 1, and flip the isForward flag.
			Conv<A> conv( ret );
			Qinfo retq( retFunc, e.index(), conv.size(), 
				1, !q->isForward() );
			char* temp = new char[ retq.size() ];
			conv.val2buf( temp ); 
			MsgFuncBinding mfb( q->mid(), retFunc );
			retq.addSpecificTargetToQ( Shell::procInfo()->outQid, mfb, 
				temp );
			delete[] temp;
		}

	private:
		A ( T::*func_ )() const;
};

#endif // _OPFUNC_H
