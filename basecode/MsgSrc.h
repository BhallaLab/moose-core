/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MSGSRC_H
#define _MSGSRC_H

#include <vector>
#include "header.h"

template <class T> void dummyFunc( Conn* c, T v)
{
	;
}

// Operator functor for msgsrcs to use with connection-arg msgfuncs.
template <class T> class Op1
{
	public:
		Op1(T value, void ( *f )( Conn*, T ) )
			: value_( value ), func_( f )
		{
			;
		}

		Op1( )
			: func_( 0 )
		{
			;
		}

		void operator()( Conn* conn ) const
		{
			func_( conn, value_ );
			/*
			cerr << "Op1::operator() with value = " << value_ << 
				", Conn->pa = " << conn->parent()->name() << "\n";
				*/
		}
		T value_;
	private:
		void ( *func_ )( Conn*, T );
};

// Operator functor for msgsrcs to use with connection-arg msgfuncs.
template < class T1, class T2 > class Op2
{
	public:
		Op2(T1 value1, T2 value2, void ( *f )( Conn*, T1, T2 ) )
			: value1_( value1 ), value2_( value2 ), func_( f )
		{
			;
		}
		void operator()( Conn* conn ) const
		{
			func_( conn, value1_, value2_ );
		}
		T1 value1_;
		T2 value2_;
	private:
		void ( *func_ )( Conn*, T1, T2 );
};

// Operator functor for msgsrcs to use with connection-arg msgfuncs.
template < class T1, class T2, class T3 > class Op3
{
	public:
		Op3(T1 value1, T2 value2, T3 value3,
			void ( *f )( Conn*, T1, T2, T3 ) )
			: value1_( value1 ), value2_( value2 ), value3_( value3 ),
			func_( f )
		{
			;
		}
		void operator()( Conn* conn ) const
		{
			func_( conn, value1_, value2_, value3_ );
		}
		T1 value1_;
		T2 value2_;
		T3 value3_;
	private:
		void ( *func_ )( Conn*, T1, T2, T3 );
};

//////////////////////////////////////////////////////////////////
//
//	Here we define a set of classes for managing message sources.
//	They basically consist of local storage for the function to
//	call on the target object, and a reference to a Conn object
//  on the parent element.
//
//  SingleMsgSrc: Base class for messages with a single target.
//	SingleMsgSrc0: Handles messages with zero arguments.
//	SingleMsgSrc1<T>: Handles messages with one argument.
//	MultiMsgSrc: Base class for messages with multiple targets.
//     This is a general class: the targets do not have to be of
//     the same class, that is, their recvFuncs can differ. It sorts
//     the outgoing connections according to recvFunc, so that all
//     outgoing messages using the same recvFunc are called 
//     sequentially.
//	MultiMsgSrc0: Messages with zero arguments, multiple targets.
//	MultiMsgSrc1<T>: Messages with one argument, multiple targets.
//
//////////////////////////////////////////////////////////////////

// In principle could implement these with the Conn coming from a
// templated function using 'this' as a starting point. Such
// an arrangement would save one pointer and is marginally faster.
// However, if we go to an element-based approach it won't work.
class SingleMsgSrc
{
	public:
		SingleMsgSrc( RecvFunc rf, UniConnBase* c)
			: rfunc_( rf ), c_( c )
		{
			;
		}

		bool add( RecvFunc rf, Conn* target ) {
			if ( c_->connect( target, 0 ) ) {
				rfunc_ = rf;
				return 1;
			}
			return 0;
		}

		bool funcDrop( const Conn* target ) {
			if ( c_ == target ) { // double check the match
				rfunc_ = dummyFunc0;
				return 1;
			}
			return 0;
		}

		void dropAll() {
			rfunc_ = dummyFunc0;
			c_->disconnectAll();
		}

		unsigned long nTargets() const {
			return c_->nTargets();
		}

		// Returns the number of matches.
		unsigned long matchRemoteFunc( RecvFunc func ) const {
			return ( func == rfunc_ );
		}

		RecvFunc targetFunc( unsigned long i ) const {
			return rfunc_;
		}

		RecvFunc targetFuncFromSlot( unsigned long slot ) const {
			return rfunc_;
		}

		unsigned long nFuncs() const {
			return ( ( rfunc_ != 0 ) && ( rfunc_ != dummyFunc0) );
		}

		Conn* conn() const {
			return c_;
		}

		void dest( std::vector< Field >& list );
		void addRecvFunc( RecvFunc rf, unsigned long position );
	
	protected:
		RecvFunc rfunc_;
		UniConnBase* c_;
};

class SingleMsgSrc0: public SingleMsgSrc
{
	public:
		SingleMsgSrc0( UniConnBase* c)
			: SingleMsgSrc( dummyFunc0, c )
		{
			;
		}

		void send() const {
			if ( c_->nTargets() > 0)
				rfunc_( c_->target(0) );
		}
};

template <class T> class SingleMsgSrc1: public SingleMsgSrc
{
	public:
		SingleMsgSrc1( UniConnBase* c )
			// : SingleMsgSrc( reinterpret_cast< RecvFunc >( dummyFunc<T> ), c)
			: SingleMsgSrc( dummyFunc0 , c)
		{
			;
		}

		void send(T v) const {
			if ( c_->nTargets() > 0)
				reinterpret_cast< void ( * )( Conn*, T ) >( rfunc_ )(
					c_->target(0), v);
		}
};

template < class T1, class T2 > class SingleMsgSrc2: public SingleMsgSrc
{
	public:
		SingleMsgSrc2( UniConnBase* c )
			: SingleMsgSrc( dummyFunc0 , c)
		{
			;
		}

		void send( T1 v1, T2 v2 ) const {
			if ( c_->nTargets() > 0)
				reinterpret_cast< void ( * )( Conn*, T1, T2 ) >(
					rfunc_ )( c_->target(0), v1, v2 );
		}
};

template < class T1, class T2, class T3 > class SingleMsgSrc3: 
	public SingleMsgSrc
{
	public:
		SingleMsgSrc3( UniConnBase* c )
			: SingleMsgSrc( dummyFunc0 , c)
		{
			;
		}

		void send( T1 v1, T2 v2, T3 v3 ) const {
			if ( c_->nTargets() > 0)
				reinterpret_cast< void ( * )( Conn*, T1, T2, T3 ) >(
					rfunc_ )( c_->target(0), v1, v2, v3 );
		}
};

// Base class for multiple outgoing messages, manages MultiConns
class NMsgSrc
{
	public:
		NMsgSrc( BaseMultiConn* c )
			: c_( c )
		{
			rfuncs_.resize( 0 );
		}
		bool add( RecvFunc rf, Conn* target );
		void resize( std::vector< unsigned long >& segments ) {
				c_->resize( segments );
		}

		// bool drop( Conn* target );
		bool funcDrop( const Conn* target );

		void dropAll() {
			rfuncs_.resize( 0 );
			c_->disconnectAll();
		}

		unsigned long nTargets() const {
			return c_->nTargets();
		}

		// Returns the index of the matching function in the src. If not
		// found returns the size of the rfuncs vector.
		unsigned long indexOfMatchingFunc( RecvFunc func ) const;

		// Returns the number of matches.
		unsigned long matchRemoteFunc( RecvFunc rf ) const;

		RecvFunc targetFunc( unsigned long i ) const;
		RecvFunc targetFuncFromSlot( unsigned long slot ) const;
		unsigned long nFuncs() const {
			return rfuncs_.size();
		}

		Conn* conn() const {
			return c_;
		}

		void dest( std::vector< Field >& list );

		void addRecvFunc( RecvFunc rf, unsigned long position );
	
	protected:
		std::vector< RecvFunc >rfuncs_;
		BaseMultiConn* c_;
};

class NMsgSrc0: public NMsgSrc
{
	public:
		NMsgSrc0( BaseMultiConn* c )
			: NMsgSrc( c )
		{
			;
		}

		void send() const
		{
			for (unsigned long i = 0; i < rfuncs_.size(); i++) {
				for_each( c_->begin( i ), c_->end( i ), rfuncs_[ i ] );
			}
		}

		void sendTo( unsigned long i ) const
		{
			Conn* target = c_->target( i );
			if ( target ) {
				rfuncs_[ c_->index( i ) ]( target );
			}
		}
};

template <class T> class NMsgSrc1: public NMsgSrc
{
	public:
		NMsgSrc1( BaseMultiConn* c )
			: NMsgSrc( c )
		{
			;
		}

		void send(T v) const
		{
			/*
			cerr << "rfuncs_.size() = " << rfuncs_.size() << "\n";
			cerr << "v =  " << v << "\n";
			*/

			for (unsigned long i = 0; i < rfuncs_.size(); i++) {
				Op1<T> op(
					v, 
					reinterpret_cast< void ( * )( Conn*, T ) >(
						rfuncs_[i] )
				);
				for_each( c_->begin( i ), c_->end( i ), op );
			}
		}

		// This may be faster on two counts:
		// - One less indirection per cycle to look up the conn
		// - Contiguous addresses for the Conns.
		/*
		void sendSolve( T v ) const
		{
			vector< SolverConn >::iterator j = vec_.begin();
			for (unsigned long i = 0; i < rfuncs_.size(); i++) {
				void( * )( Conn*, T ) rf =
					reinterpret_cast< void ( * )( Conn*, T ) >(
						rfuncs_[i] );
				vector< SolverConn >::iterator k = 
					vec_.begin() + segments_[i];
				for (; j < k; j++)
					rf( j, v );
			}
		}
		*/

		void sendTo( unsigned long i, T v ) const
		{
			Conn* target = c_->target( i );
			if ( target ) {
				reinterpret_cast< void ( * )( Conn*, T ) >
					( rfuncs_[ c_->index( i ) ] )( target, v);
			}
		}
};

template < class T1, class T2 > class NMsgSrc2: public NMsgSrc
{
	public:
		NMsgSrc2( BaseMultiConn* c )
			: NMsgSrc( c )
		{
			;
		}

		void send( T1 v1, T2 v2 ) const
		{
			for (unsigned long i = 0; i < rfuncs_.size(); i++) {
				Op2< T1, T2 > op(
					v1, v2, 
					reinterpret_cast< void ( * )( Conn*, T1, T2 ) >(
						rfuncs_[i] )
				);
				for_each( c_->begin( i ), c_->end( i ), op );
			}
		}

		/*
		void sendSolve( T1 v1, T2 v2 ) const
		{
			vector< SolverConn >::iterator j = vec_.begin();
			for (unsigned long i = 0; i < rfuncs_.size(); i++) {
				void( * )( Conn*, T1, T2 ) rf =
					reinterpret_cast< void ( * )( Conn*, T1, T2 ) >(
						rfuncs_[i] );
				vector< SolverConn >::iterator k = 
					vec_.begin() + segments_[i];
				for (; j < k; j++)
					rf( j, v1, v2 );
			}
		}
		*/

		void sendTo( unsigned long i, T1 v1, T2 v2 ) const
		{
			Conn* target = c_->target( i );
			if ( target ) {
				reinterpret_cast< void ( * )( Conn*, T1, T2 ) >
					( rfuncs_[ c_->index( i ) ] )( target, v1, v2);
			}
		}
};

template < class T1, class T2, class T3 > class NMsgSrc3:
	public NMsgSrc
{
	public:
		NMsgSrc3( BaseMultiConn* c )
			: NMsgSrc( c )
		{
			;
		}

		void send( T1 v1, T2 v2, T3 v3 ) const
		{
			for (unsigned long i = 0; i < rfuncs_.size(); i++) {
				Op3< T1, T2, T3 > op(
					v1, v2, v3,
					reinterpret_cast< void ( * )( Conn*, T1, T2, T3 ) >(
						rfuncs_[i] )
				);
				for_each( c_->begin( i ), c_->end( i ), op );
			}
		}

		/*
		void sendSolve( T1 v1, T2 v2, T3 v3 ) const
		{
			vector< SolverConn >::iterator j = vec_.begin();
			for (unsigned long i = 0; i < rfuncs_.size(); i++) {
				void( * )( Conn*, T1, T2, T3 ) rf =
					reinterpret_cast< void ( * )( Conn*, T1, T2, T3 ) >(
						rfuncs_[i] );
				vector< SolverConn >::iterator k = 
					vec_.begin() + segments_[i];
				for (; j < k; j++)
					rf( j, v1, v2, v3 );
			}
		}
		*/

		void sendTo( unsigned long i, T1 v1, T2 v2, T3 v3 ) const
		{
			Conn* target = c_->target( i );
			if ( target ) {
				reinterpret_cast< void ( * )( Conn*, T1, T2, T3 ) >
					( rfuncs_[ c_->index( i ) ] )( target, v1, v2, v3);
			}
		}
};

#endif	// _MSGSRC_H
