/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _CONN_H
#define _CONN_H

#include "header.h"

/////////////////////////////////////////////////////////////////////
// 
// The Conn family of classes handle connections between objects.
// They are meant to be used symmetrically, that is, both the
// source and destination sides look the same and in some cases
// they may be interchangeable. The proliferation of subtypes is
// to optimize for 
// UniConn: single connections
// PlainMultiConn: multiple connections, typically destinations.
// MultiConn: Source-side multiple connections with multiple possible
//   functions. Each function reaches a separate set of targets.
// RelayConn: Derived from PlainMultiConn. Other than a few hard-coded
//   messages which have precompiled conns for speed, all others use
//   RelayConns which are created on the fly.
// SynConn: Synaptic connections: a single target and a templated value
//
/////////////////////////////////////////////////////////////////////

class Conn
{
	public:
		virtual ~Conn()
		{
			;
		}
		virtual Conn* target( unsigned long index ) const = 0;
		virtual unsigned long nTargets() const = 0;
		virtual void listTargets( vector< Conn* >& ) const = 0;

		virtual Element* parent() const = 0;	// Ptr to parent Element
		bool connect( Conn* target, 
			unsigned long sourceSlot, unsigned long targetSlot = 0);

		// Returns index of disconnected conn, MAX = ~0 if it fails.
		unsigned long disconnect( Conn* target );

		virtual void disconnectAll() = 0;

		// Return index of target if found. Otherwise return MAX = ~0
		virtual unsigned long find( const Conn* target) const = 0;
		virtual bool innerConnect(Conn* target, unsigned long slot = 0)
			= 0;
		virtual void innerDisconnect(unsigned long index) = 0;
		
		// Brute force cleans up connections without communicating
		// with target.
		virtual void innerDisconnectAll() = 0;

		static const unsigned long MAX;
	protected:
		virtual bool canConnect(Conn* target) const = 0;
		virtual Conn* respondToConnect(Conn* target) = 0;

	private:
};

// The UniConnBase manages the target formation and removal.
// Note that this is also an abstract base class as it doesn't have a
// way to access parent()
class UniConnBase: public Conn
{
	public:
		UniConnBase() {
			target_ = 0;
		}

		~UniConnBase()
		{
			if ( target_ )
				disconnect( target_ );
		}

		Conn* target(unsigned long index) const {
			if (index == 0)
				return target_;
			return 0;
		}

		// Unchecked version of above.
		Conn* rawTarget() const {
			return target_;
		}

		unsigned long nTargets() const {
			return (target_ != 0);
		}

		void listTargets( vector< Conn* >& list ) const {
			if ( target_ ) {
				list.push_back( target_ );
			}
		}

		unsigned long find( const Conn* target ) const {
			if (target_ && (target_ == target))
				return 0;
			return ~1;
		}

		bool innerConnect( Conn* target, unsigned long slot = 0 ) {
			target_ = target;
			return 1;
		}

		void disconnectAll() {
			disconnect( target_ );
		}

		void innerDisconnect(unsigned long i) {
			target_ = 0;
		}

		void innerDisconnectAll() {
			target_ = 0;
		}

	protected:
		bool canConnect(Conn* target) const {
			return (target && (target_ == 0));
		}
		Conn* respondToConnect(Conn* target) {
			if ( canConnect( target ) )
				return this;
			return 0;
		}

	private:
		Conn* target_;
};


// The UniConn has a single target. It uses a template to
// look up the parent ptr, so that it uses minimal space.
// The function F returns the Element ptr when passed the ptr to the
// Conn.
template< Element* (*F)(const Conn *)>class UniConn: public UniConnBase
{
	public:
		Element* parent() const {
			return F(this);
		}
};

class UniConn2: public UniConnBase
{
	public:
		UniConn2( Element* parent )
			: parent_( parent )
		{
			;
		}

		Element* parent() const {
			return parent_;
		}

	private:
		Element* parent_;
};


// The SynConn has a single target, and stores per-synapse info T.
// We may need to create a special allocator for this as there may
// be huge numbers of them, especially for T = float, float and double.
template< class T >class SynConn: public UniConnBase
{
	public:
		SynConn( Element* parent )
			: parent_( parent )
		{
			;
		}

		T value_;

		Element* parent() const {
			return parent_;
		}

	private:
		Element* parent_;
};

// The PlainMultiConn has any number of targets. It stores its parent
// ptr. It is used for msgdests and does not worry about handling
// recvfuncs to correspond to each conn. So it is just a simple
// vector of Conns.
class PlainMultiConn: public Conn
{
	public:
		PlainMultiConn(Element* e)
			: parent_(e)
		{
			;
		}

		~PlainMultiConn();

		Conn* target(unsigned long index) const {
			if (index < target_.size())
				return target_[index];
			return 0;
		}

		unsigned long nTargets() const {
			return static_cast< unsigned long >( target_.size() );
		}

		void listTargets( vector< Conn* >& list ) const {
			list.insert( list.end(), target_.begin(), target_.end() );
		}

		Element* parent() const {
			return parent_;
		}

		vector< Conn* >::const_iterator begin() const {
			return target_.begin();
		}

		vector< Conn* >::const_iterator end() const {
			return target_.end();
		}

		unsigned long find( const Conn* other ) const;

		bool innerConnect( Conn* target, unsigned long slot = 0 ) {
			target_.push_back(target);
			return 1;
		}

		void innerDisconnect(unsigned long index) {
			if ( index < target_.size() )
				target_.erase( target_.begin() + index );
		}

		void disconnectAll();

		void innerDisconnectAll() {
			target_.resize( 0 );
		}

	protected:
		bool canConnect(Conn* target) const {
			return ( target != 0 );
		}
		Conn* respondToConnect(Conn* target) {
			if ( canConnect( target ) )
				return this;
			return 0;
		}

	private:
		vector< Conn* > target_;
		Element* parent_;
};

// The MultiConn has any number of targets. It stores its parent ptr.
// Because the Conn may be accessed by several recvfuncs, it stores
// the targets in a list of distinct vectors.
class MultiConn: public Conn
{
	public:
		MultiConn( Element* e )
			: parent_(e)
		{
			;
		}

		~MultiConn();

		Conn* target(unsigned long index) const;
		unsigned long nTargets() const;
		void listTargets( vector< Conn* >& list ) const ;

		Element* parent() const {
			return parent_;
		}

		vector< Conn* >::const_iterator begin( unsigned long i ) const {
			return connVecs_[ i ]->begin();
		}

		vector< Conn* >::const_iterator end( unsigned long i ) const {
			return connVecs_[ i ]->end();
		}

		// void moveLastEntry( unsigned long index );

		unsigned long find( const Conn* target ) const;

		// Returns the index of the ConnVec corresponding to specified
		// message number.
		unsigned long index( unsigned long msgno );

		bool innerConnect( Conn* target, unsigned long slot = 0 );

		void innerDisconnect(unsigned long index);

		void disconnectAll();
		void innerDisconnectAll();

	protected:
		// Always true as long as input is good.
		bool canConnect( Conn* target ) const {
			return ( target != 0 );
		}
		Conn* respondToConnect(Conn* target) {
			if ( canConnect( target ) )
				return this;
			return 0;
		}

	private:
		vector< PlainMultiConn* > connVecs_;
		Element* parent_;
};

// Used to send messages along connections that are not precompiled.
class RelayConn: public PlainMultiConn
{
	public:
		RelayConn( Element* e, Finfo* f )
			: PlainMultiConn( e ), f_( f )
		{
			;
		}

		// Inherits the delete func from its base.

		Finfo* finfo() const {
			return f_;
		}

		// Informs the parent finfo if it is empty.
		void innerDisconnect(unsigned long index);

	protected:

	private:
		Finfo* f_;
};

class ReturnConn: public UniConn2
{
	public:
		ReturnConn( Element* parent )
			: UniConn2( parent ), rfunc_( 0 )
		{
			;
		}

		bool addRecvFunc( RecvFunc rfunc ) {
			if ( rfunc_ != 0 )
				return 0;
			rfunc_ = rfunc;
			return 1;
		}

		RecvFunc recvFunc() const {
			return rfunc_;
		}

	private:
		RecvFunc rfunc_;
};
// The MultiReturnConn is a vector of simple conns of the 
// ReturnConn type. Each of these stores the target, the parent, and
// a recvFunc.
// Needed when multiple incoming messages each expect a return value
// for the specific function, for example when doing interpolation
// lookups.  A trifle wasteful in memory, but here speed is critical.
class MultiReturnConn: public Conn
{
	public:
		MultiReturnConn(Element* e)
			: parent_(e)
		{
			;
		}

		~MultiReturnConn();

		Conn* target(unsigned long index) const {
			if (index < vec_.size())
				return vec_[index]->rawTarget();
			return 0;
		}

		unsigned long nTargets() const {
			return static_cast< unsigned long >( vec_.size() );
		}

		RecvFunc targetFunc( unsigned long index ) const {
			if ( index < vec_.size() )
				return vec_[ index ]->recvFunc() ;
			return &dummyFunc0;
		}

		unsigned long indexOfMatchingFunc( RecvFunc rf ) const;
		unsigned long matchRemoteFunc( RecvFunc rf ) const;
		void addRecvFunc( RecvFunc rf );

		void listTargets( vector< Conn* >& list ) const;

		Element* parent() const {
			return parent_;
		}

		// Haven't figured out these. I think I will have to
		// define an iterator class here that does the right thing.
		vector< Conn* >::const_iterator begin() const {
			return static_cast< vector< Conn* >::const_iterator >( 0 );
		}

		vector< Conn* >::const_iterator end() const {
			return static_cast< vector< Conn* >::const_iterator >( 0 );
		}

		unsigned long find( const Conn* other ) const;
		bool innerConnect( Conn* target, unsigned long slot = 0 );
		void innerDisconnect(unsigned long index);
		void disconnectAll();
		void innerDisconnectAll();

	protected:
		bool canConnect(Conn* target) const {
		// Always true as long as input is good.
			return ( target != 0 );
		}
		Conn* respondToConnect( Conn* target ) {
			if ( canConnect( target ) ) {
				ReturnConn* ret = new ReturnConn( parent_ );
				vec_.push_back( ret );
				return ret;
			}
			return 0;
		}

	private:
		vector< ReturnConn* > vec_;
		Element* parent_;
};

#endif	// _MSGCONN_H
