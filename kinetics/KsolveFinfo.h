/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _KSOLVE_FINFO_H
#define _KSOLVE_FINFO_H
////////////////////////////////////////////////////////////////////
// This file has the Finfo classes for the ksolve.
// Possibly later we can template it when we know how such things
// work.
////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// KsolveMolFinfo has to do the following for the molecule on which
// it sits
// - Intercept all gets to n, and conc.
// - Intercept all sets to n and nInit, conc and concInit
// - Take over function of existing nouts if outside reac classes
// - Intercept relay messages e.g. for plots.
// - Intercept add and drop calls for manipulating messages.
//
/////////////////////////////////////////////////////////////////////

class KsolveMolFinfo: public Finfo
{
	public:
		KsolveMolFinfo(const string& name, Conn* conn)
		: Finfo(name), conn_( conn )
		{ 
			;
		}

		~KsolveMolFinfo()
		{ ; }

		// This is a key point of the interception
		// operations of the finfo
		Field match( const string& s ) {
			if ( s == "n" || s == "conc" ) {
			 	// Bypass the regulare Element::field() call because
				// it just calls the match function again.
				Field f = conn_->parent()->cinfo()->field( s );
				if ( f.good() ) {
					f.setElement( conn->parent() );
				}
			}
				return Field( this );
			return Field();
		}

		// Function for handling sets to n:
		// Can we avoid having a map for looking up conns?
		// Can this be a generic set?
		static void setNInit( Conn* c, double value ) {
			c->target()->parent()->setNInit( myIndex_, value );
			return c->parent()->setNInit( c, value );
		}

//////////////////////////////////////////////////////////////////
//  A bunch of functions to handle RecvFuncs.
//////////////////////////////////////////////////////////////////
		// Returns a RecvFunc to receive messages from the sender
		// No typechecking.
		RecvFunc recvFunc( ) const {
		}

		// Returns the function used by MsgSrcs to call out onto the
		// specified msgno.
		// Used to traverse messages.
		RecvFunc targetFunc( Element* e, unsigned long msgno ) const
		{
			return 0;
		}

		// Used by MsgSrcs. Returns length of function list if not found
		unsigned long indexOfMatchingFunc( 
			Element* e, RecvFunc rf ) const
		{
			return 0;
		}

		// Could in principle implement this using the above call.
		// Likely to be much more efficient this way.
		// Checks if a msgsrc uses the specified func to call out.
		// Used by msgdests to find msgsrcs.
		// May be multiple matches, so returns an unsigned long.
		// Many fields do not use this, so it is defaulted to 0.
		unsigned long matchRemoteFunc(
			Element* e, RecvFunc rf ) const {
			return 0;
		}

		// Adds a RecvFunc to a MsgSrc at the specified position.
		// Used primarily by SharedFinfo.
		void addRecvFunc( 
			Element* e, RecvFunc rf, unsigned long position ) 
		{
		}

//////////////////////////////////////////////////////////////////
//  A bunch of functions to handle RecvFuncs.
//////////////////////////////////////////////////////////////////
		// Default implementations for copy. If a derived class of
		// Finfo must return a new instance in the match() command,
		// then it should also provide a copy() function to return
		// a newly allocated copy of itself.
		virtual Finfo* copy() {
			return this;
		}

		// Default implementations for destroy. If a derived class of
		// Finfo must return a new instance in the match() command,
		// then it should provide a destroy() function to clean out
		// the instance. Other Finfos just ignore the call.
		virtual void destroy() const {
		}

		// Handles creation of templated RelayFinfos.
		Finfo* makeRelayFinfo( Element* e ) {
		}

		// Assigns indirection functions to ValueFinfos for use in
		// ObjFinfo. Returns true only if the operation is permitted,
		// otherwise the Finfo does not support ObjFinfos.
		Finfo* makeValueFinfoWithLookup( 
			Element* (*lookup)( Element *, unsigned long ), 
			unsigned long )
		{
			return 0;
		}

//////////////////////////////////////////////////////////////////////
// These are routines for message traversal
//////////////////////////////////////////////////////////////////////

		// Returns the incoming Conn object for this Finfo. May be blank
		Conn* inConn( Element* ) const {
				return conn_;
		}
		// Returns the outgoing Conn object for this Finfo. May be blank
		virtual Conn* outConn( Element* ) const {
				return conn_;
		}

		/*
		// Returns a list of Fields that are targets of messages
		// emanating from this Finfo.
		virtual void src( vector< Field >&, Element* e );
		// Returns a list of Fields that are sources of messages
		// received by this Finfo.
		virtual void dest( vector< Field >&, Element* e );

		// Relay Finfos use this callback to clean up if they have
		// no more connections left.
		virtual void handleEmptyConn( Conn* c ) {
			cerr << "In Finfo::handleEmptyConn at " << 
				name() << ": should never happen.\n";
		}
		*/

//////////////////////////////////////////////////////////////////////
// Here are the message creation and removal operations.
//////////////////////////////////////////////////////////////////////

		// Adds a message from this Finfo to the target field destfield.
		// The flag tells it if the function is called as part of a
		// SharedFinfo.
		bool add( Element* e, Field& destfield,
			bool useSharedConn = 0 ) {
			return 0;
		}

		// Returns a Finfo to use to build the message.
		// Sometimes is self, but often is a Relay.
		// Does type checking to ensure message compatibility
		// Returns 0 if it fails.
		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			return 0;
		}

		// Removes a connection.
		// Usually called from the dest with the src as argument,
		// because synapses (destinations) have to override this
		// function.
		bool drop( Element* e, Field& srcfield ) {
		}
		bool dropAll( Element* e ) {
		}

//////////////////////////////////////////////////////////////////////
// Here are the Finfo miscellaneous functions
//////////////////////////////////////////////////////////////////////


		// Used during init for value Finfos to associate with a class
		// Also used by SrcFinfo and DestFinfo to set up their
		// trigger list.
		void initialize( const Cinfo* c ) {
		}

		// Returns Ftype of this Finfo.
		const Ftype* ftype() const {
				return 0;
		}

	private:
		Conn* conn_;
};

#endif // _KSOLVE_FINFO_H
