/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FINFO_H
#define _FINFO_H
////////////////////////////////////////////////////////////////////
// This file has the base Finfo classes: Finfo, Finfo0, Finfo1 and
// some generic finfo classes
////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// Finfo holds field information. Here we define the core finfo 
// classes:
// Finfo is the abstract base class
// Finfo0 is a class for fields with no associated type. Used in methods
// Finfo1 has a single type. Used in value fields. See Field.h. Also
//    used in single-value methods.
// Finfo2 has two types. Used in double-value methods.
// Finfo3 has three types. Used in triple-value methods.
// In principle we could go to higher Finfos, but I have not had to
// do so as yet. 
/////////////////////////////////////////////////////////////////////

class Finfo {
	public:
		Finfo(const string& name)
		:
			name_(name)
		{ ; }

		virtual ~Finfo()
		{ ; }

		// This does arbitrary complexity parsing to do
		// lookup of indirection and array subfields.
		virtual Field match( const string& s );

//////////////////////////////////////////////////////////////////
//  A bunch of functions to handle RecvFuncs.
//////////////////////////////////////////////////////////////////
		// Returns a RecvFunc to receive messages from the sender
		// No typechecking.
		virtual RecvFunc recvFunc( ) const = 0;

		// Returns the function used by MsgSrcs to call out onto the
		// specified msgno.
		// Used to traverse messages.
		virtual RecvFunc targetFunc( Element* e, unsigned long msgno ) const
		{
			return 0;
		}

		// Used by MsgSrcs. Returns length of function list if not found
		virtual unsigned long indexOfMatchingFunc( 
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
		virtual unsigned long matchRemoteFunc(
			Element* e, RecvFunc rf ) const = 0;

		// Adds a RecvFunc to a MsgSrc at the specified position.
		// Used primarily by SharedFinfo.
		virtual void addRecvFunc( 
			Element* e, RecvFunc rf, unsigned long position ) = 0;

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
		virtual Finfo* makeRelayFinfo( Element* e ) = 0;

		// Assigns indirection functions to ValueFinfos for use in
		// ObjFinfo. Returns true only if the operation is permitted,
		// otherwise the Finfo does not support ObjFinfos.
		virtual Finfo* makeValueFinfoWithLookup( 
			Element* (*lookup)( Element *, unsigned long ), 
			unsigned long )
		{
			return 0;
		}

//////////////////////////////////////////////////////////////////////
// These are routines for message traversal
//////////////////////////////////////////////////////////////////////

		// Returns the incoming Conn object for this Finfo. May be blank
		virtual Conn* inConn( Element* ) const = 0;
		// Returns the outgoing Conn object for this Finfo. May be blank
		virtual Conn* outConn( Element* ) const = 0;

		// Utility function for returning an empty Conn, for use above
		// if the Conn object is blank.
		static Conn* dummyConn()  {
			return dummyConn_;
		}

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

//////////////////////////////////////////////////////////////////////
// Here are the message creation and removal operations.
//////////////////////////////////////////////////////////////////////

		// Adds a message from this Finfo to the target field destfield.
		// The flag tells it if the function is called as part of a
		// SharedFinfo.
		virtual bool add( Element* e, Field& destfield,
			bool useSharedConn = 0 ) = 0;

		// Returns a Finfo to use to build the message.
		// Sometimes is self, but often is a Relay.
		// Does type checking to ensure message compatibility
		// Returns 0 if it fails.
		virtual Finfo* respondToAdd( Element* e, const Finfo* sender )
			= 0;

		// Removes a connection.
		// Usually called from the dest with the src as argument,
		// because synapses (destinations) have to override this
		// function.
		virtual bool drop( Element* e, Field& srcfield );

//////////////////////////////////////////////////////////////////////
// Here are the Finfo miscellaneous functions
//////////////////////////////////////////////////////////////////////


		// Used during init for value Finfos to associate with a class
		// Also used by SrcFinfo and DestFinfo to set up their
		// trigger list.
		virtual void initialize( const Cinfo* c ) = 0;

		// Returns the name of the field.
		const string& name() const {
				return name_;
		}

		// Does type checking by referring to internal Ftype.
		bool isSameType( const Finfo* other ) const;
		// Does type checking by referring to internal Ftype.
		bool isSameType( const Ftype* other ) const;

		// Does conversion of field value to a string
		virtual bool strGet( Element* e, string& val );
		// Does assignment of field value from a string.
		// The string can also be parsed into argument(s) for
		// a Finfo that handles functions (such as DestFinfo).
		virtual bool strSet( Element* e, const string& val );

		// Returns Ftype of this Finfo.
		virtual const Ftype* ftype() const = 0;

	private:
		const string name_;
		// Add msg with this Finfo as dest. Dest provides func?
		// virtual Finfo* Add2(Element* e2) = 0;  
		static Conn* dummyConn_;
};

class FinfoDummy: public Finfo {
	public:
		FinfoDummy(const string& name)
		: Finfo(name)
		{ ; }

		~FinfoDummy()
		{ ; }

		RecvFunc recvFunc( ) const {
			return &dummyFunc_;
		}

		unsigned long matchRemoteFunc(
			Element* e, RecvFunc rf ) const {
			return 0;
		}

		void addRecvFunc( Element* e, RecvFunc rf,
			unsigned long position )
		{
			;
		}

		Conn* inConn( Element* ) const {
			return dummyConn();
		}

		Conn* outConn( Element* ) const {
			return dummyConn();
		}


		// Here are the message creation and removal operations.
		bool add( Element* e, Field& destfield, bool useSharedConn ) {
			return 0;
		}
		
		// The destination of a message checks type of sender, and if
		// all is well returns a Finfo to use for the message.
		Finfo* respondToAdd( Element* e, const Finfo* sender ) { 
			return 0;
		}

		void initialize( const Cinfo* c ) {
			;
		}

		const Ftype* ftype() const;

		Finfo* makeRelayFinfo( Element* e ) {
			return this;
		}

	private:
		const string name_;
		// Add msg with this Finfo as dest. Dest provides func?
		// virtual Finfo* Add2(Element* e2) = 0;  
		static void dummyFunc_( Conn* c ) {
			;
		}
};

/*
template < class T > void setWrapper( Conn* c, T value ) {
	RelayConn* rc = dynamic_cast< RelayConn* >( c );
	if ( rc ) {
		ValueFinfo< T >* f = dynamic_cast< ValueFinfo< T >* >( 
			rc->finfo() );
		if ( f ) {
			f->innerSet( c, value );
		}
	}
}
*/

// Used as baseclass for value Finfos, that is, Finfos representing
// value fields in the object. Still an abstract class as the
// add and respondToAdd methods are not filled.
// Also the get function cannot be put in at this stage, because
// it may involve indirection information.
// This is defined here because of header dependencies.
template <class T> class ValueFinfoBase: public Finfo
{
	public:
		ValueFinfoBase(
			const string& name,
			void (*set)(Conn*, T),
			const string& className // Name of equivalent class.
		)
		:	Finfo(name),
			set_(set),
			className_(className)
		{
			;
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( set_ );
		}

		unsigned long matchRemoteFunc(
			Element* e, RecvFunc rf ) const {
			return 0;
		}

		void addRecvFunc( 
			Element* e, RecvFunc rf, unsigned long position ) {
			;
		}

		virtual T value(const Element* e) const = 0;

		void initCinfo() {
			cinfo_ = Cinfo::find(className_);
		}

		Conn* inConn( Element* ) const {
			return dummyConn();
		}

		Conn* outConn( Element* ) const {
			return dummyConn();
		}

		void initialize( const Cinfo* c ) {
			cinfo_ = Cinfo::find( className_ ); 
			if ( !cinfo_ ) {
				className_[0] = toupper( className_[0] );
				cinfo_ = Cinfo::find( className_ ); 
			}
			if ( !cinfo_ ) {
				cerr << "ValueFinfoBase::Error: Unable to initialize class " << className_ << "\n";
			}
		}

		const string& className() const {
			return className_;
		}

		const Cinfo* cinfo() const {
			return cinfo_;
		}

		private:
			T (*get_)(const Element *);
			void (*set_)(Conn*, T);
			string className_;
			const Cinfo* cinfo_;
			static const Ftype* myFtype_;
};

#endif // _FINFO_H
