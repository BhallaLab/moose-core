/********************************************************************** ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _OBJ_FINFO_H
#define _OBJ_FINFO_H

// ObjFinfo: A finfo for a previously defined MOOSE class
// that is to be encapsulated in another. Example: Interpols
// The goal is to make all the value fields of the encapsulated class
// available for usual access, through the field matching function.
// Message fields are NOT available in this manner.
// PtrObjFinfo: Similar, except that the MOOSE class is
// saved as a ptr.
// MsgObjFinfo: Similar, except that the moose class is
// looked up through a message
// Readonly versions of above too.

/*
template< class T> void objFinfoSet( Conn* c, T value )
{
	RelayConn* rc = dynamic_cast< RelayConn* >( c );
	if ( rc ) {
		ObjFinfo* f = dynamic_cast< ObjFinfo* >( rc->finfo() );
		if ( !f ) {
			RelayFinfo1< T >* rf = 
				dynamic_cast< RelayFinfo1< T >* >( rc->finfo());
			if ( rf ) 
				f = dynamic_cast< ObjFinfo* >( rf->innerFinfo() );
		}
		if ( f ) {
			// At this stage index_ is not doing anything, but later
			// it might allow array type fields too.
			Element* pa = f->lookup( c->parent(), f->index_ );
			if ( !Ftype1< T >::set( pa, f->finfo(), value ) ) {
				cerr << "Error::ObjFinfo::set: Field '" <<
					c->parent()->path() << "." <<
					f->name() << "' not known\n";
			}
		}
	}
}

template< class T> T objFinfoGet( const Element* )
{
}
*/

/*
class ObjFinfo: public Finfo
{
	public:
		ObjFinfo(
			const string& name,
			Element* (*lookup)(const Element *, unsigned long),
			const string& className, // Name of encapsulated class.
			unsigned long index = 0
		)
		:	Finfo( name ),
			lookup_( lookup ),
			className_( className ),
			index_(index)
		{
			;
		}

		static void rfunc( Conn* c );

		RecvFunc recvFunc() const {
			return 0;
		}

		Field match(const string& s, Element* e );

		// Conditional copy
		Finfo* copy() {
			if (finfo_ != 0) {
				return ( new ObjFinfo( *this ) );
			}
			return this;
		}

		// Conditional destructor
		void destroy() const {
			if (finfo_ != 0) {
				delete this;
			}
		}

		Finfo* finfo() const {
			return finfo_;
		}

		// Creates a relay of the correct type and uses it for the add.
		bool add( Element* e, Field& destfield, bool useSharedConn = 0);
		Finfo* respondToAdd( Element* e, const Finfo* sender );

		const Ftype* ftype() const {
			if ( finfo_ )
				return finfo_->ftype();
			// Here we put in a new ftype based on a Cinfo comparison
			return 0;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			return 0;
			// should not allow relays?
		}

	private:
		Element* ( *lookup_ )(const Element *, unsigned long);
		string className_;
		unsigned long index_; // Not sure yet how to incorporate
};
*/

// May want to do some clever stuff here so we can assign entire
// elements directly.

extern Field matchObjFinfo( 
	const string& s, const string& name,
	const Cinfo* cinfo, 
	Element* (*lookup)( Element *, unsigned long)
);

template < class T > class ObjFinfo: public ValueFinfo< T >
{
	public:
		ObjFinfo(
			const string& name,
			T ( *get )( const Element *),
			void ( *set )( Conn*, T ),
			Element* (*lookup)( Element *, unsigned long),
			const string& className // Name of encapsulated class.
		)
		:	ValueFinfo< T >( name, get, set, className ), 
			lookup_( lookup )
		{
			;
		}

		Field match(const string& s ) {
			if ( s == this->name() )
				return this;
			return matchObjFinfo( s, this->name(), this->cinfo(), lookup_ );
		}

	private:
		Element* ( *lookup_ )( Element *, unsigned long );
		string className_;
};

/*
template <class T> class ReadOnlyArrayFinfo: public ArrayFinfo<T>
{
	public:
		ReadOnlyArrayFinfo(
			const string& name,
			T (*get)(const Element *, unsigned long),
			const string& className, // Name of equivalent class.
			unsigned long index = 0
		)
		:	ArrayFinfo<T>( name, get, 0, className, index )
		{
			;
		}

		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			Ftype0 f0;
			if ( sender->isSameType( &f0 ) ) { // check for trigger
				// Look for a relay sending out a value message
				return obtainValueRelay(
					&newValueRelayFinfo< T >, this, e, 1);
			}
			cerr << "ReadOnlyArrayFinfo::respondToAdd: failed with " << 
		//	e->name() << 
			"." << this->name() << ", sender = " <<
			sender->name() << "\n";
			return 0;
		}
};
*/

#endif	//  _OBJ_FINFO_H
