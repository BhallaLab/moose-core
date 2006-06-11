/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _VALUE_FINFO_H
#define _VALUE_FINFO_H

///////////////////////////////////////////////////////////////////////
// Here we define a set of Finfo1 classes derived from ValueFinfoBase,
// that handle field access.
// ValueFinfo: Access fields through set and get functions.
// Can read/write through these access functions
// ReadOnlyValueFinfo: Constant value fields, can read.
// ArrayFinfo: Handles array read and writes.
///////////////////////////////////////////////////////////////////////

extern long findIndex(const string& s, const string& name);

extern bool valueFinfoAdd( 
	Finfo* vf, Finfo *( *createValueRelayFinfo )( Field& ),
	Element* e, Field& destfield, bool useSharedConn );

extern Finfo* valueFinfoRespondToAdd(
	Finfo* vf, Finfo *( *createValueRelayFinfo )( Field& ),
	Finfo *( *createRelayFinfo )( Field& ),
	Element* e, const Finfo* sender );


extern Finfo* obtainValueRelay(
	Finfo *( *createValueRelayFinfo )( Field& ),
	Finfo* f, Element* e, bool useSharedConn );

//////////////////////////////////////////////////////////////////////
// Here are routines for creating various templated Finfos
//////////////////////////////////////////////////////////////////////
template < class T > Finfo* newValueRelayFinfo( Field& temp ) {
	return new ValueRelayFinfo< T >( temp );
}

template< class T > Finfo* newRelayFinfo( Field& temp ) {
	return new RelayFinfo1< T >( temp );
}

//////////////////////////////////////////////////////////////////////
// ValueFinfo is the generic Finfo for handling regular fields
// in objects. It provides support for set and get operations, as
// well as for creation of Relay Finfos to handle any messaging
// to and from the field.
// The ValueFinfo uses get and set functions to talk to the actual
// field value. In many cases these functions do internal calculations
// and may even have element side effects. For example, the 
// size field of an array class might resize the array if its value
// is changed.
//////////////////////////////////////////////////////////////////////
template <class T> class ValueFinfo: public ValueFinfoBase<T>
{
	public:
		ValueFinfo(
			const string& name,
			T (*get)(const Element *),
			void (*set)(Conn*, T),
			const string& className // Name of equivalent class.
		)
		:	ValueFinfoBase<T>(name, set, className ),
			get_( get ),
			localSet_( set ),
			lookup_( 0 ),
			objIndex_( 0 )
		{
			;
		}

		T value(const Element* e) const {
			if ( lookup_ == 0 )
				return get_( e );
			else {
				// Should be safe as the get_ function does not
				// change e.
				Element* ce = const_cast< Element* >( e );
				return get_( lookup_( ce, objIndex_ ) );
			}
		}

		void innerSet( Conn* c, T value ) {
			if ( lookup_ && localSet_ ) {
				Element* e = lookup_( c->parent(), objIndex_ );
				RelayConn rc( e, this );
				localSet_( &rc, value );
			}
		}

		static void setWrapper( Conn* c, T value ) {
			RelayConn* rc = dynamic_cast< RelayConn* >( c );
			if ( rc ) {
				ValueFinfo< T >* vf = dynamic_cast< ValueFinfo< T >* >(
					rc->finfo() );
				if ( ! vf ) {
					RelayFinfo1< T >* rf =
						dynamic_cast< RelayFinfo1< T >* >( rc->finfo());
					if ( rf )
						vf = dynamic_cast< ValueFinfo< T >* >(
							rf->innerFinfo() );
				}
				if ( vf ) {
					vf->innerSet( c, value );
				}
			}
		}

		bool add( Element* e, Field& destfield, bool useSharedConn = 0){
			return valueFinfoAdd( 
				this, &newValueRelayFinfo< T >,
				e, destfield, useSharedConn ) ;
		}

		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			return valueFinfoRespondToAdd(
				this, &newValueRelayFinfo< T >, &newRelayFinfo< T >,
				e, sender );
		}
	
		const Ftype* ftype() const {
			static const Ftype1< T > myFtype_;
			return &myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo1< T >( temp );
		}

		Finfo* makeValueFinfoWithLookup(
			Element* (*lookup)( Element *, unsigned long ),
			unsigned long objIndex ) {
			ValueFinfo< T >* ret = new ValueFinfo< T >( 
				this->name(), get_, setWrapper, this->className() );
			ret->localSet_ = localSet_;
			ret->lookup_ = lookup;
			ret->objIndex_ = objIndex;
			return ret;
		}

		Finfo* copy() {
			if ( lookup_ )
				return new ValueFinfo< T >( *this );
			else
				return this;
		}

		void destroy() {
			if ( lookup_ )
				delete this;
		}

	private:
		T (*get_)(const Element *);
		void ( *localSet_ )( Conn*, T );
		Element* (*lookup_)( Element *, unsigned long);
		unsigned long objIndex_;
};

//////////////////////////////////////////////////////////////////////
// ReadOnlyValueFinfo is a dumbed-down version of ValueFinfo,
// and as the name implies permits only read operations to the value.
//////////////////////////////////////////////////////////////////////
template <class T> class ReadOnlyValueFinfo: public ValueFinfo<T>
{
	public:
		ReadOnlyValueFinfo(
			const string& name,
			T (*get)(const Element *),
			const string& className // Name of equivalent class.
		)
		:	ValueFinfo< T >( name, get, 0, className )
		{
			;
		}

		// The only case we permit is to attach a trigger
		// requesting a value. 
		// All other cases, especially the request to assign a field,
		// are blocked.
		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			if ( sender->isSameType( this ) ) { // illegal
				cerr << "ReadOnlyValueFinfo::respondToAdd Warning: attempt to make assignment message.\nDenied\n";
				return 0;
			}
			return ValueFinfo< T >::respondToAdd( e, sender );
			/*
			static Ftype0 f0;
			// Field temp( this, e );
			if ( sender->isSameType( &f0 ) ) {
				return obtainValueRelay(
					&newValueRelayFinfo< T >, this, e, 1 );
			}
			return 0;
			*/
		}
};


//////////////////////////////////////////////////////////////////////
// ArrayFinfo provides the ability to look up array entries in a
// class. The core of this class is the 'match' function, which 
// parses the field name string to do the lookup. ArrayFinfo is 
// meant to be dynamically generated with a specific internal index_
// value that looks up the specific entry in the array. When the
// match function finds a valid index in the lookup string, then
// it returns a newly allocated ArrayFinfo with the index_ set
// appropriately. It is the job of the encapsulating Field object
// to ensure that the allocated ArrayFinfo instance is deleted after
// use.
// Set and get functions are correspondingly complicated, because the
// specially allocated instance of the ArrayFinfo must be referred to
// in order to find which index is to be used.
//////////////////////////////////////////////////////////////////////
template <class T> class ArrayFinfo: public ValueFinfoBase<T>
{
	public:
		ArrayFinfo(
			const string& name,
			T (*get)(const Element *, unsigned long),
			void (*innerSet)(Element*, unsigned long, T),
			const string& className, // Name of equivalent class.
			unsigned long index = 0
		)
		:	ValueFinfoBase<T>( name, this->set, className ),
			get_( get ),
			innerSet_( innerSet ),
			index_(index),
			lookup_( 0 ),
			objIndex_( 0 )
		{
			;
		}

		// lookup of array subfields.
		// Issue: We cannot do size check here.
		Field  match(const string& s )
		{
			int i = findIndex(s, this->name());
			if (i == 0) {
				return this;
			} else if (i > 0) {
				ArrayFinfo* ret = new ArrayFinfo(*this);
				ret->index_ = i;
				return ret;
			} else {
				return Field();
			}
		}

		// Conditional copy
		Finfo* copy() {
			if (index_ != 0) {
				return ( new ArrayFinfo( *this ) );
			}
			return this;
		}

		// Conditional destructor
		void destroy() const {
			if (index_ != 0) {
				delete this;
			}
		}

		T value(const Element* e) const {
			if ( lookup_ == 0 )
				return get_(e, index_);
			else {
				Element* ce = const_cast< Element* >( e );
				return get_( lookup_( ce, objIndex_ ), index_ );
			}
		}

		// We do some of the nasty casts and index extraction here
		// to keep things nice for the user defined function for sets.
		static void set( Conn* c, T value ) {
			RelayConn* rc = dynamic_cast< RelayConn* >( c );
			if ( rc ) {
				// This first option should work for set commands.
				ArrayFinfo< T >* af = 
					dynamic_cast< ArrayFinfo< T >* >( rc->finfo() );
				if ( !af ) {
				// This second option should work for relay messages.
					RelayFinfo1< T >* rf = 
						dynamic_cast< RelayFinfo1< T >* >( rc->finfo());
					if ( rf )
						af = dynamic_cast< ArrayFinfo< T >* >(
							rf->innerFinfo() );
				}
				if ( af ) {
					Element* pa = c->parent();
					if ( af->lookup_ != 0 )
						pa = af->lookup_( pa, af->objIndex_ );
					af->innerSet_( pa, af->index(), value );
				}
			}
		}

		unsigned long index() const {
			return index_;
		}

		// Creates a relay of the correct type and uses it for the add.
		bool add( Element* e, Field& destfield, bool useSharedConn = 0){
			Finfo* f = destfield.respondToAdd( this );
			if ( f ) {
				// Look for a relay on the element, and associated with
				// this field, already receiving a trigger message
				// Possible issue here is indexing. The comparison
				// compares finfo pointers, but array Finfos are
				// spawned off according to index and two distinct
				// ones may be allocated for the same index.
				Finfo* rf = obtainValueRelay(
					&newValueRelayFinfo< T >, this, e, 0);
				if ( rf )
					return rf->add( e, destfield );
			}
			return 0;
		}

		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			Ftype0 f0;
			Finfo* ret = 0;
			Field temp( this, e );
			if ( this->isSameType( sender ) ) {
				ret = new RelayFinfo1< T >( temp );
				temp.appendRelay( ret );
				return ret;
			} else {
				if ( sender->isSameType( &f0 ) ) { // check for trigger
					// Look for a relay sending out a value message
					return obtainValueRelay(
						&newValueRelayFinfo< T >, this, e, 1);
				}
			}

			cerr << "ArrayFinfo::respondToAdd: failed with " << 
		//	e->name() << 
			"." << this->name() << ", sender = " <<
			sender->name() << "\n";
			return 0;
		}

		const Ftype* ftype() const {
			static const Ftype1< T > myFtype_;
			return &myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo1< T >( temp );
		}

		Finfo* makeValueFinfoWithLookup(
			Element* (*lookup)( Element *, unsigned long ),
			unsigned long objIndex ) {
			ArrayFinfo< T >* ret = new ArrayFinfo< T >( 
				this->name(), get_, innerSet_, this->className(), index_ );
			ret->lookup_ = lookup;
			ret->objIndex_ = objIndex;
			return ret;
		}

	private:
		T (*get_)(const Element *, unsigned long);
		void ( *innerSet_ )( Element*, unsigned long, T );
		unsigned long index_;
		Element* (*lookup_)( Element *, unsigned long);
		unsigned long objIndex_;
};

//////////////////////////////////////////////////////////////////////
// ReadOnlyArrayFinfo is a dumbed-down version of ArrayFinfo,
// and as the name implies permits only read operations to the value.
//////////////////////////////////////////////////////////////////////
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

#endif // _VALUE_FINFO_H
