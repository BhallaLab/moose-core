/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FIELD_H
#define _FIELD_H

#include <string>
#include <vector>

// Goes to the parent of the specified Conn, and looks for a Finfo
// that either uses (Dest) or calls (Src) the specified RecvFunc.
// Used to traverse messages.
// extern Field lookupSrcField( Conn* c, RecvFunc rf );
// extern Field lookupDestField( Conn* c, RecvFunc rf );


// This is a container for field information, and is the interface 
// class that the rest of the system sees. It ensures that dynamically
// allocated Finfos are cleaned out when needed.
// Field cannot be allowed to have an invalid finfo pointer.
class Field {
	public:
		Field();

		Field( Finfo *f, Element* e = 0 );

		Field(const Field& other);

		Field(const std::string& path);

		Field( Element* e, const std::string& finfoName );

		~Field();

		// Returns true if Field contains a good Finfo.
		bool good() const {
			return ( f_ && ( f_ != dummy_ ) );
		}

		// Field assignment
		const Field& operator=( const Field& other );

		// Getting at the operations of the contained Finfo.
		Finfo* operator->() {
			return f_;
		}

		// Returns the contained Finfo
		Finfo* getFinfo() const {
			return f_;
		}

		// Returns the contained Element
		Element* getElement() const {
			return e_;
		}

		// Should elminate this. Too dangerous.
		void setElement( Element* e ) {
			e_ = e;
		}

		// Returns the element.field name with a period separator
		std::string name() const;
		// Returns the element/field path with a slash separator
		std::string path() const;

		// Update f_ and return True if the finfo has changed.
		bool refreshFinfo();

		// Wrapper for Finfo::respondToAdd
		// Returns a Finfo to use to build the message.
		// Sometimes is self, but often is a Relay.
		// Does type checking to ensure message compatibility
		// Returns 0 if it fails.
		Finfo* respondToAdd( Finfo* f );

		// Wrapper for Finfo::src
		// Returns a list of Fields that are targets of messages
		// emanating from this Finfo.
		void src( std::vector< Field >& list );

		// Wrapper for Finfo::dest
		// Returns a list of Fields that are sources of messages
		// received by this Finfo.
		void dest( std::vector< Field >& list );

		// Adds a message from this Field to the target field 'other'.
		bool add( Field& other );
		// Drops a message from this Field to the target field 'other'.
		bool drop( Field& other );

		// Drops all incoming messages into this dest field.
		bool dropAll();

		// Assigns a value to this field, doing type conversions if
		// needed. If the Finfo is a DestFinfo, it will try to 
		// call its recvFunc with the converted value as arguments.
		bool set( const std::string& value );
		// Extracts a value to this field, doing type conversions if
		// needed
		bool get( std::string& value );

		// applies operation op to compare field value with value.
		bool valueComparison( const std::string& op, const std::string& value );

		// Adds the relay f to the element.
		void appendRelay( Finfo* f );

	private:
		Finfo* f_;
		Element* e_;
		static Finfo* dummy_;
};

#endif // _FIELD_H
