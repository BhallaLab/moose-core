/********************************************************************** ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

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
void ObjFinfo::set( Conn* c, T value )
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
			Element* pa = lookup( c->parent(), f->index_ );
			if ( !Ftype1< T >::set( pa, f->finfo(), value ) ) {
				cerr << "Error::ObjFinfo::set: Field '" <<
					c->parent()->path() << "." <<
					f->name() << "' not known\n";
			}
		}
	}
}

void ObjFinfo::set( Conn* c, T value )
{
	Element* pa = lookup( c->parent(), f->index_ );
	SillyConn sc( pa );
}

T value( const Element* e ) const {
	return get_( lookup( e ) );
}
*/

Field matchObjFinfo(const string& objName, const string& name, 
	const Cinfo* cinfo,
	Element* (*lookup)(Element *, unsigned long)
	)
{
	// The separators should really be language specific. I don't
	// have a good way to do this as yet, so I replace -> with . below.
	static const string OLDSEPARATOR = "->";
	static const string DOTSEPARATOR = ".";

	// must accommodate separator + subfield name
	if ( objName.substr( 0, name.length() ) == name ) {
		string s = objName;
		size_t pos = s.find( OLDSEPARATOR );
		if ( pos != string::npos )
			s.replace( pos, OLDSEPARATOR.length(), DOTSEPARATOR );
		
		if ( s.length() <= name.length() + DOTSEPARATOR.length() ) {
		// too short to work
			return Field();
		}
		long index = 0;
		pos = name.length();
		if ( s[ pos ] == '[' ) {
			index = atol( s.substr( name.length() + 1 ).c_str() );
			if ( index < 0 ) {
				cerr << "Error:ObjFinfo::match: negative index for " <<
					s << "\n";
					return Field();
			}
			size_t max = s.length() - ( DOTSEPARATOR.length() + 2 );
			// max should be long enough to hold ']DOTSEPARATORfield'
			while ( s[ pos ] != ']' ) {
				if ( ++pos >= max ) {
					cerr << "Error:ObjFinfo::match: no closing ] in " <<
						s << "\n";
						return Field();
				}
			}
			pos++;
		}
		// if ( s[ pos ] == SEPARATOR )
		if ( s.substr(pos).find( DOTSEPARATOR ) == 0 ) {
			string fieldName = s.substr( name.length() + 1 );
			Field field = cinfo->field( fieldName );
			if ( field.good() ) {
				Finfo* ret = field->makeValueFinfoWithLookup(
					lookup, index );
				if ( ret )
					return ret;
			}
		}
	}
	return Field();
}
/*
// Creates a relay of the correct type and uses it for the add.
bool ObjFinfo::add( Element* e, Field& destfield, bool useSharedConn)
{
	if ( finfo_ ) { // Only relays are permitted. Have to be careful.
		return finfo_->add( e, destfield, useSharedConn );
	}
	return 0;
}

Finfo* ObjFinfo::respondToAdd( Element* e, const Finfo* sender )
{
	if ( finfo_ ) { // Only relays are permitted. Have to be careful.
		return finfo_->respondToAdd( e, sender );
	}
	return 0;
}
*/
