/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <fstream>
#include "header.h"
#include "CinfoWrapper.h"

//////////////////////////////////////////////////////////////////
// Cinfo is the class info for the MOOSE classes.
//////////////////////////////////////////////////////////////////

Cinfo::Cinfo(const string& name,
				const string& author,
				const string& description,
				const string& baseName,
				Finfo** fieldArray,
				const unsigned long nFields,
				Element* (*createWrapper)(const string&, Element*,
					const Element*)
)
	: name_(name), author_(author), 
	description_(description), baseName_(baseName),
	fieldArray_(fieldArray), nFields_(nFields),
	createWrapper_(createWrapper)
{
	lookup()[name] = this;
}

const Cinfo* Cinfo::find( const string& name )
{
	map<string, Cinfo*>::iterator i;
	i = lookup().find(name);
	if ( i != lookup().end() )
		return i->second;
	return 0;
}

Field Cinfo::field( const string& name ) const
{
	for ( unsigned int i = 0; i < nFields_; i++ ) {
		Field ret = fieldArray_[ i ]->match( name );
		if ( ret.good() )
			return ret;
	}

	// Fallthrough. No matches were found, so ask the base class.
	if (base_ != this)
		return base_->field(name);

	// Give up. Return a dummy field.
	return Field();
}

void Cinfo::listFields( vector< Finfo* >& ret ) const
{
	for ( unsigned int i = 0; i < nFields_; i++ )
		ret.push_back( fieldArray_[ i ] );

	if (base_ != this)
		const_cast< Cinfo* >( base_ )->listFields( ret );
}

const Finfo* Cinfo::findMsg( const Conn* c, RecvFunc func ) const
{
	for (unsigned int i = 0; i < nFields_; i++) {
		Finfo* f = fieldArray_[i];
		if ( f->inConn( c->parent() ) == c  && f->recvFunc() == func )
			return f;
	}

	// Fallthrough. No matches were found, so ask the base class.
	if (base_ != this)
		return base_->findMsg( c, func );

	// Give up
	return 0;
}

/*
const Finfo* Cinfo::findMsg( const Conn* c ) const
{
	for (unsigned int i = 0; i < nFields_; i++) {
		if ( fieldArray_[i]->inConn( c->parent() ) == c )
			return fieldArray_[i];
	}

	// Fallthrough. No matches were found, so ask the base class.
	if (base_ != this)
		return base_->findMsg( c );

	// Give up
	return 0;
}
*/

/*
bool Cinfo::funcDrop( Element* e, const Conn* target  ) const
{
	bool dropped = 0;
	for (unsigned int i = 0; i < nFields_; i++) {
		dropped |= fieldArray_[i]->funcDrop( e, target );
	}

	// Fallthrough. Also drop any messages to the base class.
	if (base_ != this)
		dropped |= base_->funcDrop( e, target );
	return dropped;
}
*/

// Finfo* Cinfo::findRemoteMsg( Element* e, RecvFunc func ) const
Finfo* Cinfo::findRemoteMsg( Conn* c, RecvFunc func ) const
{
	Element* e = c->parent();
	for (unsigned int i = 0; i < nFields_; i++) {
		Finfo* f = fieldArray_[i];
		if ( f->outConn( e ) == c && f->matchRemoteFunc( e, func ) )
			return f;
	}

	// Fallthrough. No matches were found, so ask the base class.
	if (base_ != this)
		return base_->findRemoteMsg( c, func );

	// Give up
	return 0;
}

// Called by main() when starting up.
void Cinfo::initialize()
{
	map<string, Cinfo*>::iterator i;
	for (i = lookup().begin(); i != lookup().end(); i++) {
		// Make the Cinfo object on /classes
		Element* e = new CinfoWrapper( i->first, i->second );
		Element::classes()->adoptChild( e );

		// Identify base classes
		const Cinfo* c = find(i->second->baseName_);
		if (!c) {
			cerr << "Error: Cinfo::initalize(): Invalid base name '" <<
				i->second->baseName_ << "'\n";
			exit(0);
		}
		i->second->base_ = c;
	}

	// Do the field inits after the classes are inited, otherwise
	// there are unfilled base_ pointers.
	for (i = lookup().begin(); i != lookup().end(); i++) {
		// Initialize field equivalences
		for (unsigned int k = 0; k < i->second->nFields_; k++) {
			i->second->fieldArray_[k]->initialize( i->second );
		}
	}
}

Element* Cinfo::create(
	const string& name, Element* parent, const Element* proto ) const
{
	Element* e = createWrapper_( name, parent, proto );
	if ( e ) {
		if ( parent->adoptChild( e ) ) {
			return e;
		} else {
			delete e;
		}
	}
	return 0;
}

/*
Element* Cinfo::create(const string& name) const
{
	return createWrapper_( name, Element::root(), 0);
}
*/
