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
#include <map>
#include "header.h"
#include "Cinfo.h"
#include "ThisFinfo.h"

// The three includes below are needed because of the create function
// requiring an instantiable Element class. Could get worse if we
// permit multiple variants of Element, say an array form.

#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"

//////////////////////////////////////////////////////////////////
// Cinfo is the class info for the MOOSE classes.
//////////////////////////////////////////////////////////////////

Cinfo::Cinfo(const std::string& name,
				const std::string& author,
				const std::string& description,
				const std::string& baseName,
				Finfo** finfoArray,
				unsigned int nFinfos,
				const Ftype* ftype
)
		: name_(name), author_(author), 
		description_(description), baseName_(baseName),
		base_( 0 ), ftype_( ftype ), nSrc_( 0 ), nDest_( 0 )
{
	unsigned int i;
	for ( i = 0 ; i < nFinfos; i++ ) {
		finfoArray[i]->countMessages( nSrc_, nDest_ );
		finfos_.push_back( finfoArray[i] );
	}
	thisFinfo_ = new ThisFinfo( this );
	lookup()[name] = this;
	// This funny call is used to ensure that the root element is
	// created at static initialization time.
	Element::root();
}

const Cinfo* Cinfo::find( const string& name )
{
	map<string, Cinfo*>::iterator i;
	i = lookup().find(name);
	if ( i != lookup().end() )
		return i->second;
	return 0;
}

const Finfo* Cinfo::findFinfo( Element* e, const string& name ) const
{
	vector< Finfo* >::const_iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ ) {
		const Finfo* ret = (*i)->match( e, name );
		if ( ret )
			return ret;
	}

	// Fallthrough. No matches were found, so ask the base class.
	if (base_ != this)
		return base_->findFinfo( e, name );

	return 0;
}

const Finfo* Cinfo::findFinfo( 
		const Element* e, unsigned int connIndex) const
{
	vector< Finfo* >::const_iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ ) {
		const Finfo* ret = (*i)->match( e, connIndex );
		if ( ret )
			return ret;
	}

	// Fallthrough. No matches were found, so ask the base class.
	// This could be problematic, if the base class indices disagree
	// with the child class.
	///\todo: Figure out how to manage base class index alignment here
	if ( base_ && base_ != this)
		return base_->findFinfo( e, connIndex );

	return 0;
}

/*
void Cinfo::listFinfos( vector< Finfo* >& ret ) const
{
	ret.insert( ret.end(), finfos_.begin(), finfos.end() );
	
	// for ( unsigned int i = 0; i < nFinfos; i++ )
		// ret.push_back( finfoArray_[ i ] );

	if (base_ != this)
		const_cast< Cinfo* >( base_ )->listFinfos( ret );
}
*/


// Called by main() when starting up.
void Cinfo::initialize()
{
		/*
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
	*/
}

std::map<std::string, Cinfo*>& Cinfo::lookup()
{
	static std::map<std::string, Cinfo*> lookup_;
	return lookup_;
}

/**
 * Create a new element, complete with data, a set of Finfos and
 * the MsgSrc and MsgDest allocated.
 */
Element* Cinfo::create( const std::string& name ) const
{
	SimpleElement* ret = 
		new SimpleElement( name, nSrc_, nDest_, ftype_->create(1) );
	ret->addFinfo( thisFinfo_ );
	return ret;
}

/**
 * listFinfo fills in the finfo list onto the flist.
 * \todo: Should we nest the finfos in Cinfo? Or should we only show
 * the deepest layer?
 */
void Cinfo::listFinfos( vector< const Finfo* >& flist ) const
{
	flist.insert( flist.end(), finfos_.begin(), finfos_.end() );
}
