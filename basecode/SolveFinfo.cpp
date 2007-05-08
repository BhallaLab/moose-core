/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <map>
#include "Cinfo.h"
#include "ThisFinfo.h"
#include "SolveFinfo.h"

SolveFinfo::SolveFinfo( 
	Finfo** finfos, unsigned int nFinfos, const ThisFinfo* tf )
	: ThisFinfo( *tf )
{
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		finfos_.push_back( finfos[i] );
	}
	procSlot_ = tf->cinfo()->getSlotIndex( "process" );
}

const Finfo* SolveFinfo::match( Element* e, const string& name ) const
{
	if ( name == "" || name == "this" )
		return this;

	vector< Finfo* >::const_iterator i;

	for ( i = finfos_.begin(); i != finfos_.end(); i++ ) {
		if ( (*i)->name() == name )
			return *i;
	}
	return cinfo()->findFinfo( e, name );
}


void SolveFinfo::listFinfos( vector< const Finfo* >& flist ) const
{
	flist.push_back( this );
	cinfo()->listFinfos( flist );
}

/**
* Returns the Conn going from solved object e to the solver
*/
const Conn& SolveFinfo::getSolvedConn( const Element* e ) const
{
	vector< Conn >::const_iterator i = 
		e->connDestBegin( procSlot_ );
	assert( e->connDestEnd( procSlot_ ) != i );
	return *i;
}
