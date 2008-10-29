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
	Finfo** finfos, unsigned int nFinfos, const ThisFinfo* tf, const string& doc )
	: ThisFinfo( *tf )
{
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		finfos[i]->addFuncVec( tf->cinfo()->name() + ".solve" );
		finfos_.push_back( finfos[i] );
	}
	/// \todo Need to fix.
	procSlot_ = tf->cinfo()->getSlot( "process" ).msg();
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
* Why is this on the solver?
* \todo Also, will need to do something about the memory of the Conn
* deprecated
const Conn* SolveFinfo::getSolvedConn( const Element* e ) const
{
	return e->msg( procSlot_ )->findConn( 0, 0 );
}
*/
