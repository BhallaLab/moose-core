/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <map>
#include "Cinfo.h"
#include "ThisFinfo.h"

/**
 * Check the request to form a message, and return true with 
 * the necessary information if all is well.
 * Here we need to create a dynamic Finfo to manage the message,
 * as the 'thisFinfo' does not manage predefined messages.
 */
bool ThisFinfo::respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcfl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numThis
) const
{
		/*
	return makeDynamicFinfo( e, src, srcType, srcfl, returnfl,
		destIndex, numThis, this );
		*/
		return 0;
}
			
unsigned int ThisFinfo::srcList(
					const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

unsigned int ThisFinfo::destList(
					const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

/**
 * Directly call the recvFunc on the element with the string argument
 * typecast appropriately.
 */
bool ThisFinfo::strSet( Element* e, const std::string &s ) const
{
	/**
	 * \todo Here we will ask the Ftype to do the string conversion
	 * and call the properly typecast rfunc.
	 */
	return 0;
}

const Finfo* ThisFinfo::match( Element* e, const string& name ) const
{
	if ( name == "" )
		return this;

	return cinfo_->findFinfo( e, name );
}

const Finfo* ThisFinfo::match( 
				const Element* e, unsigned int connIndex) const
{
	return cinfo_->findFinfo( e, connIndex );
}

void ThisFinfo::listFinfos( vector< const Finfo* >& flist ) const
{
	flist.push_back( this );
	cinfo_->listFinfos( flist );
}
