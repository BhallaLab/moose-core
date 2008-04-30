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
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
) const
{
		/*
	return makeDynamicFinfo( e, src, srcType, srcfl, returnfl,
		destIndex, numThis, this );
		*/
		return 0;
}

/**
 * Directly call the recvFunc on the element with the string argument
 * typecast appropriately.
 */
bool ThisFinfo::strSet( Eref e, const std::string &s ) const
{
	/**
	 * \todo Here we will ask the Ftype to do the string conversion
	 * and call the properly typecast rfunc.
	 */
	return 0;
}

const Finfo* ThisFinfo::match( Element* e, const string& name ) const
{
	if ( name == "" || name == "this" )
		return this;

	return cinfo_->findFinfo( e, name );
}

const Finfo* ThisFinfo::match( 
				const Element* e, const ConnTainer* c ) const
{
	return cinfo_->findFinfo( e, c );
}

void ThisFinfo::listFinfos( vector< const Finfo* >& flist ) const
{
	flist.push_back( this );
	cinfo_->listFinfos( flist );
}
