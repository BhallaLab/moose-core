/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "AsyncDestFinfo.h"


AsyncDestFinfo::AsyncDestFinfo( const string& name, 
	const Ftype *f, RecvFunc rfunc, const string& doc, unsigned int destIndex ) 
	: DestFinfo( name, Ftype2< char*, unsigned int >::global(), 
		rfunc, doc, destIndex )
{ ; }


/**
 * This responds to a message request on a postmaster directed to
 * a remote node. All the fancy setup work is done by the Shell,
 * which also tells the remote node what it should do to complete the
 * message.
 */
bool AsyncDestFinfo::respondToAdd(
					Eref e, Eref dest, const Ftype* srcType,
					unsigned int& myFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
) const
{
	assert( srcType != 0 );
	assert( e.e != 0 );
	assert( dest.e != 0 );
	assert( e->className() == "PostMaster" );
	// PostMaster* post = static_cast< PostMaster* >( e->data());
	returnFuncId = srcType->asyncFuncId();
	destMsgId = msg();
	destIndex = 0; // not used.
	return 1;
}
