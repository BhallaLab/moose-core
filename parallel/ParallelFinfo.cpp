/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ParallelFinfo.h"
#include "ParallelMsgSrc.h"
#include "PostMaster.h"
#include "PostMasterWrapper.h"

//////////////////////////////////////////////////////////////////
// ParallelDummyFinfo functions
//////////////////////////////////////////////////////////////////

const Ftype* ParallelDummyFinfo::ftype() const
{
	static const Ftype0 myFtype;
	return &myFtype;
}

//////////////////////////////////////////////////////////////////
// ParallelDestFinfo functions
//////////////////////////////////////////////////////////////////


Finfo* ParallelDestFinfo::respondToAdd( 
				Element* e, const Finfo* sender )
{
	static ParallelDummyFinfo df( sender->name() );
	if ( sharesConn() ) {
	// Need to insert dummy RecvFuncs in any shared MsgSrcs.
	// Locate Shared Finfo
	// Scan through its list of SharedOut, put dummies on them.
	// Or, as here, simply create a RelayFinfo to deal with it.
		cerr << "Warning:DestFinfo::respondToAdd::" << e->name() <<
			"." << name() << ", " << sender->name() << "\n";
		cout << "Shared message handling not yet implementd, trying relay\n";
		return makeRelayFinfo( e );
	}
	PostMasterWrapper* pm = dynamic_cast< PostMasterWrapper* >( e );
	if ( !pm ) {
			cerr << "Error: ParallelDestFinfo on non-postmaster element\n";
			return 0;
	}
	pm->addSender( sender );
	// pm->outgoingSize_.pushBack( sender.size() );
	// pm->outgoingSchedule_.push_back( getSchedule( sender ) );
	df.addRecvFunc( e, sender->ftype()->getPostRecvFunc(), 0L );
	df.setInConn( inConn( e ) );
	return( &df );
}

//////////////////////////////////////////////////////////////////
// ParallelSrcFinfo functions
//////////////////////////////////////////////////////////////////

bool ParallelSrcFinfo::add( Element* e, Field& destfield, bool useSharedConn )
{
	// Hack here: We are telling the destfield that it is its own src.
	Finfo* dest = destfield.respondToAdd( destfield.getFinfo() );
	// Finfo* dest = destfield->getFinfo();
	ParallelMsgSrc* p = static_cast< ParallelMsgSrc* >( getSrc_( e ) );
	if ( p->add(
		dest->recvFunc(), dest->ftype(),
		dest->inConn( destfield.getElement() ) )
		)
		return 1;
	finfoErrorMsg( "ParallelSrcFinfo::add", destfield );
	return 0;
}
