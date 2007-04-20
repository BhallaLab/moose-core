/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "PostMaster.h"
#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * Declaration of the neutralCinfo() function is here because
 * we ensure the correct sequence of static initialization by having
 * each Cinfo use this call to find its base class. Most Cinfos
 * inherit from neutralCinfo. This function
 * uses the common trick of having an internal static value which
 * is created the first time the function is called.
 * The function for neutralCinfo has an additional line to statically
 * initialize the root element.
 */
const Cinfo* initPostMasterCinfo()
{
	static Finfo* postMasterFinfos[] = 
	{
		new ValueFinfo( "localNode", 
					ValueFtype1< unsigned int >::global(),
					GFCAST( &PostMaster::getMyNode ), &dummyFunc 
		),
		new ValueFinfo( "remoteNode", 
					ValueFtype1< unsigned int >::global(),
					GFCAST( &PostMaster::getRemoteNode ),
					RFCAST( &PostMaster::setRemoteNode )
		),
	};

	static Cinfo postMasterCinfo(
				"PostMaster",
				"Upi Bhalla",
				"PostMaster object. Manages parallel communications.",
				initNeutralCinfo(),
				postMasterFinfos,
				sizeof( postMasterFinfos ) / sizeof( Finfo* ),
				ValueFtype1< PostMaster >::global()
	);

	return &postMasterCinfo;
}

static const Cinfo* postMasterCinfo = initPostMasterCinfo();

//////////////////////////////////////////////////////////////////
// Here we put the PostMaster class functions.
//////////////////////////////////////////////////////////////////
PostMaster::PostMaster()
	: remoteNode_( 0 )
{
	localNode_ = MPI::COMM_WORLD.Get_rank();
}

//////////////////////////////////////////////////////////////////
// Here we put the PostMaster Moose functions.
//////////////////////////////////////////////////////////////////

unsigned int PostMaster::getMyNode( const Element* e )
{
		return static_cast< PostMaster* >( e->data() )->localNode_;
}

unsigned int PostMaster::getRemoteNode( const Element* e )
{
		return static_cast< PostMaster* >( e->data() )->remoteNode_;
}

void PostMaster::setRemoteNode( const Conn& c, unsigned int node )
{
		static_cast< PostMaster* >( c.data() )->remoteNode_ = node;
}

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "Ftype2.h"
#include "setget.h"

void testPostMaster()
{
		cout << "\nTesting PostMaster";
}
#endif
