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
#include "ParFinfo.h"
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <typeinfo>


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
	//	new ParFinfo( "data" ),
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
// Here we handle passing messages to off-nodes
/////////////////////////////////////////////////////////////////////

// Just to help me remember how to use the typeid from RTTI.
// This will work only between identical compilers, I think.
const char* ftype2str( const Ftype *f )
{
	return typeid( *f ).name();
}


/////////////////////////////////////////////////////////////////////
// Utility function for accessing postmaster data buffer.
/////////////////////////////////////////////////////////////////////
/**
 * This function puts in the id of the message into the data buffer
 * and passes the next free location over to the calling function.
 * It internally increments the current location of the buffer.
 * If we don't use MPI, then this whole file is unlikely to be compiled.
 * So we define the dummy version of the function in DerivedFtype.cpp.
 */
void* PostMaster::innerGetParBuf( 
				unsigned int targetIndex, unsigned int size )
{
	if ( size + outBufPos_ > outBufSize_ ) {
		cout << "in getParBuf: Out of space in outBuf.\n";
		// Do something clever here to send another installment
		return 0;
	}
	*static_cast< unsigned int* >( 
			static_cast< void* >( outBuf_ + outBufPos_ ) ) =
			targetIndex;
	outBufPos_ += sizeof( unsigned int ) + size;
	return static_cast< void* >( outBuf_ + outBufPos_ - size );
}

#ifdef USE_MPI
void* getParBuf( const Conn& c, unsigned int size )
{
	PostMaster* pm = static_cast< PostMaster* >( c.data() );
	assert( pm != 0 );
	return pm->innerGetParBuf( c.targetIndex(), size );
}
#endif

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "Ftype2.h"
#include "setget.h"
#include "../builtins/Interpol.h"
#include "../builtins/Table.h"

void testPostMaster()
{
		cout << "\nTesting PostMaster";
		/*
		cout << "\n ftype2str( Ftype1< double > ) = " <<
				ftype2str( Ftype1< double >::global() );
		cout << "\n ftype2str( Ftype2< string, vector< unsigned int > > ) = " <<
				ftype2str( Ftype2< string, vector< unsigned int > >::global() );
		cout << "\n ftype2str( ValueFtype1< Table >::global() ) = " <<
				ftype2str( ValueFtype1< Table >::global() );
				*/
}
#endif
