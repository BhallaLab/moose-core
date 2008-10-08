/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifdef USE_MPI
#include <mpi.h>
#include "MMPI.h"
#endif

void MMPI::Init( int argc, char** argv )
{
#ifdef USE_MPI
#ifdef USE_MUSIC
	Id music( "/music" );
	assert( music.good() );
	
	MUSIC::setup* setup = new MUSIC::setup( argc, argv );
	set< MUSIC::setup* >( music(), "setup", setup_ );
	communicator_ = setup_->communicator( );
#else
	MPI::Init( argc, argv );
	communicator_ = MPI::COMM_WORLD;
#endif // USE_MUSIC
#endif // USE_MPI
}

void MMPI::Finalize( )
{
#ifdef USE_MPI
#ifdef USE_MUSIC
	Id music( "/music" );
	assert( music.good() );
	set( music(), "finalize" );
#else
	MPI::Finalize();
#endif // USE_MUSIC
#endif // USE_MPI
}

const MPI::Intracomm& MMPI::INTRA_COMM( )
{
	return communicator_;
}
