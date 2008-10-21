/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifdef USE_MPI
#include "moose.h"
#include <mpi.h>
#ifdef USE_MUSIC
#include <music.hh>
#include "music/Music.h"
#endif // USE_MUSIC
#include "MuMPI.h"

MPI::Intracomm MuMPI::communicator_ = MPI::Intracomm();

void MuMPI::Init( int& argc, char**& argv )
{

#ifdef USE_MPI
#ifdef USE_MUSIC
	communicator_ = Music::setup( argc, argv );
#else
	MPI::Init( argc, argv );
	communicator_ = MPI::COMM_WORLD;
#endif // USE_MUSIC
#endif // USE_MPI
}

void MuMPI::Finalize( )
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

const MPI::Intracomm& MuMPI::INTRA_COMM( )
{
	return communicator_;
}

#endif
