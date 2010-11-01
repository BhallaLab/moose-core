/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <iostream>
#include <unistd.h>
#include <mpi.h>
#include <pthread.h>
#include <cassert>
#include <stdlib.h>

using namespace std;

int main( int argc, char** argv )
{
	int numNodes = 1;
	int numCores = 1;
	int myNode = 0;
	int opt;

	int provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	MPI_Comm_size( MPI_COMM_WORLD, &numNodes );
	MPI_Comm_rank( MPI_COMM_WORLD, &myNode );

	while ( ( opt = getopt( argc, argv, "n:c:" ) ) != -1 ) {
		switch ( opt ) {
			case 'n': // Num nodes
				numNodes = atoi( optarg );
				break;
			case 'c': // Num cores
				numCores = atoi( optarg );
				break;
		}
	}

	cout << "on node " << myNode << ", numNodes = " << numNodes << ", numCores = " << numCores << endl;

	MPI_Finalize();
	return 0;
}
