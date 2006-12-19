/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#define DATA_TAG 0
#define BUF_SIZE 100

#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main( int argc, char** argv )
{
	MPI::Init( argc, argv );
	int totalNodes = MPI::COMM_WORLD.Get_size();
	int myNode = MPI::COMM_WORLD.Get_rank();
	int nextNode = myNode + 1;
	if ( nextNode >= totalNodes )
			nextNode = 0;
	int prevNode = myNode - 1;
	if ( prevNode < 0 )
			prevNode = totalNodes - 1;

	MPI::Comm& comm( MPI::COMM_WORLD ); 
	MPI::Request request( 0 );
	char buf[BUF_SIZE];
	sprintf( buf, "Start node: %d. ", myNode );
	char inbuf[BUF_SIZE];
	int i;

	for ( i = 0; i < 10; i++ ) {
		request = comm.Irecv( inbuf, BUF_SIZE, MPI_CHAR, prevNode, DATA_TAG );
		comm.Send( buf, strlen( buf ) + 1, MPI_CHAR, nextNode, DATA_TAG );
		request.Wait();
		sprintf( buf, "%s(%d,%d)", inbuf, i, prevNode );
	}
	printf( "%s\n", buf );
	
	MPI::Finalize();
	return 0;
}
