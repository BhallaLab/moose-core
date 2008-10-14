/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Mu_MPI_H
#define _Mu_MPI_H

/**
 * This class provides overridden MPI calls which are compatible with MUSIC.
 * Use these calls even if MUSIC is not linked.
 */

class MuMPI
{
public:
	static void setupMusic( );
	static void Init( int& argc, char**& argv );
	static void Finalize( );
	static const MPI::Intracomm& INTRA_COMM( );
	
private:
	static MPI::Intracomm communicator_;
};

#endif // _Mu_MPI_H
