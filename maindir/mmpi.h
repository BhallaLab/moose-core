/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

namespace MPI
{
	class Intracomm;
};
extern const MPI::Intracomm INTRA_COMM;

class MMPI
{
public:
	void Init( int argc, char** argv );
	void Finalize( int argc, char** argv );

private:
};
