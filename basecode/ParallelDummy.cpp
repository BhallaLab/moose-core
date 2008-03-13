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

//////////////////////////////////////////////////////////////////
//  This is here as a fallback in case we do not have the 
//  parallel code to use.
//////////////////////////////////////////////////////////////////
#ifndef USE_MPI
void* getParBuf( const Conn* c, unsigned int size )
{
	return 0;
}
void* getAsyncParBuf( const Conn* c, unsigned int size )
{
	return 0;
}
#endif

