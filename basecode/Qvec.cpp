/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

void Qvec::push_back( unsigned int thread, const Qinfo* q, const char* arg )
{
	unsigned int tbe = threadBlockEnd_[ thread ];
	unsigned int tbs = threadBlockNextStart_[ thread ];
	if ( tbe + sizeof( Qinfo ) + q->size() + threadOverlapProtection > tbs )
	{
		// Do some clever reallocation here.
	}
	// Need to figure out pos
	char* pos = &data_[ tbe ];
	memcpy( pos, q, sizeof( Qinfo ) );
	memcpy( pos + sizeof( Qinfo ), arg, q->size() );
	threadBlockEnd_[ thread ] += sizeof( Qinfo ) + q->size();
}
