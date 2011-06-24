/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "lookupSizeFromMesh.h"

/// Utility function

double lookupSizeFromMesh( const Eref& e, const SrcFinfo* sf )
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( sf->getBindIndex() );
	if ( !mfb ) return 1.0;
	if ( mfb->size() == 0 ) return 1.0;

	double size = 
		Field< double >::fastGet( e, (*mfb)[0].mid, (*mfb)[0].fid );

	if ( size <= 0 ) size = 1.0;

	return size;
}

