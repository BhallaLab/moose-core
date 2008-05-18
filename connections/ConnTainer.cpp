/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SimpleConn.h"
#include "One2AllConn.h"
#include "../utility/SparseMatrix.h"
#include "Many2ManyConn.h"
#include "One2OneMapConn.h"

const unsigned int ConnTainer::Default = UINT_MAX;
const unsigned int ConnTainer::Simple = 0;
const unsigned int ConnTainer::One2All = 2;
const unsigned int ConnTainer::Many2Many = 4;
const unsigned int ConnTainer::One2OneMap = 8;

/*
ConnTainer* findExistingConnTainer( Eref src, Eref dest, 
	int srcMsg, int destMsg, 
	int srcFuncId, unsigned int destFuncId,
	unsigned int connTainerOption );
	*/

/**
 * Add a new message with a container decided by the optional final 
 * argument. It defaults to a sensible guess based on the indices of the
 * src and dest Erefs.
 * Cases:
 * 0: src = simple, dest = simple:				SimpleConn
 * 1: src = simple, dest = Array, single:		check and fill One2Many 
 * 2: src = simple, dest = Array, AnyIndex:	One2All
 * 3: src = Array, singleIndex, dest = simple:	SimpleConn
 * 4: src = Array, AnyIndex, dest = simple:	All2One
 * 5: src = Array,single, dest = Array,single:	check and fill Many2Many
 * 6: src = Array,Any, dest = Array,single:	check and fill All2Many
 * 7: src = Array,Single, dest = Array,Any:	check and fill Many2All
 * 8: src = Array,Any, dest = Array,Any, size match: One2OneMap
 * 9: src = Array,Any, dest = Array,Any, size match: All2AllMap
 * 	How do I set up a fully connected matrix? Override the default.
 */
ConnTainer* selectConnTainer( Eref src, Eref dest, 
	unsigned int srcMsg, unsigned int destMsg,
	unsigned int srcIndex, unsigned int destIndex,
	unsigned int connTainerOption )
{
	// This is flawed: We could have a single-entry ArrayElement that
	// later gets expanded.
	
	if ( connTainerOption == ConnTainer::Default ) {
		int srcNum;
		if ( src.e->numEntries() == 1 ) srcNum = 0;
		else if ( src.i == Id::AnyIndex ) srcNum = 2;
		else srcNum = 1;
	
		int destNum;
		if ( dest.e->numEntries() == 1 ) destNum = 0;
		else if ( dest.i == Id::AnyIndex ) destNum = 2;
		else destNum = 1;
		connTainerOption = srcNum * 3 + destNum;
	}

	ConnTainer* ct = 0;
		
	switch ( connTainerOption )
	{
		case 0:
			ct = new SimpleConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		case 1:
			ct = new One2ManyConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		case 2:
			ct = new One2AllConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		case 3:
			ct = new Many2OneConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		case 4:
			ct = new Many2ManyConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
			/*
			ct = new SimpleConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
			*/
		break;
		case 5:
			ct = new Many2AllConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		case 6:
			ct = new All2OneConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		case 7:
			ct = new All2ManyConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		case 8:
			ct = new One2OneMapConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		case 9:
			ct = new All2AllMapConnTainer( src, dest, srcMsg, destMsg,
				srcIndex, destIndex );
		break;
		default:
			return 0;
		break;
	}
	return ct;
}

/*
ConnTainer* findExistingConnTainer( Eref src, Eref dest, 
	int srcMsg, int destMsg, 
	int srcFuncId, unsigned int destFuncId,
	unsigned int connTainerOption )
{
	if ( srcMsg >= 0 ) {
		Msg* m = src->varMsg( static_cast< unsigned int >( srcMsg ) );
		m = m->matchByFuncId( src.e, destFuncId );
		vector< ConnTainer* >::iterator i;
		for ( i = m->varBegin(); i != m->varEnd(); i++ ) {
			if ( (*i)->e2() == dest.e && (*i)->msg2() == destMsg && 
				(*i)->option() == connTainerOption )
				return *i;
		}
	}
	return 0;
}
*/
