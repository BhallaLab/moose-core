/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "moose.h"

/**
* This should be almost the regular SrcFinfo::add
* operation, since at this point we have acquired
* a MsgSrc slot.
*/
bool ExtFieldFinfo::add(
	Element* e, Element* destElm, const Finfo* destFinfo) const
{ 
	return 0;
}

/**
* Again, this should be similar to the regular
* DestFinfo::respondToAdd operation, using the
* MsgDest slot.
*/
bool ExtFieldFinfo::respondToAdd(
	Element* e, Element* src, const Ftype *srcType,
	unsigned int& srcFuncId, unsigned int& returnFuncId,
	unsigned int& destIndex, unsigned int& numDest
) const
{
	return 0;
}
