/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "DeletionMarkerFinfo.h"
#include "GlobalMarkerFinfo.h"

GlobalMarkerFinfo* GlobalMarkerFinfo::global()
{
		static GlobalMarkerFinfo* ret = new GlobalMarkerFinfo();

		return ret;
}
