/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MOOSE_H
#define _MOOSE_H

/**
 * The moose.h header is used when one makes MOOSE classes. It is not
 * needed for the external MOOSE API used to control MOOSE and 
 * access fields, but is used by developers implementing
 * their own MOOSE classes.
 */

#include "header.h"
#include <algorithm>
#include <map>
#include "Cinfo.h"

/**
 * \todo We may be able to remove the next 3 includes: 
 * MsgSrc, MsgDest and SimpleElement, if the Element is expanded
 * to give an interface to the Conn lookup functions.
 */
#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"
#include "ArrayElement.h"
#include "ArrayWrapperElement.h"

#include "DynamicFinfo.h"
#include "ValueFinfo.h"
#include "LookupFinfo.h"

#include "ProcInfo.h"
#include "send.h"
#include "DerivedFtype.h"
#include "Ftype2.h"
#include "Ftype3.h"
#include "Ftype4.h"
#include "ValueFtype.h"
#include "SharedFtype.h"
#include "LookupFtype.h"

#include "SrcFinfo.h"
#include "DestFinfo.h"
#include "SharedFinfo.h"

#include "setget.h"
#include "setgetLookup.h"

/// This is here because most classes derive from NeutralCinfo
extern const Cinfo* initNeutralCinfo();

// This is here because it is a common utility function.
// It is defined in strconv.cpp
extern void separateString( const string& s, vector< string>& v, 
				const string& separator );
// Another variant on it, same place.
extern void parseString( const string& s, vector< string>& v, 
				const char* separators );
#endif
