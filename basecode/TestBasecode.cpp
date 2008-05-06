/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include <fstream>
#include "header.h"

extern void connTest(); // Defined in UnitTests.cpp
extern void msgSrcTest(); // Defined in UnitTests.cpp
extern void msgFinfoTest(); // Defined in UnitTests.cpp
extern void cinfoTest(); // Defined in UnitTests.cpp
extern void finfoLookupTest(); // Defined in UnitTests.cpp
extern void valueFinfoTest(); // Defined in UnitTests.cpp
// extern void arrayFinfoTest(); // Defined in UnitTests.cpp
extern void transientFinfoDeletionTest(); // in UnitTests.cpp
extern void sharedFtypeTest(); // in SharedFtype.cpp
extern void sharedFinfoTest(); // in SharedFinfo.cpp
extern void lookupFinfoTest(); // in LookupFinfo.cpp
extern void copyTest(); // in Copy.cpp
extern void arrayElementTest(); // in ArrayElement.cpp

void testBasecode()
{
	connTest();
	msgSrcTest();
	msgFinfoTest();
	cinfoTest();
	finfoLookupTest();
	valueFinfoTest();
//	arrayFinfoTest();
	transientFinfoDeletionTest();
	sharedFtypeTest();
	sharedFinfoTest();
	lookupFinfoTest();
	copyTest();
	arrayElementTest();
}

#endif
