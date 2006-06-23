/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HEADER_H
#define _HEADER_H
#define UNIX
//#define WINDOWS


#define MOOSE_THREADS 0

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
// Special headers needed for some compilers, like gcc2.96
#include <sstream>
#include <stdio.h>
// End of special headers

using namespace std;

// #ifdef WINDOWS
#ifdef NO_OFFSETOF
#define		FIELD_OFFSET( T, F ) \
	( unsigned long )( &T::F )
#else
#define		FIELD_OFFSET( T, F ) \
	static_cast< unsigned long >( offsetof( T, F ) )
#endif

class Element;
class Finfo;
class Field;
class Conn;
class Ftype;
typedef void ( *RecvFunc )( Conn* );
void dummyFunc0( Conn* c );

#include "Conn.h"
#include "Cinfo.h"
#include "Field.h"
#include "Finfo.h"
#include "MsgSrc.h"
// #include "AssignFinfo.h"
#include "Ftype.h"
#include "SharedFinfo.h"
#include "RelayFinfo.h"
#include "ValueFinfo.h"
#include "DestFinfo.h"
#include "SingleSrcFinfo.h"
#include "NSrcFinfo.h"
#include "ReturnFinfo.h"
#include "ObjFinfo.h"
#include "Element.h"
#include "Neutral.h"
#include "ProcInfo.h"

#endif // _HEADER_H
