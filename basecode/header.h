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

/// Here we set up an enhanced variant of assert, used in unit tests.
#ifndef NDEBUG
# define ASSERT( isOK, message ) \
	if ( !(isOK) ) { \
   cout << "\nERROR: Assert '" << #isOK << "' failed on line " << __LINE__ << "\nin file " << __FILE__ << ": " << #message << endl; \
    exit( 1 ); \
} else { \
	   	cout << "."; \
}
#else
# define ASSERT( unused, message ) do {} while ( false )
#endif

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <cassert>

extern const unsigned int BAD_ID;

using namespace std;

class Element;
class Conn;
class Finfo;

#include "RecvFunc.h"
#include "Conn.h"
#include "Ftype.h"
#include "Finfo.h"
#include "Element.h"

#endif // _HEADER_H
