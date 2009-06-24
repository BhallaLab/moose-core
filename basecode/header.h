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

/**
 * This header file includes the essential files in the correct order.
 * moose.h has this header plus more of the basecode headers.
 * You should NEVER have to do includes within any of your headers,
 * and if you do you are likely to get the order wrong.
 * Instead your .cpp should include header.h, or possibly moose.h,
 * and then some file specific headers.
 */

/// Here we set up an enhanced variant of assert, used in unit tests.
#ifdef DO_UNIT_TESTS
# define ASSERT( isOK, message ) \
	if ( !(isOK) ) { \
   cout << "\nERROR: Assert '" << #isOK << "' failed on line " << __LINE__ << "\nin file " << __FILE__ << ": " << message << endl; \
    exit( 1 ); \
} else { \
	   	cout << "."; \
}
#else
# define ASSERT( unused, message ) do {} while ( false )
#endif

#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <climits> // Required for g++ 4.3.2
#include <cstring> // Required for g++ 4.3.2
#include <cstdlib> // Required for g++ 4.3.2
#include <cstdio>  // Required for g++ 4.4


// Used for INT_MAX and UINT_MAX, but may be done within the compiler
// #include <limits.h>
//
#include <cassert>

using namespace std;

class Element;
class Conn;
class Finfo;

// This is here because parallel messaging needs a way to
// access PostMaster buffer from within all the templated
// Ftypes. Ugly.
extern void* getParBuf( const Conn* c, unsigned int size );
extern void* getAsyncParBuf( const Conn* c, unsigned int size );

// Another ugly global, this one for accessing the ids.
class Id;
class IdManager;
// extern IdManager* idManager();

#include "Eref.h"
#include "RecvFunc.h"
#include "../connections/Conn.h"
#include "../connections/ConnTainer.h"

#include "Ftype.h"
#include "FuncVec.h"
#include "Slot.h"
#include "Finfo.h"
#include "IdGenerator.h"
#include "Id.h"
#include "Msg.h"
#include "Element.h"

#endif // _HEADER_H
