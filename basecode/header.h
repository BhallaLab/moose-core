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

#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <climits> // Required for g++ 4.3.2
#include <cstring> // Required for g++ 4.3.2
#include <cstdlib> // Required for g++ 4.3.2


// Used for INT_MAX and UINT_MAX, but may be done within the compiler
// #include <limits.h>
//
#include <cassert>

using namespace std;

typedef unsigned int Slot;
typedef unsigned int FuncId;
extern const FuncId ENDFUNC;

class Element;
class Eref;

#include "Finfo.h"
#include "ProcInfo.h"
#include "Data.h"
#include "Element.h"
#include "Eref.h"
#include "Send.h"

#endif // _HEADER_H
