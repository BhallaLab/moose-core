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

#include <math.h>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include <typeinfo> // used in Conv.h to extract compiler independent typeid
#include <climits> // Required for g++ 4.3.2
#include <cstring> // Required for g++ 4.3.2
#include <cstdlib> // Required for g++ 4.3.2

// Used for INT_MAX and UINT_MAX, but may be done within the compiler
// #include <limits.h>
//
#include <cassert>

using namespace std;

/**
 * Looks up and uniquely identifies functions, on a per-Cinfo basis.
 * These are NOT global indices to identify the function.
 */
typedef unsigned int FuncId;

/**
 * Looks up and uniquely identifies Msgs. This is a global index
 */
typedef unsigned int MsgId;

/**
 * Index into Element::vector< vector< MsgFuncBinding > > msgBinding_;
 */
typedef unsigned short BindIndex;

/**
 * Identifier for threads.
 */
typedef unsigned short ThreadId;

extern const double PI;	// Defined in consts.cpp
extern const double NA; // Defined in consts.cpp

class Element;
class Eref;
class OpFunc;
class Qinfo;
class Cinfo;
class SetGet;
class FuncBarrier;

#include "doubleEq.h"
#include "Id.h"
#include "DataId.h"
#include "ObjId.h"
#include "Finfo.h"
#include "DestFinfo.h"
#include "SimGroup.h"
#include "ProcInfo.h"
#include "Cinfo.h"
#include "MsgFuncBinding.h"
#include "../msg/Msg.h"
#include "Qvec.h"
#include "Qinfo.h"
#include "Dinfo.h"
#include "DataHandler.h"
#include "ZeroDimGlobalHandler.h"
#include "ZeroDimHandler.h"
#include "OneDimGlobalHandler.h"
#include "OneDimHandler.h"
#include "DataDimensions.h"
#include "AnyDimGlobalHandler.h"
#include "AnyDimHandler.h"
#include "Element.h"
#include "Eref.h"
#include "PrepackedBuffer.h"
#include "Conv.h"
#include "SrcFinfo.h"
#include "FieldDataHandlerBase.h"
#include "FieldDataHandler.h"

extern DestFinfo* receiveGet();
class Neutral;
#include "OpFuncBase.h"
#include "SetGet.h"
#include "OpFunc.h"
#include "EpFunc.h"
#include "UpFunc.h"
#include "ProcOpFunc.h"
#include "ValueFinfo.h"
#include "LookupValueFinfo.h"
#include "SharedFinfo.h"
#include "FieldElementFinfo.h"
#include "ReduceBase.h"
#include "../shell/Neutral.h"

#endif // _HEADER_H
