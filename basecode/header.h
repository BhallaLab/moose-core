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
 * Looks up Synchronous messages on elements
 */
typedef unsigned int SyncId;

/**
 * Index into Element::vector< vector< MsgFuncBinding > > msgBinding_;
 */
typedef unsigned short BindIndex;

class Element;
class Eref;
class OpFunc;
class Qinfo;
class Data;
class Cinfo;
class SetGet;

#include "Id.h"
#include "DataId.h"
#include "FullId.h"
#include "Finfo.h"
#include "DestFinfo.h"
#include "SimGroup.h"
#include "ProcInfo.h"
#include "Cinfo.h"
#include "Data.h"
#include "Msg.h"
// #include "MsgManager.h"
// #include "SingleMsg.h"
// #include "OneToOneMsg.h"
// #include "OneToAllMsg.h"
#include "MsgFuncBinding.h"
#include "Qinfo.h"
#include "Dinfo.h"
#include "DataHandler.h"
#include "ZeroDimHandler.h"
#include "OneDimHandler.h"
#include "ZeroDimGlobalHandler.h"
#include "OneDimGlobalHandler.h"
#include "Element.h"
#include "Eref.h"
#include "PrepackedBuffer.h"
#include "Conv.h"
#include "SrcFinfo.h"
#include "../shell/Shell.h"
#include "FieldDataHandler.h"
#include "SetGet.h"
#include "OpFunc.h"
#include "EpFunc.h"
#include "UpFunc.h"
#include "ValueFinfo.h"
#include "SharedFinfo.h"
#include "FieldElementFinfo.h"
#include "../shell/Neutral.h"

#endif // _HEADER_H
