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
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <typeinfo> // used in Conv.h to extract compiler independent typeid

#include <cassert>

using namespace std;

constexpr double PI = 3.141592653589793;
constexpr double NA = 6.0221415e23;
constexpr double FaradayConst =  96485.3415; // s A / mol
constexpr double GasConst = 8.3144621; // R, units are J/(K.mol)

typedef unsigned short BindIndex;

/**
 * Looks up and uniquely identifies functions, on a per-Cinfo basis.
 * These are NOT global indices to identify the function.
 */
typedef unsigned int FuncId;

/**
 * Looks up data entries.
 */
typedef unsigned int DataId;

/// Used by ObjId and Eref
const unsigned int ALLDATA = ~0U;

/// Used by ObjId and Eref
const unsigned int BADINDEX = ~1U;

#include "doubleEq.h"
#include "Id.h"
#include "ObjId.h"
#include "Cinfo.h"

#include "Finfo.h"
#include "DestFinfo.h"
#include "ProcInfo.h"
#include "MsgFuncBinding.h"
#include "../msg/Msg.h"
#include "Dinfo.h"
#include "MsgDigest.h"
#include "Element.h"
#include "DataElement.h"
#include "GlobalDataElement.h"
#include "LocalDataElement.h"
#include "Eref.h"
#include "Conv.h"
#include "SrcFinfo.h"

extern DestFinfo* receiveGet();
#include "OpFuncBase.h"
#include "HopFunc.h"
#include "SetGet.h"
#include "OpFunc.h"
#include "EpFunc.h"
#include "ProcOpFunc.h"
#include "ValueFinfo.h"
#include "LookupValueFinfo.h"
#include "ValueFinfo.h"
#include "SharedFinfo.h"
#include "FieldElementFinfo.h"
#include "FieldElement.h"
#include "../shell/Neutral.h"


#endif // _HEADER_H
