/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment,
 ** also known as GENESIS 3 base code.
 **           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#ifndef  CYTHON_HEADER_INC
#define  CYTHON_HEADER_INC

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
 * Looks up data entries.
 */
typedef unsigned int DataId;

/**
 * Identifies data entry on an Element. This is a global index,
 * in that it does not refer to the array on any given node, but uniquely
 * identifies the entry over the entire multinode simulation.
 */
extern const unsigned int ALLDATA; // Defined in consts.cpp

/// Identifies bad DataIndex or FieldIndex in ObjId.
extern const unsigned int BADINDEX; // Defined in consts.cpp

/**
 * Index into Element::vector< vector< MsgFuncBinding > > msgBinding_;
 */
typedef unsigned short BindIndex;

extern const double PI;	// Defined in consts.cpp
extern const double NA; // Defined in consts.cpp
extern const double FaradayConst; // Defined in consts.cpp
class ProcInfo;
class Cinfo;
class DestFinfo;
class Msg;
class Eref;

template<typename T>
class SrcFinfo1;

typedef const ProcInfo* ProcPtr;


#endif   /* ----- #ifndef CYTHON_HEADER_INC  ----- */
