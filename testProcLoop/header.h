/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <iostream>
#include <vector>
#include <cassert>
#include <pthread.h>
#include "FuncBarrier.h"
#include "ProcInfo.h"
#include "Tracker.h"

extern void addToOutQ( const ProcInfo* p, const Tracker* t );
extern void* eventLoop( void* info );
extern void* mpiEventLoop( void* info );
extern void* shellEventLoop( void* info );
extern void allocQs();
extern void swapQ();
extern void swapMpiQ();
extern bool isAckPending();
extern void setBlockingParserCall( bool val );
