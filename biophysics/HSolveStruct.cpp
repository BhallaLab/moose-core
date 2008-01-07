/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <queue>
#include "SynInfo.h"
#include "HSolveStruct.h"

void SynChanStruct::process( ProcInfo info ) {
	while ( !pendingEvents_->empty() &&
		pendingEvents_->top().delay <= info->currTime_ ) {
		*activation_ += pendingEvents_->top().weight / info->dt_;
		pendingEvents_->pop();
	}
	X_ = *modulation_ * *activation_ * xconst1_ + X_ * xconst2_;
	Y_ = X_ * yconst1_ + Y_ * yconst2_;
	Gk_ = Y_ * norm_;
	*activation_ = 0.0;
	*modulation_ = 1.0;
}
