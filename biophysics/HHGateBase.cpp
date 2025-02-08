/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment.
 **           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
#include "HHGateBase.h"

///////////////////////////////////////////////////
// Core class functions
///////////////////////////////////////////////////
HHGateBase::HHGateBase() : originalChanId_(0), originalGateId_(0)
{
    cerr << "# HHGateBase::HHGateBase() should never be called" << endl;
}

HHGateBase::HHGateBase(Id originalChanId, Id originalGateId)
    : originalChanId_(originalChanId), originalGateId_(originalGateId)
{
    // cerr << "# HHGateBase::HHGateBase(): originalChanId:" << originalChanId << ", originalGateId: " << originalGateId << endl;
    ;
}

///////////////////////////////////////////////////////////////////////
// Utility funcs
///////////////////////////////////////////////////////////////////////

bool HHGateBase::checkOriginal(Id id, const string& field) const
{
    if(id == originalGateId_)
        return true;

    cout << "Warning: HHGateBase: attempt to set field '" << field << "' on "
         << id.path() << ", which is not the original Gate element. Ignored.\n";
    return false;
}

bool HHGateBase::isOriginalChannel(Id id) const
{
    // cerr << "# Received: " << id << ", original: " << originalChanId_ << endl;
    return (id == originalChanId_);
}

bool HHGateBase::isOriginalGate(Id id) const
{
    return (id == originalGateId_);
}

Id HHGateBase::originalChannelId() const
{
    return originalChanId_;
}

Id HHGateBase::originalGateId() const
{
    return originalGateId_;
}
