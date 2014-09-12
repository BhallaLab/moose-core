/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ChanBase.h"
#include "ChanCommon.h"

///////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////
ChanCommon::ChanCommon()
			:
			Vm_( 0.0 ),
			Gbar_( 0.0 ), Ek_( 0.0 ),
			Gk_( 0.0 ), Ik_( 0.0 )
{
	;
}

ChanCommon::~ChanCommon()
{;}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void ChanCommon::vSetGbar( double Gbar )
{
	Gbar_ = Gbar;
}

double ChanCommon::vGetGbar() const
{
	return Gbar_;
}

void ChanCommon::vSetEk( double Ek )
{
	Ek_ = Ek;
}
double ChanCommon::vGetEk() const
{
	return Ek_;
}

void ChanCommon::vSetGk( double Gk )
{
	Gk_ = Gk;
}
double ChanCommon::vGetGk() const
{
	return Gk_;
}

void ChanCommon::vSetIk( double Ik )
{
	Ik_ = Ik;
}
double ChanCommon::vGetIk() const
{
	return Ik_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ChanCommon::vHandleVm( double Vm )
{
	Vm_ = Vm;
}

///////////////////////////////////////////////////
// Looks like a dest function, but it is only called
// from the child class. Sends out various messages.
///////////////////////////////////////////////////

void ChanCommon::sendProcessMsgs(  const Eref& e, const ProcPtr info )
{
		ChanBase::channelOut()->send( e, Gk_, Ek_ );
	// This is used if the channel connects up to a conc pool and
	// handles influx of ions giving rise to a concentration change.
		ChanBase::IkOut()->send( e, Ik_ );
	// Needed by GHK-type objects
		ChanBase::permeability()->send( e, Gk_ );
}


void ChanCommon::sendReinitMsgs(  const Eref& e, const ProcPtr info )
{
		ChanBase::channelOut()->send( e, Gk_, Ek_ );
	// Needed by GHK-type objects
		ChanBase::permeability()->send( e, Gk_ );
}

void ChanCommon::updateIk()
{
	Ik_ = ( Ek_ - Vm_ ) * Gk_;
}

double ChanCommon::getVm() const
{
	return Vm_;
}
