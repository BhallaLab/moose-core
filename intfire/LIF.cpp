/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "../biophysics/CompartmentBase.h"
#include "../biophysics/Compartment.h"
#include "IntFireBase.h"
#include "LIF.h"

using namespace moose;

const Cinfo* LIF::initCinfo()
{
	static string doc[] =
	{
		"Name", "LIF",
		"Author", "Upi Bhalla",
		"Description", "Leaky Integrate-and-Fire neuron"
	};
    static Dinfo< LIF > dinfo;
	static Cinfo lifCinfo(
				"LIF",
				IntFireBase::initCinfo(),
				0, 0,
				&dinfo,
                doc,
                sizeof(doc)/sizeof(string)
	);

	return &lifCinfo;
}

static const Cinfo* lifCinfo = LIF::initCinfo();

//////////////////////////////////////////////////////////////////
// Here we put the Compartment class functions.
//////////////////////////////////////////////////////////////////

LIF::LIF()
{;}

LIF::~LIF()
{;}

//////////////////////////////////////////////////////////////////
// LIF::Dest function definitions.
//////////////////////////////////////////////////////////////////

void LIF::vProcess( const Eref& e, ProcPtr p )
{
	fired_ = false;
	if ( p->currTime < lastEvent_ + refractT_ ) {
		Vm_ = vReset_;
		A_ = 0.0;
		B_ = 1.0 / Rm_;
		sumInject_ = 0.0;
		VmOut()->send( e, Vm_ );
	} else {
		Vm_ += activation_;
		activation_ = 0.0;
		if ( Vm_ > threshold_ ) {
			Vm_ = vReset_;
			lastEvent_ = p->currTime;
			fired_ = true;
			spikeOut()->send( e, p->currTime );
			VmOut()->send( e, Vm_ );
		} else {
			Compartment::vProcess( e, p );
		}
	}
}

void LIF::vReinit(  const Eref& e, ProcPtr p )
{
	activation_ = 0.0;
	fired_ = false;
	lastEvent_ = -refractT_; // Allow it to fire right away.
	Compartment::vReinit( e, p );
}
