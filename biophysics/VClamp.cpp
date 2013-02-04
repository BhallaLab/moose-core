// VClamp.cpp --- 
// 
// Filename: VClamp.cpp
// Description: 
// Author: 
// Maintainer: 
// Created: Fri Feb  1 19:30:45 2013 (+0530)
// Version: 
// Last-Updated: Mon Feb  4 19:06:29 2013 (+0530)
//           By: subha
//     Update #: 244
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// Implementation of Voltage Clamp
// 
// 

// Change log:
// 
// 
// 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#include "header.h"
#include "Dinfo.h"
#include "VClamp.h"

using namespace moose;

SrcFinfo1< double >* VClamp::currentOut()
{
    static SrcFinfo1< double > currentOut("currentOut",
                                          "Sends out current output of the clamping circuit. This should be"
                                          " connected to the `injectMsg` field of a compartment to voltage clamp"
                                          " it.");
    return &currentOut;
}


const Cinfo * VClamp::initCinfo()
{
    static DestFinfo process("process",
                             "Handles 'process' call on each time step.",
                             new ProcOpFunc< VClamp >( &VClamp::process));
    static DestFinfo reinit("reinit",
                            "Handles 'reinit' call",
                            new ProcOpFunc< VClamp >( &VClamp::reinit));
    static Finfo * processShared[] = {
        &process,
        &reinit
    };

    static SharedFinfo proc("proc",
                            "Shared message to receive Process messages from the scheduler",
                            processShared, sizeof(processShared) / sizeof(Finfo*));

    static ValueFinfo< VClamp, double> holdingPotential("holdingPotential",
                                                        "Holding potential of the clamp circuit.",
                                                        &VClamp::setHoldingPotential,
                                                        &VClamp::getHoldingPotential);
    static ValueFinfo< VClamp, double> gain("gain",
                                            "Access resistance of the clamp circuit. The amount of current injected"
                                            " is scaled down by this value. The default small value should suffice"
                                            " for normal cases. Setting this incorrectly may cause oscillations. A"
                                            " good default is Compartment.Cm/dt where dt is the timestep of"
                                            " integration for the compartment.",
                                                        &VClamp::setGain,
                                                        &VClamp::getGain);
    static ReadOnlyValueFinfo< VClamp, double> current("current",
                                            "The amount of current injected by the clamp into the membrane.",
                                            &VClamp::getCurrent);


    static DestFinfo voltageIn("voltageIn",
                              "Handles membrane potential read from compartment. The `VmOut` message"
                               " of the Compartment object should be connected to this.",
                                new OpFunc1< VClamp, double>( &VClamp::setVin));

    static Finfo* vclampFinfos[] = {
        currentOut(),
        &holdingPotential,
        &gain,
        &current,
        &voltageIn,
        &proc
    };

    static string doc[] = {
        "Name", "VClamp",
        "Author", "Subhasis Ray",
        "Description", "Voltage clamp object for holding neuronal compartments at a specific"
        " voltage."
        "\n"
        "\nUsage: Connect the `currentOut` source of VClamp to `injectMsg` of"
        "\ndest of Compartment. Connect the `VmOut` source of Compartment to"
        "\n`voltageIn` dest of VClamp. Either set `holdingPotential` field to a"
        "\nfixed value, or connect an appropriate source of command potential"
        "\n(like the `outputOut` message of an appropriately configured"
        "\nPulseGen) to `set_holdingPotential` dest.",
    };
    static Cinfo vclampCinfo(
        "VClamp",
        Neutral::initCinfo(),
        vclampFinfos,
        sizeof(vclampFinfos) / sizeof(Finfo*),
        new Dinfo< VClamp >(),
        doc,
        sizeof(doc)/sizeof(string));

    return &vclampCinfo;            
}

static const Cinfo * vclampCinfo = VClamp::initCinfo();

VClamp::VClamp(): vIn_(0.0), holding_(0.0), current_(0.0), gain_(1.0)
{
}

VClamp::~VClamp()
{
    ;
}

void VClamp::setHoldingPotential(double value)
{
    holding_ = value;
}

double VClamp::getHoldingPotential() const
{
    return holding_;
}

void VClamp::setGain(double value)
{
    gain_ = value;
}

double VClamp::getGain() const
{
    return gain_;
}

double VClamp::getCurrent() const
{
    return current_;
}

void VClamp::setVin(double value)
{
    vIn_ = value;
}

void VClamp::process(const Eref& e, ProcPtr p)
{
    current_ = (holding_ - vIn_) * gain_;
    currentOut()->send(e, p->threadIndexInGroup, current_);
}

void VClamp::reinit(const Eref& e, ProcPtr p)
{

    vector<Id> compartments;
    vIn_ = 0.0;
}

// 
// VClamp.cpp ends here
