// VClamp.cpp --- 
// 
// Filename: VClamp.cpp
// Description: 
// Author: 
// Maintainer: 
// Created: Fri Feb  1 19:30:45 2013 (+0530)
// Version: 
// Last-Updated: Sat Feb  2 19:16:31 2013 (+0530)
//           By: subha
//     Update #: 112
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
                                          "Sends out current output of the clamping circuit");
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
        &reinit};

    static SharedFinfo proc("proc",
                            "Shared message to receive Process messages from the scheduler",
                            processShared, sizeof(processShared) / sizeof(Finfo*));

    static DestFinfo voltageIn("voltageIn",
                              "Handles membrane potential read from compartment",
                                new OpFunc1< VClamp, double>( &VClamp::setVin));

    static ValueFinfo< VClamp, double> holdingPotential("holdingPotential",
                                                        "Holding potential of the clamp circuit.",
                                                        &VClamp::setHoldingPotential,
                                                        &VClamp::getHoldingPotential);

    static Finfo* vclampFinfos[] = {
        currentOut(),
        &holdingPotential,
        &voltageIn,
        &proc
    };

    static string doc[] = {
        "Name", "VClamp",
        "Author", "Subhasis Ray",
        "Description", "Voltage clamp object for holding neuronal compartments at a specific voltage.",
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

VClamp::VClamp():holding_(-60e-3), current_(0.0), Rm_(1.0), vIn_(0.0)
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

void VClamp::setVin(double value)
{
    vIn_ = value;
}

void VClamp::process(const Eref& e, ProcPtr p)
{
    current_ = (holding_ - vIn_)/Rm_;
    currentOut()->send(e, p->threadIndexInGroup, current_);
}

void VClamp::reinit(const Eref& e, ProcPtr p)
{

    vector<Id> compartments;
    Field< Id >::getVec(e.id(), "voltageIn", compartments);
    cout << "name: " << e.element()->getName() << endl;
    for (int i = 0; i < compartments.size(); ++i){
        cout << "neighbour: "  << compartments[i].path() << endl;
    }
    vIn_ = 0.0;
}

#ifdef DO_UNIT_TESTS

#include "../shell/Shell.h"

void testVClampProcess()
{
    Shell * shell = reinterpret_cast<Shell*>(Id().eref().data());
    unsigned int size = 1;
    vector<int> dims(1, size);
    double Rm = 1.0;
    double Cm = 1.0;
    double dt = 0.01;
    double runtime = 10.0;
    double lambda = sqrt(Rm/Ra);
    

    Id comptId = shell->doCreate("Compartment", Id(), "compt", dims);
    assert(comptId != Id());
    assert(comptId()->dataHandler()->totalEntries() == size);

    Id clampId = shell->doCreate("VClamp", Id(), "vclamp", dims);
    MsgId mid = shell->doAddMsg("OneToOneMsg",
                                ObjId(comptId),
                                "VmOut",
                                ObjectId(clampId),
                                "voltageIn");
    assert(mid != Msg::bad);
    mid = shell->doAddMsg("OneToOneMsg",
                          ObjId(clampId),
                          "current",
                          ObjId(comptId),
                          "injectMsg");
    assert(mid != Msg::bad);    
}

#endif

// 
// VClamp.cpp ends here
