// Efield.cpp --- 
// 
// Filename: Efield.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Wed Jul 20 14:52:24 2011 (+0530)
// Version: 
// Last-Updated: Thu Oct  6 08:38:07 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 123
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
// 
// 

// Code:

#include "header.h"
#include "moose.h"
#include "Efield.h"

const Cinfo * initEfieldCinfo()
{
    static Finfo * processShared[] = {
        new DestFinfo( "process", Ftype1<ProcInfo>::global(),
                       RFCAST(&Efield::processFunc)),
        new DestFinfo( "reinit", Ftype1<ProcInfo>::global(),
                       RFCAST( &Efield::reinitFunc)),
        };

    static Finfo* process = new SharedFinfo( "process", processShared,
                                             sizeof(processShared)/sizeof(Finfo*));
    static Finfo* efieldFinfos[] = 
            {
            new ValueFinfo("x", ValueFtype1<double>::global(),
                           GFCAST( &Efield::getX),
                           RFCAST( &Efield::setX),
                           "X coordinate of the efield object."),
            new ValueFinfo("y", ValueFtype1<double>::global(),
                           GFCAST( &Efield::getY),
                           RFCAST( &Efield::setY),
                           "Y coordinate of the efield object."),
            new ValueFinfo("z", ValueFtype1<double>::global(),
                           GFCAST( &Efield::getZ),
                           RFCAST( &Efield::setZ),
                           "Z coordinate of the efield object."),
            new ValueFinfo("scale", ValueFtype1<double>::global(),
                           GFCAST( &Efield::getScale),
                           RFCAST( &Efield::setScale),
                           "1/(4 * pi * sigma) - where sigma is the conductance of the "
                           "extracellular medium. This is multiplied to sum( Im_i/d_i) over all "
                           "the compartments connected to this efield object [ where Im_i is "
                           "membrane current of the i-th compartment and d_i is the distance of "
                           "the i-th compartment.] to compute the local field potential at the "
                           "efield."),
            new ValueFinfo("potential", ValueFtype1<double>::global(),
                           GFCAST( &Efield::getPotential),
                           &dummyFunc,
                           "The computed local field potential at the Efield object."
                           ),
            process,
            new DestFinfo("currentDest", Ftype1<double>::global(),
                          RFCAST( &Efield::currentFunc )),
            };

    static SchedInfo schedInfo[] = {{process, 0, 0}};

    static string doc[] = {
        "Name", "Efield",
        "Author", "Subhasis Ray, 2011, NCBS",
        "Description", "Electrode for measuring local field potential due to membrane currents in neuronal compartments.",
    };

    static Cinfo efieldCinfo(
            doc,
            sizeof(doc)/sizeof(string),
            initNeutralCinfo(),
            efieldFinfos,
            sizeof(efieldFinfos)/sizeof(Finfo*),
            ValueFtype1<Efield>::global(),
            schedInfo, 1);
    return &efieldCinfo;
}

static const Cinfo * efieldCinfo = initEfieldCinfo();

Efield::Efield(): x_(0.0), y_(0.0), z_(0.0), scale_(-3.33e4), pot_(0.0), innerPot_(0.0)
{
    ;
}

void Efield::setX(const Conn* conn, double value)
{
    static_cast<Efield*>(conn->data())->x_ = value;
}

double Efield::getX(Eref eref)
{
    return static_cast<Efield*>(eref.data())->x_;
}

void Efield::setY(const Conn* conn, double value)
{
    static_cast<Efield*>(conn->data())->y_ = value;
}

double Efield::getY(Eref eref)
{
    return static_cast<Efield*>(eref.data())->y_;
}
    
void Efield::setZ(const Conn* conn, double value)
{
    static_cast<Efield*>(conn->data())->z_ = value;
}

double Efield::getZ(Eref eref)
{
    return static_cast<Efield*>(eref.data())->z_;
}

void Efield::setScale(const Conn* conn, double value)
{
    static_cast<Efield*>(conn->data())->scale_ = value;
}

double Efield::getScale(Eref eref)
{
    return static_cast<Efield*>(eref.data())->scale_;
}

double Efield::getPotential(Eref eref)
{
    return static_cast<Efield*>(eref.data())->pot_;    
}

void Efield::currentFunc(const Conn *conn, double value)
{
    unsigned int index = conn->targetIndex();
    Efield * instance = static_cast<Efield*>(conn->data());
    assert(instance!= NULL);
    assert((index < instance->distance_.size()));
    instance->innerPot_ += value / instance->distance_[index];
#ifndef NDEBUG
    cout << "Efield::currentFunc " << conn->target().id().path() << "<-" << conn->source().id().path() << ": Im = " << value << ", distance = " << instance->distance_[index] << ", potential = " << instance->innerPot_ << endl;
#endif
}

void Efield::reinitFunc(const Conn * conn, ProcInfo proc)
{
    static_cast<Efield*>(conn->data())->innerReinitFunc(conn, proc);    
}

void Efield::innerReinitFunc(const Conn * conn, ProcInfo proc)
{
    pot_ = 0.0;
    innerPot_ = 0.0;
    updateDistances(conn->target());    
}

void Efield::updateDistances(Eref eref)
{
    unsigned int n = eref.e->numTargets("currentDest");
    distance_.resize(n);
    Conn * comps_it = eref.e->targets("currentDest", eref.i);
    unsigned ii = 0;
    while (comps_it->good()){
        Eref target = comps_it->target();
        double x, y, z;
        get<double>(target, "x", x);
        get<double>(target, "y", y);
        get<double>(target, "z", z);
        distance_[ii] = sqrt((x_ - x) * (x_ - x) + (y_ - y) * (y_ - y) + (z_ - z) * (z_ - z));
        ii += 1;
        comps_it->increment();
    }
    delete comps_it;
}

void Efield::processFunc(const Conn* conn, ProcInfo proc)
{
    Efield * instance = static_cast<Efield*>(conn->data());
    instance->pot_ = instance->scale_ * instance->innerPot_;
    instance->innerPot_ = 0.0;
}

// 
// Efield.cpp ends here
