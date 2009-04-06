// IntFire.cpp --- 
// 
// Filename: IntFire.cpp
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Thu Mar 19 22:57:33 2009 (+0530)
// Version: 
// Last-Updated: Tue Apr  7 04:06:25 2009 (+0530)
//           By: subhasis ray
//     Update #: 206
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// This is a Leaky integrate and fire model. It simulates
// the ODE:
//
// tau * dVm/dt = -Vm + RI(t) for Vm < Vt
//
// When Vm = Vt it will send a
// spike event and the Vm is reset to Vr.
// 

// Change log:
//
// 2009-03-19: Subhasis created inital version from Upi's test code in
// MSG branch.
// 
// 2009-03-25: Subhasis combined SpikeGen and Compartment code to give
// one single inteface for Integrate and Fire neuron.
// 
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// Code:

#include "moose.h"

#include "IntFire.h"

const double IntFire::EPSILON = 1e-15;
const Cinfo* initIntFireCinfo()
{
    static Finfo* processShared[] = {
        new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                  RFCAST( &IntFire::processFunc )),
        new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                       RFCAST( &IntFire::reinitFunc )),
    };
    static Finfo* process = new SharedFinfo( "process", processShared,
                                             sizeof(processShared) / sizeof(Finfo*),
			"This is a shared message to receive Process messages from the scheduler objects. "
			"The first entry is a MsgDest for the Process operation. It has a single argument,ProcInfo, "
			"which holds lots of information about current time, thread, dt and so on. "
			"The second entry is a MsgDest for the Reinit operation. "
			"It also uses ProcInfo. " );
    // TODO: Do we need initShared ?
    // TODO: channelShared should be able to allow the SynChan to be
    // connected as synapse - or should I implement a separate kind of
    // synapse for IF?
    static Finfo* channelShared[] = {
        new DestFinfo("channel", Ftype2<double, double>::global(),
                      RFCAST(&IntFire::channelFunc)),
        new SrcFinfo("Vm", Ftype1<double>::global()),
    };
    static Finfo* intFireFinfos[] = {
        new ValueFinfo("Vt", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getVt),
                       RFCAST(&IntFire::setVt),
                       "Threshold potential. The membrane potential is reset to Vr and a spike"
                       " event at this point of time is sent to all post-synaptic neurons."),
        new ValueFinfo("Vr", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getVr),
                       RFCAST(&IntFire::setVr),
                       "Reset potential. Membrane potential is reset to Vr whenever it reaches"
                       " Vt."),
        new ValueFinfo("Rm", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getRm),
                       RFCAST(&IntFire::setRm),
                       "Leakage resistance"),
        new ValueFinfo("Cm", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getCm),
                       RFCAST(&IntFire::setCm),
                       "Capacitance of the membrane."),
        new ValueFinfo("Vm", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getVm),
                       RFCAST(&IntFire::setVm),
                       "The membrane potential."),
        new ValueFinfo("tau", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getTau),
                       &dummyFunc,
                       "Membrane time constant. This is calculated as tau = Rm * Cm."),
        new ValueFinfo("Em", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getEm),
                       RFCAST(&IntFire::setEm),
                       "Reversal potential for the leak current."),
        new ValueFinfo("refractT", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getRefractT),
                       RFCAST(&IntFire::setRefractT),
                       "Absolute refractory period for this neuron."),
        new ValueFinfo("initVm", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getInitVm),
                       RFCAST(&IntFire::setInitVm),
                       "Initial membrane potential."),
        new ValueFinfo("inject", ValueFtype1<double>::global(),
                       GFCAST(&IntFire::getInject),
                       RFCAST(&IntFire::setInject),
                       "This is fixed injection via a field value. For time-varying injection, use injectDest."),
        ///////////////////////////////
        // MsgSrc definition
        ///////////////////////////////
        new SrcFinfo("VmSrc", Ftype1<double>::global()),
        new SrcFinfo("eventSrc", Ftype1<double>::global(),
                     "Sends out a trigger for an event whenever Vm > Vt and current time is:"
                     " not within refractT of time to last spike."),
        ///////////////////////////////
        // MsgSrc definition
        ///////////////////////////////
        new DestFinfo( "injectDest", Ftype1< double >::global(),
                       RFCAST( &IntFire::injectDestFunc ), 
                       "The injectMsg corresponds to the INJECT message in the GENESIS compartment. "
                       "It does different things from the inject field, and perhaps should just be merged in. "
                       "In particular, it needs to be updated every dt to have an effect. " ),

        ///////////////////////////////
        // SharedFinfos
        ///////////////////////////////
        process,
        //        init,
        new SharedFinfo("channel", channelShared,
                        sizeof(channelShared) / sizeof(Finfo*),
                        "This is a shared message from compartment to channels. The first entry"
                        " is a MsgDest for the info coming from the channel. It expects Gk and"
                        " Ek from channel as args. The second entry is a MsgSrc sending Vm."),
    };
    
    static SchedInfo schedInfo[] = { {process, 0, 0 },};

    static string doc[] = {
        "Name", "IntFire",
        "Author", "Subhasis Ray",
        "Description", "Leaky integrate and fire neuron with absolute refractory period."
        " This class is an implementation of leaky integrate and fire neuron"
        " obeying the differential equation:\n"
        " tau * dVm / dt = (Em - Vm) + Rm * inject\n"
        " The frequency of spiking is given by:\n"
        " f = 1 / (refractT + tau * ln((Rm * inject + Em - Vr) / (Rm * inject + Em - Vt)))"
    };
    static Cinfo intFireCinfo(
            doc,
            sizeof(doc) / sizeof(string),
            initNeutralCinfo(),
            intFireFinfos,
            sizeof(intFireFinfos) / sizeof(Finfo*),
            ValueFtype1<IntFire>::global(),
            schedInfo, 1);

    return &intFireCinfo;
}

static const Cinfo* intFireCinfo = initIntFireCinfo();

static const Slot channelSlot = initIntFireCinfo()->getSlot("channel.Vm");
static const Slot VmSlot = initIntFireCinfo()->getSlot("VmSrc");
static const Slot eventSlot = initIntFireCinfo()->getSlot("eventSrc");

IntFire::IntFire():
        initVm_(0.0),
        Vt_(1.0),
        Vr_(0.0),
        Vm_(0.0),
        Rm_(1.0),
        Cm_(1.0),
        refractT_(0.0),
        Em_(0.0),
        A_(0.0),
        B_(0.0),
        lastEvent_(-DBL_MAX),
        inject_(0.0),
        sumInject_(0.0),
        Im_(0.0)
        {
            //do nothing
        }

double IntFire::getVm(Eref e)
{
    return static_cast<IntFire*>(e.data())->Vm_;
}

void IntFire::setVm(const Conn* conn, double Vm)
{
    static_cast<IntFire*>(conn->data())->Vm_ = Vm;
}

double IntFire::getVt(Eref e)
{
    return static_cast<IntFire*>(e.data())->Vt_;
}

void IntFire::setVt(const Conn* conn, double Vt)
{
    static_cast<IntFire*>(conn->data())->Vt_ = Vt;
}

double IntFire::getVr(Eref e)
{
    return static_cast<IntFire*>(e.data())->Vr_;
}

void IntFire::setVr(const Conn* conn, double Vr)
{
    static_cast<IntFire*>(conn->data())->Vr_ = Vr;
}

double IntFire::getCm(Eref e)
{
    return static_cast<IntFire*>(e.data())->Cm_;
}

void IntFire::setCm(const Conn* conn, double Cm)
{
    static_cast<IntFire*>(conn->data())->Cm_ = Cm;
}

double IntFire::getRm(Eref e)
{
    return static_cast<IntFire*>(e.data())->Rm_;
}

void IntFire::setRm(const Conn* conn, double Rm)
{
    static_cast<IntFire*>(conn->data())->Rm_ = Rm;
}

double IntFire::getEm(Eref e)
{
    return static_cast<IntFire*>(e.data())->Em_;
}

void IntFire::setEm(const Conn* conn, double Em)
{
    static_cast<IntFire*>(conn->data())->Em_ = Em;
}

double IntFire::getInitVm(Eref e)
{
    return static_cast<IntFire*>(e.data())->initVm_;
}

void IntFire::setRefractT(const Conn* conn, double refractT)
{
    static_cast<IntFire*>(conn->data())->refractT_ = refractT;
}
double IntFire::getRefractT(Eref eref)
{
    return static_cast<IntFire*>(eref.data())->refractT_;
}

double IntFire::getInject(Eref eref)
{
    return static_cast<IntFire*>(eref.data())->inject_;
}

void IntFire::setInject(const Conn* conn, double inject)
{
    static_cast<IntFire*>(conn->data())->inject_ = inject;
}

void IntFire::setInitVm(const Conn* conn, double initVm)
{
    static_cast<IntFire*>(conn->data())->initVm_ = initVm;
}

double IntFire::getTau(Eref e)
{
    IntFire* instance = static_cast<IntFire*>(e.data());
    return instance->Rm_ * instance->Cm_;
}

void IntFire::innerProcessFunc(Eref eref, ProcInfo proc)
{
    double t = proc->currTime_;
    if (t > lastEvent_ + refractT_){
        A_ += inject_ + sumInject_ + Em_ / Rm_; 
        if ( B_ > EPSILON ) {
            double x = exp( -B_ * proc->dt_ / Cm_ );
            Vm_ = Vm_ * x + ( A_ / B_ )  * ( 1.0 - x );
        } else {
            Vm_ += ( A_ - Vm_ * B_ ) * proc->dt_ / Cm_;
        }
        A_ = 0.0;
        B_ = 1.0 / Rm_;
        Im_ = 0.0;
        sumInject_ = 0.0;
    }
    // Send out the channel messages
    send1< double >( eref, channelSlot, Vm_ );
    // Send out the message to any SpikeGens.
    send1< double >( eref, VmSlot, Vm_ );
    if (Vm_ > Vt_ && t > lastEvent_ + refractT_){
        send1<double>(eref, eventSlot, t);
        lastEvent_ = t;
        Vm_ = Vr_;        
    }
    // Reset Vm if it has crossed threshold. This must happen after
    // sending out the VmSrc message because if we connect it to a
    // SpikeGen object, that must also see Vm_ > SpikeGen.Vt_
    // But this is a bad design as we have to make sure IntFire.Vt =
    // SpikeGen.Vt for the model to be valid.
    // Perhaps I should just make IntFire with a built-in SpikeGen.
    // One argument for external SpikeGen is that we can control
    // presynaptic refractory period. On the otherhand we can
    // create a diferent kind of synapse with refractory period as a
    // property of the synapse.
}

void IntFire::processFunc(const Conn* conn, ProcInfo proc)
{
    static_cast<IntFire*>(conn->data())->innerProcessFunc(conn->target(), proc);    
}

void IntFire::channelFunc(const Conn* conn, double Gk, double Ek)
{
    IntFire* instance = static_cast<IntFire*>(conn->data());
    instance->A_ += Gk * Ek;
    instance->B_ += Gk;
}

void IntFire::injectDestFunc(const Conn* conn, double I)
{
    IntFire* instance = static_cast<IntFire*>(conn->data());
    instance->sumInject_ += I;
    instance->Im_ += I;
}

void IntFire::innerReinitFunc(Eref eref, ProcInfo proc)
{
    Vm_ = initVm_;
    A_ = 0.0;
    B_ = 1.0 / Rm_;
    Im_ = 0.0;
    sumInject_ = 0.0;
    // Send the Vm over to the channels at reset.
    send1< double >( eref, channelSlot, Vm_ );
    // Send the Vm over to the SpikeGen
    send1< double >( eref, VmSlot, Vm_ );
    
}

void IntFire::reinitFunc(const Conn* conn, ProcInfo proc)
{
    static_cast<IntFire*>(conn->data())->innerReinitFunc(conn->target(), proc);
}

#ifdef DO_UNIT_TESTS

#include "../element/Neutral.h"

void testIntegrateFire()
{
    cout << "\nTesting IntFire" << flush;
    Element* container = Neutral::create("Neutral", "container", Element::root()->id(),
                                 Id::scratchId());
    Element* c0 = Neutral::create("IntFire", "c0", container->id(), Id::scratchId());
    ASSERT(c0 != 0, "creating IntFire");
    ProcInfoBase p;
    SetConn c(c0, 0);
    p.dt_ = 1e-3;
    IntFire::setInject(&c, 1.0);
    IntFire::setRm(&c, 2e-4);
    IntFire::setCm(&c, 0.5e-6);
    IntFire::setEm(&c, 0.0);
}

#endif //DO_UNIT_TESTS
// 
// IntFire.cpp ends here
