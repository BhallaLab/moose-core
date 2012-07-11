// LeakyIaF.cpp --- 
// 
// Filename: LeakyIaF.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Thu Jul  7 12:16:55 2011 (+0530)
// Version: 
// Last-Updated: Wed Jul 11 14:22:03 2012 (+0530)
//           By: subha
//     Update #: 174
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

#include <cfloat>

#include "header.h"
#include "../utility/numutil.h"

#include "LeakyIaF.h"

static SrcFinfo1< double >* spike() {
    static SrcFinfo1< double > spike(
            "spike", 
            "Sends out spike events");
    return &spike;
}

static SrcFinfo1< double >* VmOut() {
    static SrcFinfo1< double > VmOut(
            "VmOut", 
            "Sends out Vm");
    return &VmOut;
}

const Cinfo* LeakyIaF::initCinfo()
{
    //////////////////////////////////////////////////////////////
    // Field Definitions
    //////////////////////////////////////////////////////////////
		static ValueFinfo< LeakyIaF, double > Rm(
			"Rm",
			"Membrane resistance, inverse of leak-conductance.",
			&LeakyIaF::setRm,
			&LeakyIaF::getRm
		);

		static ValueFinfo< LeakyIaF, double > Cm(
			"Cm",
			"Membrane capacitance.",
			&LeakyIaF::setCm,
			&LeakyIaF::getCm
		);

		static ValueFinfo< LeakyIaF, double > Em(
			"Em",
			"Leak reversal potential",
			&LeakyIaF::setEm,
			&LeakyIaF::getEm
		);
                
		static ValueFinfo< LeakyIaF, double > initVm(
			"initVm",
			"Inital value of membrane potential",
			&LeakyIaF::setInitVm,
			&LeakyIaF::getInitVm
		);

		static ValueFinfo< LeakyIaF, double > Vm(
			"Vm",
			"Membrane potential",
			&LeakyIaF::setVm,
			&LeakyIaF::getVm
		);

		static ValueFinfo< LeakyIaF, double > Vthreshold(
			"Vthreshold",
			"firing threshold",
			&LeakyIaF::setVthreshold,
			&LeakyIaF::getVthreshold
		);

		static ValueFinfo< LeakyIaF, double > Vreset(
			"Vreset",
			"Reset potnetial after firing.",
			&LeakyIaF::setVreset,
			&LeakyIaF::getVreset
		);

		static ValueFinfo< LeakyIaF, double > refractoryPeriod(
			"refractoryPeriod",
			"Minimum time between successive spikes",
			&LeakyIaF::setRefractoryPeriod,
			&LeakyIaF::getRefractoryPeriod
		);
                
		static ValueFinfo< LeakyIaF, double > tSpike(
			"tSpike",
			"Time of the last spike",
			&LeakyIaF::setTspike,
			&LeakyIaF::getTspike
		);

		/*
		static ValueFinfo< LeakyIaF, unsigned int > numSynapses(
			"numSynapses",
			"Number of synapses on LeakyIaF",
			&LeakyIaF::setNumSynapses,
			&LeakyIaF::getNumSynapses
		);
		*/
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

                static DestFinfo injectDest(
                        "injectDest",
                        "Destination for current input.",
                        new OpFunc1<LeakyIaF, double>(&LeakyIaF::handleInject));
                                
		static DestFinfo process(
                        "process",
			"Handles process call",
			new ProcOpFunc< LeakyIaF >( &LeakyIaF::process ) );
                
		static DestFinfo reinit(
                        "reinit",
			"Handles reinit call",
			new ProcOpFunc< LeakyIaF >( &LeakyIaF::reinit ) );

		/*
		//////////////////////////////////////////////////////////////
		// FieldElementFinfo definition for Synapses
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< LeakyIaF, Synapse > synFinfo( "synapse",
			"Sets up field Elements for synapse",
			Synapse::initCinfo(),
			&LeakyIaF::getSynapse,
			&LeakyIaF::setNumSynapses,
			&LeakyIaF::getNumSynapses
		);
		*/
		//////////////////////////////////////////////////////////////
		// SharedFinfo Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};

                static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* leakyIaFFinfos[] = {
            &Cm,	// Value
            &Rm, // Value
            &Em, // Value
            &Vm,	// Value
            &initVm,
            &Vreset, // Value
            &Vthreshold,				// Value
            &refractoryPeriod,		// Value
            &tSpike,
            &proc,					// SharedFinfo
            spike(), 		// MsgSrc
            VmOut(),
            &injectDest,            
	};

	static Cinfo leakyIaFCinfo (
		"LeakyIaF",
		Neutral::initCinfo(),
		leakyIaFFinfos,
		sizeof( leakyIaFFinfos ) / sizeof ( Finfo* ),
		new Dinfo< LeakyIaF >()
	);

	return &leakyIaFCinfo;
}

static const Cinfo* intFireCinfo = LeakyIaF::initCinfo();

LeakyIaF::LeakyIaF():
        Rm_(1.0),
        Cm_(1.0),
        Em_(0.0),
        initVm_(0.0),
        Vm_(0.0),
        Vreset_(0.0),
        Vthreshold_(1.0),
        refractoryPeriod_(0.0),
        tSpike_(-DBL_MAX),
        sumInject_(0.0),
        dtRm_(0.0)
{
    ;
}

LeakyIaF::~LeakyIaF()
{
    ;
}

void LeakyIaF::setRm(double value)
{
    Rm_ = value;    
}

double LeakyIaF::getRm() const
{
    return Rm_;
}

void LeakyIaF::setCm(double value)
{
    Cm_ = value;    
}

double LeakyIaF::getCm() const
{
    return Cm_;
}


void LeakyIaF::setEm(double value)
{
    Em_ = value;    
}

double LeakyIaF::getEm() const
{
    return Em_;
}


void LeakyIaF::setInitVm(double value)
{
    initVm_ = value;    
}

double LeakyIaF::getInitVm() const
{
    return initVm_;
}


void LeakyIaF::setVm(double value)
{
    Vm_ = value;    
}

double LeakyIaF::getVm() const
{
    return Vm_;
}


void LeakyIaF::setVreset(double value)
{
    Vreset_ = value;    
}

double LeakyIaF::getVreset() const
{
    return Vreset_;
}

void LeakyIaF::setVthreshold(double value)
{
    Vthreshold_ = value;    
}

double LeakyIaF::getVthreshold() const
{
    return Vthreshold_;
}

void LeakyIaF::setRefractoryPeriod(double value)
{
    refractoryPeriod_ = value;    
}

double LeakyIaF::getRefractoryPeriod() const
{
    return refractoryPeriod_;
}

void LeakyIaF::setTspike(double value)
{
    tSpike_ = value;    
}

double LeakyIaF::getTspike() const
{
    return tSpike_;
}

void LeakyIaF::handleInject(double current)
{
    sumInject_ += current;
}

void LeakyIaF::process(const Eref & eref, ProcPtr proc)
{
    double time = proc->currTime;
    Vm_ += ((Em_ - Vm_) * dtRm_ + sumInject_)/Cm_; // Forward Euler
    sumInject_ = 0.0;
    VmOut()->send(eref, proc->threadIndexInGroup, Vm_);
    if ((Vm_ > Vthreshold_) && (time > tSpike_ + refractoryPeriod_)){
        tSpike_ = time;
        spike()->send(eref, proc->threadIndexInGroup, time);
        Vm_ = Vreset_;
    }
}

void LeakyIaF::reinit(const Eref& eref, ProcPtr proc)
{
    Vm_ = initVm_;
    sumInject_ = 0.0;
    tSpike_ = -DBL_MAX;
    dtRm_ = proc->dt / Rm_;
    VmOut()->send(eref, proc->threadIndexInGroup, Vm_);
}

///////////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS

void testLeakyIaF()
{
    const Cinfo * leakyIaFCinfo = LeakyIaF::initCinfo();
    Id leakyIaFId = Id::nextId();
    vector< DimInfo > dims;
    Element * elem = new Element(leakyIaFId, leakyIaFCinfo, "LeakyIaF", 
		dims, 1, true );
    assert(elem != 0);
    Eref eref(elem, 0);
    LeakyIaF* instance_ptr = reinterpret_cast<LeakyIaF*>(eref.data());
    ProcInfo proc;
    instance_ptr->setVm(-0.70);
    assert(almostEqual(instance_ptr->getVm(), -0.70));
    instance_ptr->setRm(1.0);
    assert(almostEqual(instance_ptr->getRm(), 1.0));
    instance_ptr->setCm(1.0);
    assert(almostEqual(instance_ptr->getCm(), 1.0));    
    instance_ptr->setInitVm(-0.65);
    assert(almostEqual(instance_ptr->getInitVm(), -0.65));    
    instance_ptr->setVreset(-0.65);
    assert(almostEqual(instance_ptr->getVreset(), -0.65));    
    instance_ptr->setVthreshold(-0.40);
    assert(almostEqual(instance_ptr->getVthreshold(), -0.40));    
    instance_ptr->setEm(-0.65);
    assert(almostEqual(instance_ptr->getEm(), -0.65));    
    double delta = 0.0;
    double Vm = 0.0;
    proc.dt = 0.002;
    instance_ptr->reinit(eref, &proc);
    assert(almostEqual(instance_ptr->getVm(), instance_ptr->getInitVm()));
    for (proc.currTime = 0.0; proc.currTime < 2.0; proc.currTime += proc.dt)
    {
        Vm = instance_ptr->getVm();
        double x = (instance_ptr->getEm() - instance_ptr->getVm()) / instance_ptr->getRm();
        instance_ptr->process(eref, &proc);
        delta += instance_ptr->getVm() - Vm - x;
    }
    assert(almostEqual(delta, 0.0));
    cout << "." << flush;
}

#endif //! DO_UNIT_TESTS

// 
// LeakyIaF.cpp ends here
