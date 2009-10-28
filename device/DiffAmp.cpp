// DiffAmp.cpp --- 
// 
// Filename: DiffAmp.cpp
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Mon Dec 29 16:01:22 2008 (+0530)
// Version: 
// Last-Updated: Wed Oct 28 09:58:36 2009 (+0530)
//           By: subhasis ray
//     Update #: 184
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// This implements an equivalent of the diffamp object in GENESIS.
// 
// 
// 

// Change log:
// 2008-12-30 16:21:19 (+0530) - Initial version.
// 
// 
/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment,
 ** also known as GENESIS 3 base code.
 **           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU General Public License version 2
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

// Code:
#include "DiffAmp.h"

const Cinfo* initDiffAmpCinfo()
{
    static Finfo* processShared[] = {
	new DestFinfo( "process", Ftype1< ProcInfo>::global(),
		      RFCAST(&DiffAmp::processFunc)),
	new DestFinfo("reinit", Ftype1< ProcInfo >::global(),
		      RFCAST(&DiffAmp::reinitFunc)),
    };
    static Finfo* process = new SharedFinfo("process", processShared, sizeof(processShared) / sizeof(Finfo*),
					    "This is a shared message to receive process messages from the"
                                            " scheduler objects.");
    
	static Finfo* plusShared[] =
	{
	    new SrcFinfo(
	        "request",
	        Ftype0::global(),
	        "Sends out the request. Issued from the process call." ),
	    new DestFinfo( "handle",
	        Ftype1< double >::global(),
            RFCAST( &DiffAmp::plusFunc ),
            "Handle the returned value." ),
    };
    
	static Finfo* minusShared[] =
	{
	    new SrcFinfo(
	        "request",
	        Ftype0::global(),
	        "Sends out the request. Issued from the process call." ),
	    new DestFinfo( "handle",
	        Ftype1< double >::global(),
            RFCAST( &DiffAmp::minusFunc ),
            "Handle the returned value." ),
    };
    
    static Finfo* diffAmpFinfos[] = {
	new ValueFinfo( "gain", ValueFtype1< double >::global(),
                        GFCAST(&DiffAmp::getGain),
                        RFCAST(&DiffAmp::setGain),
                        "Gain of the amplifier. The output of the amplifier is the difference"
                        " between the totals in plus and minus inputs multiplied by the"
                        " gain. Defaults to 1" ),
	new ValueFinfo( "saturation", ValueFtype1< double >::global(),
                        GFCAST(&DiffAmp::getSaturation),
                        RFCAST(&DiffAmp::setSaturation),
                        "Saturation is the bound on the output. If output goes beyond the +/-"
                        "saturation range, it is truncated to the closer of +saturation and"
                        " -saturation. Defaults to the maximum double precision floating point"
                        " number representable on the system." ),
	new ValueFinfo( "plus", ValueFtype1< double >::global(),
                        GFCAST(&DiffAmp::getPlus),
                        RFCAST(&DiffAmp::plusFunc),
                        "Total input to the positive terminal of the amplifier." ),
	new ValueFinfo( "minus", ValueFtype1< double >::global(),
                        GFCAST(&DiffAmp::getMinus),
                        RFCAST(&DiffAmp::minusFunc),
                        "Total input to the negative terminal of the amplifier."
 ),
	new ValueFinfo( "output", ValueFtype1< double >::global(),
                        GFCAST(&DiffAmp::getOutput),
                        RFCAST(&dummyFunc),
                        "Output of the amplifier, i.e. gain * (plus - minus)." ),
	process,
	    
	new SrcFinfo( "outputSrc", Ftype1< double >::global(),
                      "Sends the output of this difference amplifier."),
	new DestFinfo( "gainDest", Ftype1< double >::global(),
                       RFCAST(&DiffAmp::setGain),
                       "This is a destination message to control gain dynamically."),
	new SharedFinfo( "plusMsg",
                      plusShared, 
                      sizeof( plusShared ) / sizeof( Finfo* ),
                      "This shared message requests for the plus value, and handles "
                      "the returned with a callback function." ),
	new SharedFinfo( "minusMsg",
                      minusShared, 
                      sizeof( minusShared ) / sizeof( Finfo* ),
                      "This shared message requests for the minus value, and handles "
                      "the returned with a callback function." ),
	//~ new DestFinfo( "plusDest", Ftype1< double >::global(),
                       //~ RFCAST(&DiffAmp::plusFunc),
                       //~ "Positive input terminal of the amplifier. All the messages connected"
                       //~ " here are summed up to get total positive input."),
	//~ new DestFinfo( "minusDest", Ftype1< double >::global(),
                       //~ RFCAST(&DiffAmp::minusFunc),
                       //~ "Negative input terminal of the amplifier. All the messages connected"
                       //~ " here are summed up to get total positive input."),
    };
    
    static SchedInfo schedInfo[] = {{ process, 0, 0 }};
    static string doc[] = {
        "Name", "DiffAmp",
        "Author", "Subhasis Ray, 2008, NCBS",
        "Description", "A difference amplifier. "
        "Output is the difference between the total plus inputs and the total "
        "minus inputs multiplied by gain. Gain can be set statically as a field"
        " or can be a destination message and thus dynamically determined by the"
        " output of another object. Same as GENESIS diffamp object."
    };
    
    static Cinfo diffAmpCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),
            initNeutralCinfo(),
            diffAmpFinfos,
            sizeof( diffAmpFinfos ) / sizeof( Finfo* ),
            ValueFtype1< DiffAmp >::global(),
            schedInfo, 1 );
    
    return &diffAmpCinfo;    
}

static const Cinfo* diffAmpCinfo = initDiffAmpCinfo();

static const Slot outputSlot = initDiffAmpCinfo()->getSlot("outputSrc");
static const Slot plusSlot = initDiffAmpCinfo()->getSlot("plusMsg.request");
static const Slot minusSlot = initDiffAmpCinfo()->getSlot("minusMsg.request");

DiffAmp::DiffAmp():gain_(1.0), saturation_(DBL_MAX), plus_(0), minus_(0), output_(0)
{
}

void DiffAmp::plusFunc(const Conn* conn, double input)
{
    DiffAmp* instance = static_cast < DiffAmp* >(conn->data());
#ifndef NDEBUG
    cout << "PLUS " << conn->target().id().path() << " : " << instance->plus_ << ", " << instance->minus_<< " : " << input << endl;
#endif

    instance->plus_ += input;
}

void DiffAmp::minusFunc(const Conn* conn, double input)
{
    DiffAmp* instance = static_cast < DiffAmp* >(conn->data());
#ifndef NDEBUG
    cout << "MINUS " << conn->target().id().path() << " : " << instance->plus_ << ", " << instance->minus_<< " : " << input <<endl;
#endif
    instance->minus_ += input;
}

void DiffAmp::setGain(const Conn* conn, double gain)
{
    DiffAmp* instance = static_cast< DiffAmp* >(conn->data());
    instance->gain_ = gain;
}

void DiffAmp::setSaturation(const Conn* conn, double saturation)
{
    DiffAmp* instance = static_cast< DiffAmp* >(conn->data());
    instance->saturation_ = saturation;
}

double DiffAmp::getGain(Eref e)
{
    DiffAmp* instance = static_cast< DiffAmp* >(e.data());
    return instance->gain_;
}

double DiffAmp::getSaturation(Eref e)
{
    DiffAmp* instance = static_cast< DiffAmp* >(e.data());
    return instance->saturation_;
}

double DiffAmp::getPlus(Eref e)
{
    DiffAmp* instance = static_cast< DiffAmp* >(e.data());
    return instance->plus_;
}

double DiffAmp::getMinus(Eref e)
{
    DiffAmp* instance = static_cast< DiffAmp* >(e.data());
    return instance->minus_;
}

double DiffAmp::getOutput(Eref e)
{
    DiffAmp* instance = static_cast< DiffAmp* >(e.data());
    return instance->output_;
}

void DiffAmp::processFunc(const Conn* conn, ProcInfo p)
{
    DiffAmp* instance = static_cast< DiffAmp* >(conn->data());
    
	// Resetting plus and minus fields to 0.0
    instance->plus_ = 0.0;
    instance->minus_ = 0.0;
    
    /*
     * Request for + and - values.
     * All incoming values from callbacks will be accumulated.
     */
    send0(conn->target(), plusSlot);
    send0(conn->target(), minusSlot);
    
    double output = instance->gain_ * (instance->plus_ - instance->minus_);
#ifndef NDEBUG
    cout << conn->target().id().path() << ": plus = " << instance->plus_ << " :minus = " << instance->minus_ << " :output = " << output << " :gain = " << instance->gain_ << " :saturation = " << instance->saturation_ << endl;
#endif
    if ( output > instance->saturation_ ) {
	output = instance->saturation_;
    }
    if ( output < -instance->saturation_ ) {
	output = -instance->saturation_;
    }    
    instance->output_ = output;
    send1<double>(conn->target(), outputSlot, output);
}

void DiffAmp::reinitFunc(const Conn* conn, ProcInfo p)
{
    DiffAmp* instance = static_cast< DiffAmp* >(conn->data());
    instance->output_ = 0.0;
    instance->plus_ = 0.0;
    instance->minus_ = 0.0;
}

// 
// DiffAmp.cpp ends here
