/*******************************************************************
 * File:            PulseGen.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-02-01 12:23:50
 ********************************************************************/
/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment,
 ** also known as GENESIS 3 base code.
 **           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU General Public License version 2
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#ifndef _PULSEGEN_CPP
#define _PULSEGEN_CPP
#include "moose.h"
#include <cmath>
#include "PulseGen.h"

const Cinfo* initPulseGenCinfo()
{
    static Finfo* processShared[] =
        {
            new DestFinfo( "process", Ftype1<ProcInfo>::global(),
                           RFCAST(&PulseGen::processFunc)),
            new DestFinfo( "reinit", Ftype1<ProcInfo>::global(),
                           RFCAST( &PulseGen::reinitFunc)),
        };
    static Finfo* process = new SharedFinfo( "process", processShared,
                                             sizeof(processShared)/sizeof(Finfo*));
    static Finfo* pulseGenFinfos[] = 
        {
            new ValueFinfo("firstLevel", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getFirstLevel),
                           RFCAST( &PulseGen::setFirstLevel)),
            new ValueFinfo("firstWidth", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getFirstWidth),
                           RFCAST( &PulseGen::setFirstWidth)),
            new ValueFinfo("firstDelay", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getFirstDelay),
                           RFCAST( &PulseGen::setFirstDelay)),
            
    
            new ValueFinfo("secondLevel", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getSecondLevel),
                           RFCAST( &PulseGen::setSecondLevel)),
            new ValueFinfo("secondWidth", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getSecondWidth),
                           RFCAST( &PulseGen::setSecondWidth)),
            new ValueFinfo("secondDelay", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getSecondDelay),
                           RFCAST( &PulseGen::setSecondDelay)),
            new ValueFinfo("baseLevel", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getBaseLevel),
                           RFCAST( &PulseGen::setBaseLevel)),
            new ValueFinfo("output", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getOutput),
                           &dummyFunc),            
            new ValueFinfo("trigTime", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getTrigTime),
                           RFCAST( &PulseGen::setTrigTime)),
            
            /** TRIGGER MODES: 	
             **         trig_mode = 0	free run
             **         trig_mode = 1	ext. trig
             **         trig_mode = 2	ext. gate
             **/
            new ValueFinfo("trigMode", ValueFtype1<int>::global(),
                           GFCAST( &PulseGen::getTrigMode),
                           RFCAST( &PulseGen::setTrigMode)),                                       
            new ValueFinfo("prevInput", ValueFtype1<int>::global(),
                           GFCAST( &PulseGen::getPreviousInput),
                           &dummyFunc),
      
            //////////////////////////////////////////////////////////////////
            // SharedFinfos
            //////////////////////////////////////////////////////////////////
            process,

            ///////////////////////////////////////////////////////
            // MsgSrc definitions
            ///////////////////////////////////////////////////////
            new SrcFinfo("outputSrc", Ftype1<double>::global()),
            
            //////////////////////////////////////////////////////////////////
            // MessageDestinations
            //////////////////////////////////////////////////////////////////
            new DestFinfo("input", Ftype1<double>::global(),
                          RFCAST(&PulseGen::inputFunc)),
            new DestFinfo("level", Ftype2<int, double>::global(),
                          RFCAST( &PulseGen::setPulseLevel)),
            new DestFinfo("width", Ftype2<int, double>::global(),
                          RFCAST( &PulseGen::setPulseWidth)),
            new DestFinfo("delay", Ftype2<int, double>::global(),
                          RFCAST( & PulseGen::setPulseDelay)),
        };

    static SchedInfo schedInfo[] = { { process, 0, 0 } };

    static Cinfo pulseGenCinfo("PulseGen",
                               "Subhasis Ray, 2007, NCBS",
                               "PulseGen: general purpose pulse generator",
                               initNeutralCinfo(),
                               pulseGenFinfos,
                               sizeof(pulseGenFinfos)/sizeof(Finfo*),
                               ValueFtype1<PulseGen>::global(),
                               schedInfo, 1);
    return &pulseGenCinfo;
}

static const Cinfo* pulseGenCinfo = initPulseGenCinfo();

static const Slot outputSlot = initPulseGenCinfo()->getSlot( "outputSrc");

PulseGen::PulseGen()
{
    firstLevel_ = 0.0;
    firstWidth_ = 0.0;
    firstDelay_ = 0.0;
    secondLevel_ = 0.0;
    secondWidth_ = 0.0;
    secondDelay_ = 0.0;
    
    output_ = 0.0;
    baseLevel_ = 0.0;
    trigTime_ = -1;
    trigMode_ = 0;
    prevInput_ = 0;
    secondPulse_ = true;    
}

void PulseGen::setFirstLevel(const Conn& c, double level)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT(obj != NULL, "PulseGen::setFirstLevel(const Conn&, double) - target data pointer is NULL.");
    obj->firstLevel_ = level;    
}

double PulseGen::getFirstLevel(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT(obj != NULL,"PulseGen::getFirstLevel(Eref ) - target data pointer is NULL.");
    return obj->firstLevel_;    
}
    
void PulseGen::setFirstWidth(const Conn& c, double width)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT(obj != NULL, "PulseGen::setFirstWidth(const Conn&, double) - target data pointer is NULL." );
    obj->firstWidth_ = width;    
}

double PulseGen::getFirstWidth(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getFirstWidth(Eref ) - target data pointer is NULL." );
    return obj->firstWidth_;    
}
void PulseGen::setFirstDelay(const Conn & c, double delay)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setFirstDelay(const Conn&, double) - target data pointer is NULL.");
    obj->firstDelay_ = delay;    
}
double PulseGen::getFirstDelay(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getFirstDelay(Eref ) - target data pointer is NULL.");    
    return obj->firstDelay_;
}
    
void PulseGen::setSecondLevel(const Conn& c, double level)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setSecondLevel(const Conn&, double) - target data pointer is NULL.");
    obj->secondLevel_ = level;
}
double PulseGen::getSecondLevel(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getSecondLevel(Eref ) - target data pointer is NULL.");
    return obj->secondLevel_;
}
void PulseGen::setSecondWidth(const Conn& c, double width)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setFirstWidth(const Conn&, double) - target data pointer is NULL.");
    obj->secondWidth_ = width;
}
double PulseGen::getSecondWidth(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getSecondWidth(Eref ) - target data pointer is NULL.");    
    return obj->secondWidth_;
}
void PulseGen::setSecondDelay(const Conn& c, double delay)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setSecondDelay(const Conn&, double) - target data pointer is NULL.");
    obj->secondDelay_ = delay;
}
double PulseGen::getSecondDelay(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getSecondDelay(Eref ) - target data pointer is NULL.");
    return obj->secondDelay_;    
}

void PulseGen::setBaseLevel(const Conn& c, double level)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setBaseLevel(const Conn&, double) - target data pointer is NULL.");
    obj->baseLevel_ = level;    
}
double PulseGen::getBaseLevel(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getFirstDelay(Eref ) - target data pointer is NULL.");
    return obj->baseLevel_;    
}
void PulseGen::setTrigMode(const Conn& c, int mode)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setTrigMode(const Conn&, double) - target data pointer is NULL.");
    obj->trigMode_ = mode;    
}
int PulseGen::getTrigMode(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getTrigMode(Eref ) - target data pointer is NULL.");
    return obj->trigMode_;
}
double PulseGen::getOutput(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getOutput(Eref ) - target data pointer is NULL.");
    return obj->output_;
}

/**
   trigTime is supposed to be an internal state variable according to
   GENESIS documentation. But in GENESIS it is available for
   manipulation by the user and there are scripts out there which use
   this.

   One particular case one changes this field is in association with
   the generation of single pulse. In trigMode = 1 (EXT_TRIG), if
   there is 0 input to the PulseGen object, and trigTime >= 0, then a
   pulse is generated at firstDelay time after the trigTime, i.e. the
   pulse starts at time = (trigTime + firstDelay).

   But note that the reset method sets the trigTime to -1, so if you
   want a single pulse, you need to set trigTime after the reset.
*/
void PulseGen::setTrigTime(const Conn& conn, double trigTime)
{
    PulseGen* obj = static_cast<PulseGen*> (conn.data());
    ASSERT( obj != NULL, "PulseGen::setTrigTime(const Conn&, double) - target data pointer is NULL.");
    obj->trigTime_ = trigTime;    
}

double PulseGen::getTrigTime(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getTrigTime(Eref ) - target data pointer is NULL." );    
    return obj->trigTime_;
}

int PulseGen::getPreviousInput(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getPreviousInput(Eref ) - target data pointer is NULL.");
    return obj->prevInput_;    
}

void PulseGen::setPulseLevel(const Conn& c, int index, double level)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setPulseLevel(const Conn&, int, double) - target data pointer is NULL.");
    index == 0? obj->firstLevel_ = level: obj->secondLevel_ = level;
}

void PulseGen::setPulseWidth(const Conn& c, int index, double width)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setPulseWidth(const Conn&, int, double) - target data pointer is NULL.");
    index == 0? obj->firstWidth_ = width: obj->secondWidth_ = width;
}
    
void PulseGen::setPulseDelay(const Conn& c, int index, double delay)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::setPulseDelay(const Conn&, int, double) - target data pointer is NULL.");
    index == 0? obj->firstDelay_ = delay: obj->secondDelay_ = delay;
}

void PulseGen::inputFunc(const Conn& c, int value)
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::inputFunc(const Conn&, int) - target data pointer is NULL.");
    obj->input_ = value;
}

//////////////////////////////////////////////////////////////////
// Message dest functions.
//////////////////////////////////////////////////////////////////

void PulseGen::processFunc( const Conn& c, ProcInfo p )
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::processFunc(const Conn&, ProcInfo) - target data pointer is NULL.");
    obj->innerProcessFunc(c, p);
}


void PulseGen::reinitFunc( const Conn& c, ProcInfo p )
{
    PulseGen* obj = static_cast<PulseGen*> (c.data());
    ASSERT( obj != NULL, "PulseGen::reinitFunc(const Conn&, ProcInfo) - target data pointer is NULL.");
    obj->trigTime_ = -1;
    obj->prevInput_ = 0;
    obj->output_ = obj->baseLevel_;
    obj->input_ = 0;    
}
/**
   This has been adapted from the original genesis code written by
   M. Nelson
*/
void PulseGen::innerProcessFunc(const Conn& c, ProcInfo p)
{
    double currentTime = p->currTime_;
    double period = 0.0;
    double phase = 0.0;
    
    if ( firstWidth_ > secondDelay_ + secondWidth_ )
    {
        period = firstDelay_ + firstWidth_;
    }
    else
    {
        period = firstDelay_ + secondDelay_ + secondWidth_;
    }
    switch ( trigMode_)
    {
        case PulseGen::FREE_RUN :
            phase = fmod(currentTime,period);
            break;
        case PulseGen::EXT_TRIG :
            if ( input_ == 0 )
            {
                if ( trigTime_ < 0 )
                {
                    phase = period;
                }
                else
                {
                    phase = currentTime - trigTime_;
                }
            }
            else
            {
                if ( prevInput_ == 0 )
                {
                    trigTime_ = currentTime;
                }
                phase = currentTime - trigTime_;
            }
            prevInput_ = input_;
    
            break;
        case PulseGen::EXT_GATE :
            if(input_ == 0)
            {
                phase = period;		/* output = baselevel */
            }
            else
            {				/* gate high */ 
                if(prevInput_ == 0)
                {	/* low -> high */
                    trigTime_ = currentTime;
                }
                phase = fmod(currentTime - trigTime_, period);
            }
            prevInput_ = input_;
            break;
        default:
            cerr << "ERROR: PulseGen::innerProcessFunc( const Conn& , ProcInfo ) - invalid triggerMode - " << trigMode_ << endl;
    }
    if ( phase < firstDelay_  || phase >= period )
        output_ = baseLevel_;
    else if (phase < firstDelay_ + firstWidth_)
        output_ = firstLevel_;
    else if(phase < firstDelay_ + secondDelay_)
        output_ = baseLevel_;
    else 
        output_ = secondLevel_;
    send1<double>( c.target(), outputSlot, output_);    
}
    
                     
#endif
