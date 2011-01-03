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
 **           copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
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
                           RFCAST( &PulseGen::setFirstLevel),
                           "Amplitude of the first pulse output."),
            new ValueFinfo("firstWidth", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getFirstWidth),
                           RFCAST( &PulseGen::setFirstWidth),
                           "Duration of the first pulse output."),
            new ValueFinfo("firstDelay", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getFirstDelay),
                           RFCAST( &PulseGen::setFirstDelay),
                           "Delay to first pulse output. In FREE RUN mode, this is the interval" \
                           "from the end of the previous pulse till the start of the first pulse." \
                           "In case of TRIGGERED mode, this is how long it takes to start the "
                           "triggered pulse after the start of the triggering pulse."

                           "NOTE: If another triggering pulse comes before the triggered pulse "
                           "was generated, the triggered pulse will be lost as the last trigger "
                           "time is reset to the latest one and the pulsegen never reaches a state "
                           "where the time interval since the last trigger input never crosses "
                           "firstDelay."),
            
    
            new ValueFinfo("secondLevel", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getSecondLevel),
                           RFCAST( &PulseGen::setSecondLevel),
                           "Amplitude of the second pulse. Default value: 0"),
            new ValueFinfo("secondWidth", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getSecondWidth),
                           RFCAST( &PulseGen::setSecondWidth),
                           "Duration of the second pulse. Default value: 0"),
            new ValueFinfo("secondDelay", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getSecondDelay),
                           RFCAST( &PulseGen::setSecondDelay),
                           "Time interval between first and second pulse. If 0, there will be no "
                           "second pulse. Default value: 0"),
            new ValueFinfo("baseLevel", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getBaseLevel),
                           RFCAST( &PulseGen::setBaseLevel),
                           "Baseline output (when no pulse is being generated). Default value: 0."),
            new ValueFinfo("output", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getOutput),
                           &dummyFunc,
                           "Pulse output."),            
            new ValueFinfo("trigTime", ValueFtype1<double>::global(),
                           GFCAST( &PulseGen::getTrigTime),
                           RFCAST( &PulseGen::setTrigTime),
                           "Time since last time the input switched from 0 to non-zero. This is "
                           "supposed to be an internal variable, but old GENESIS scripts use this "
                           "field to generate a single pulse. If you set trigTime to a positive "
                           "value after reset, a single output will be generated at firstDelay "
                           "time."),
            new ValueFinfo("count", ValueFtype1<int>::global(),
                           GFCAST( &PulseGen::getCount),
                           RFCAST( &PulseGen::setCount),
                           "Number of pulses in each period. Default is 2, where the second is "
                           "same as baseline at delay 0 with width 0."),
            new LookupFinfo("width", LookupFtype<double, int>::global(),
                            GFCAST( &PulseGen::getWidth),
                            RFCAST( &PulseGen::setWidth),
                            "Width of i-th pulse in a period. Before you can set this, make sure "
                            "you have >= i pulses using pulseCount field."),
            new LookupFinfo("delay", LookupFtype<double, int>::global(),
                            GFCAST( &PulseGen::getDelay),
                            RFCAST( &PulseGen::setDelay),
                            "Delay of i-th pulse in a period. Before you can set this, make sure"
                            " you have >= i pulses using pulseCount field."),
            new LookupFinfo("level", LookupFtype<double, int>::global(),
                            GFCAST( &PulseGen::getLevel),
                            RFCAST( &PulseGen::setLevel),
                            "Amplitude of i-th pulse in a period. Before you can set this, make"
                            " sure you have >= i pulses using pulseCount field."),
            
                           
            new ValueFinfo("trigMode", ValueFtype1<int>::global(),
                           GFCAST( &PulseGen::getTrigMode),
                           RFCAST( &PulseGen::setTrigMode),
						   "TRIGGER MODES: \n"	
							"trig_mode = 0	free run \n"
							"trig_mode = 1	ext. trig \n"
							"trig_mode = 2	ext. gate" ),                                       
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
            new DestFinfo("levelDest", Ftype2<int, double>::global(),
                          RFCAST( &PulseGen::setLevelFunc)),
            new DestFinfo("widthDest", Ftype2<int, double>::global(),
                          RFCAST( &PulseGen::setWidthFunc)),
            new DestFinfo("delayDest", Ftype2<int, double>::global(),
                          RFCAST( & PulseGen::setDelayFunc)),
        };

    static SchedInfo schedInfo[] = { { process, 0, 0 } };


	static string doc[] =
	{
		"Name", "PulseGen",
		"Author", "Subhasis Ray, 2007, NCBS",
		"Description", "PulseGen: general purpose pulse generator. This can generate any "
                "number of pulses with specified level and duration.",
	};
    static Cinfo pulseGenCinfo(
                               doc,
			       sizeof( doc ) / sizeof( string ),                               
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
    level_.reserve(2);
    width_.reserve(2);
    delay_.reserve(2);
    level_.resize(2);
    width_.resize(2);
    delay_.resize(2);
    level_.assign(2, 0.0);
    delay_.assign(2, 0.0);
    width_.assign(2, 0.0);
    output_ = 0.0;
    baseLevel_ = 0.0;
    trigTime_ = -1;
    trigMode_ = 0;
    prevInput_ = 0;
}

void PulseGen::setFirstLevel(const Conn* c, double level)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT(obj != NULL, "PulseGen::setFirstLevel(const Conn*, double) - target data pointer is NULL.");
    obj->level_[0] = level;    
}

double PulseGen::getFirstLevel(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT(obj != NULL,"PulseGen::getFirstLevel(Eref ) - target data pointer is NULL.");
    return obj->level_[0];    
}
    
void PulseGen::setFirstWidth(const Conn* c, double width)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT(obj != NULL, "PulseGen::setFirstWidth(const Conn*, double) - target data pointer is NULL." );
    obj->width_[0] = width;    
}

double PulseGen::getFirstWidth(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getFirstWidth(Eref ) - target data pointer is NULL." );
    return obj->width_[0];    
}
void PulseGen::setFirstDelay(const Conn* c, double delay)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setFirstDelay(const Conn*, double) - target data pointer is NULL.");
    obj->delay_[0] = delay;    
}
double PulseGen::getFirstDelay(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getFirstDelay(Eref ) - target data pointer is NULL.");    
    return obj->delay_[0];
}
    
void PulseGen::setSecondLevel(const Conn* c, double level)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setSecondLevel(const Conn*, double) - target data pointer is NULL.");
    if (obj->level_.size() >= 2){
        obj->level_[obj->level_.size() - 1] = level;
    }
}
double PulseGen::getSecondLevel(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getSecondLevel(Eref ) - target data pointer is NULL.");
    if (obj->level_.size() >= 2){
        return obj->level_[obj->level_.size() - 1];
    } else {
        return 0.0;
    }
}
void PulseGen::setSecondWidth(const Conn* c, double width)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setFirstWidth(const Conn*, double) - target data pointer is NULL.");
    if (obj->width_.size() >= 2){
        obj->width_[obj->width_.size()-1] = width;
    }
}
double PulseGen::getSecondWidth(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getSecondWidth(Eref ) - target data pointer is NULL.");
    if (obj->width_.size() >= 2){
        return obj->width_[obj->width_.size()-1];
    } else {
        return 0.0;
    }
}
void PulseGen::setSecondDelay(const Conn* c, double delay)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setSecondDelay(const Conn*, double) - target data pointer is NULL.");
    if (obj->delay_.size() >= 2){
        obj->delay_[obj->delay_.size() - 1] = delay;
    }
}
double PulseGen::getSecondDelay(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getSecondDelay(Eref ) - target data pointer is NULL.");
    if (obj->delay_.size() >= 2){
        return obj->delay_[obj->delay_.size() - 1];
    }
    return 0.0;
}

void PulseGen::setBaseLevel(const Conn* c, double level)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setBaseLevel(const Conn*, double) - target data pointer is NULL.");
    obj->baseLevel_ = level;    
}
double PulseGen::getBaseLevel(Eref e)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getFirstDelay(Eref ) - target data pointer is NULL.");
    return obj->baseLevel_;    
}
void PulseGen::setTrigMode(const Conn* c, int mode)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setTrigMode(const Conn*, double) - target data pointer is NULL.");
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
void PulseGen::setTrigTime(const Conn* conn, double trigTime)
{
    PulseGen* obj = static_cast<PulseGen*> (conn->data());
    ASSERT( obj != NULL, "PulseGen::setTrigTime(const Conn*, double) - target data pointer is NULL.");
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

void PulseGen::setCount(const Conn* conn, int count)
{
    if (count <= 0){
        cout << "WARNING: invalid pulse count." << endl;
        return;
    }
    PulseGen* obj = static_cast<PulseGen*> (conn->data());
    // we want to keep it compact
    obj->level_.reserve(count);
    obj->delay_.reserve(count);
    obj->width_.reserve(count);
    obj->level_.resize(count);
    obj->delay_.resize(count);
    obj->width_.resize(count);
}

int PulseGen::getCount(Eref e)
{
    PulseGen* obj = static_cast<PulseGen*> (e.data());
    return obj->level_.size();        
}

double PulseGen::getLevel(Eref e, const int& index)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getLevel(Eref ) - target data pointer is NULL.");
    if (index >= 0 && index < obj->level_.size()){
        return obj->level_[index];
    } else {
        cout << "WARNING: PulseGen::getLevel - invalid index." << endl;
        return 0.0;
    }
}
    
void PulseGen::setLevel(const Conn* c, double level, const int& index)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setLevel(const Conn*, int, double) - target data pointer is NULL.");
    if (index >= 0 && index < obj->level_.size()){
        obj->level_[index] = level;
    } else {
        cout << "WARNING: PulseGen::setLevel - invalid index." << endl;
    }
}

double PulseGen::getWidth(Eref e, const int& index)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getWidth(Eref ) - target data pointer is NULL.");
    if (index >= 0 && index < obj->width_.size()){
        return obj->width_[index];
    } else {
        cout << "WARNING: PulseGen::getWidth - invalid index." << endl;
        return 0.0;
    }
}
void PulseGen::setWidth(const Conn* c, double width, const int& index)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setWidth(const Conn*, int, double) - target data pointer is NULL.");
    if (index >= 0 && index < obj->width_.size()){
        obj->width_[index] = width;
    } else {
        cout << "WARNING: PulseGen::setWidth - invalid index." << endl;
    }
}
double PulseGen::getDelay(Eref e, const int& index)
{
    PulseGen* obj = static_cast <PulseGen*> (e.data());
    ASSERT( obj != NULL, "PulseGen::getDelay(Eref ) - target data pointer is NULL.");
    if (index >= 0 && index < obj->delay_.size()){
        return obj->delay_[index];
    } else {
        cout << "WARNING: PulseGen::getDelay - invalid index." << endl;
        return 0.0;
    }
}

void PulseGen::setDelay(const Conn* c, double delay, const int& index)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::setDelay(const Conn*, int, double) - target data pointer is NULL.");
    if ( index >= 0 && index < obj->delay_.size() ){
        obj->delay_[index] = delay;
    } else {
        cout << "WARNING: PulseGen::setDelay - invalid index" << endl;
    }
}


//////////////////////////////////////////////////////////////////
// Message dest functions.
//////////////////////////////////////////////////////////////////

void PulseGen::inputFunc(const Conn* c, double value)
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::inputFunc(const Conn*, int) - target data pointer is NULL.");
    obj->input_ = value;
}

void PulseGen::setLevelFunc(const Conn* c, int index, double level)
{
    PulseGen::setLevel(c, level, index);
}
void PulseGen::setDelayFunc(const Conn* c, int index, double delay)
{
    PulseGen::setDelay(c, delay, index);
}
void PulseGen::setWidthFunc(const Conn* c, int index, double width)
{
    PulseGen::setWidth(c, width, index);
}

void PulseGen::processFunc( const Conn* c, ProcInfo p )
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::processFunc(const Conn*, ProcInfo) - target data pointer is NULL.");
    obj->innerProcessFunc(c, p);
}


void PulseGen::reinitFunc( const Conn* c, ProcInfo p )
{
    PulseGen* obj = static_cast<PulseGen*> (c->data());
    ASSERT( obj != NULL, "PulseGen::reinitFunc(const Conn*, ProcInfo) - target data pointer is NULL.");
    obj->trigTime_ = -1;
    obj->prevInput_ = 0;
    obj->output_ = obj->baseLevel_;
    obj->input_ = 0;    
}

void PulseGen::innerProcessFunc(const Conn* c, ProcInfo p)
{
    double currentTime = p->currTime_;
    double period = width_[0] + delay_[0];
    double phase = 0.0;
    for (unsigned int ii = 1; ii < width_.size(); ++ii){
        double incr = delay_[ii] + width_[ii] - width_[ii-1];
        if  (incr > 0){
            period += incr;
        }
    }
    switch (trigMode_){
        case PulseGen::FREE_RUN:
            phase = fmod(currentTime, period);
            break;
        case PulseGen::EXT_TRIG:
            if (input_ == 0){
                if (trigTime_ < 0){
                    phase = period;                
                }else{
                    phase = currentTime - trigTime_;
                }
            } else {
                if (prevInput_ == 0){
                    trigTime_ = currentTime;
                }
                phase = currentTime - trigTime_;
            }
            prevInput_ = input_;            
            break;
        case PulseGen::EXT_GATE:
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
            cerr << "ERROR: PulseGen::newProcessFunc( const Conn* , ProcInfo ) - invalid triggerMode - " << trigMode_ << endl;
    }
    if (phase >= period){ // we have crossed all pulses 
        output_ = baseLevel_;
        return;
    }
    // go through all pulse positions to check which pulse/interpulse
    // are we are in and set the output level accordingly
    for (unsigned int ii = 0; ii < width_.size(); ++ii){
        if (phase < delay_[ii]){ // we are in the baseline area - before ii-th pulse
            output_ = baseLevel_;
            break;
        } else if (phase < delay_[ii] + width_[ii]){ // we are inside th ii-th pulse
            output_ = level_[ii];
            break;
        }
        phase -= delay_[ii];
    }
    send1<double>( c->target(), outputSlot, output_);    
}
    
                     
#endif
