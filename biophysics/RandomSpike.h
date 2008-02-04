
/*******************************************************************
 * File:            RandomSpike.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-02-04 11:26:23
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

#ifndef _RANDOMSPIKE_H
#define _RANDOMSPIKE_H
/*
Object Type:    randomspike

Description:    place a random event into the buffer

Author:         M. Wilson, Caltech 6/88, Dave Bilitch 1/94

------------------------------------------------------------------------------

ELEMENT PARAMETERS

DataStructure:  Randomspike_type  [src/newconn/newconn_struct.h]

Size:           100 bytes

Fields:         min_amp         minimum amplitude of event
                max_amp         maximum amplitude of event
                rate            rate of generation of events
                reset           flag for whether to reset after each event
                reset_value     what to reset state to
                state           current state of object
                abs_refract     minimum time between events

------------------------------------------------------------------------------

SIMULATION PARAMETERS

Function:       RandomEvent  [in src/newconn/randomspike.c]

Classes:        buffer

Actions:        INIT
                RESET
                PROCESS

Messages:       RATE    rate
                MINMAX  min max

------------------------------------------------------------------------------

Notes:          Generates a time series of events at a rate given by the rate
                parameter. The probability of an event for a single time step
                is given by rate*dt where dt is the clock rate of the
                element.  However, no event will be generated at a time less
                than abs_refract.  When an event has been generated, the
                amplitude of the event is a random variable uniformly
                distributed between min_amp and max_amp.  The state field
                has the value of the event amplitude if an event has been
                generated. If an event is not generated then the value of
                the state field depends on the reset field.  If reset is
                non-zero then the state is takes on the value given in
                reset_value. Otherwise the state will behave like a latch
                containing the amplitude of the previous event.

Example:        Scripts/tutorials/tutorial4.g

See also:       
*/

class RandomSpike
{
  public:
    static void setMinAmp(const Conn& c, double value);    
    static double getMinAmp(const Element* e);
    static void setMaxAmp(const Conn& c, double value);
    static double getMaxAmp(const Element* e);
    static void setRate(const Conn& c, double value);
    static double getRate(const Element* e);
    static void setResetValue(const Conn& c, double value);
    static double getResetValue(const Element* e);
    static void setState(const Conn& c, double value);
    static double getState(const Element* e);
    static void setAbsRefract(const Conn& c, double value);
    static double getAbsRefract(const Element* e);
    static void setLastEvent(const Conn& c, double value);
    static double getLastEvent(const Element* e);
    static void setReset(const Conn& c, int value);
    static int getReset(const Element* e);
    static void setMinMaxAmp(const Conn & c, double min, double max);    
    static void processFunc(const Conn& c, ProcInfo p);
    void innerProcessFunc(const Conn & c, ProcInfo p);
    static void reinitFunc(const Conn& c, ProcInfo p);
    
    RandomSpike();
    
  private:
    double minAmp_;
    double maxAmp_;
    double rate_;
    int reset_;
    double resetValue_;
    double state_;
    double absRefract_;
    double lastEvent_;    
};

    
#endif
