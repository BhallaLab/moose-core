/*******************************************************************
 * File:            PulseGen.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-02-01 12:01:21
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

#ifndef _PULSEGEN_H
#define _PULSEGEN_H
class PulseGen
{
  public:
    static const int FREE_RUN = 0;
    static const  int EXT_TRIG = 1;
    static const int EXT_GATE = 2;    

    PulseGen();

    //////////////////////////////////////////////////////////////////
    // Field functions.
    //////////////////////////////////////////////////////////////////
    
    static void setFirstLevel(const Conn& c, double level);
    static double getFirstLevel(const Element* e);
    static void setFirstWidth(const Conn& c, double width);
    static double getFirstWidth(const Element * e);
    static void setFirstDelay(const Conn & c, double delay);
    static double getFirstDelay(const Element* e);
    
    static void setSecondLevel(const Conn& c, double level);
    static double getSecondLevel(const Element* e);
    static void setSecondWidth(const Conn& c, double width);
    static double getSecondWidth(const Element* e);
    static void setSecondDelay(const Conn& c, double delay);
    static double getSecondDelay(const Element* e);

    static void setBaseLevel(const Conn& c, double level);
    static double getBaseLevel(const Element* e);
    static void setTrigMode(const Conn& c, int mode);
    static int getTrigMode(const Element* e);
    static double getOutput(const Element* e);
    static double getTrigTime(const Element* e);
    static int getPreviousInput(const Element* e);
    
    //////////////////////////////////////////////////////////////////
    // Message dest functions.
    //////////////////////////////////////////////////////////////////
    static void inputFunc(const Conn& c, double input);
    static void setPulseLevel(const Conn& c, int pulseNo, double level);
    static void setPulseWidth(const Conn& c, int pulseNo, double width);
    static void setPulseDelay(const Conn& c, int pulseNo, double delay);
    void innerProcessFunc( const Conn& c, ProcInfo p );
    static void processFunc( const Conn& c, ProcInfo p );
    static void reinitFunc( const Conn& c, ProcInfo p );

  private:
    double firstLevel_;
    double firstWidth_;
    double firstDelay_;
    
    double secondLevel_;
    double secondWidth_;
    double secondDelay_;
    
    double output_;
    double baseLevel_;
    double trigTime_;
    int trigMode_;
    bool secondPulse_;
    
    int prevInput_;
    int input_;    
};

    
#endif
