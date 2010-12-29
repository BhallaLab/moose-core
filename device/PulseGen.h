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
    /**
       With trigMode = FREE_RUN the pulse generator will create a
       train of pulses determined by the firstDealy, firstWidth,
       firstLevel, secondDelay, secondWidth and secondLevel.       
    */
    static const int FREE_RUN = 0;
    /**
       With trigMode = EXT_TRIG the pulse generator may be triggered
       or may generate a single pulse and then stop.  The logic
       depends on trigTime field. If trigTime is negative ( which it
       is by default and remains so unless there is a non zero input
       trigger to the PulseGen object ), the output remains at
       baselevel. For trigTime >= 0, if there is no input trigger, the
       PulseGen object will produce pair of pulses, the first one
       starting at firstDelay and staying at firstLevel for firstWidth
       time. And then the second one starting at secondDelay time
       after the start of the first pulse, staying at secondLevel for
       secondWidth time. If secondDelay is less than firstWidth, there
       will be no second pulse.

     */
    static const  int EXT_TRIG = 1;
    /**
       With trigMode = EXT_GATE, the pulse occurs firstDelay later
       from the leading edge of the input.
     */
    static const int EXT_GATE = 2;    

    PulseGen();

    //////////////////////////////////////////////////////////////////
    // Field functions.
    //////////////////////////////////////////////////////////////////
    
    static void setFirstLevel(const Conn* c, double level);
    static double getFirstLevel(Eref e);
    static void setFirstWidth(const Conn* c, double width);
    static double getFirstWidth(Eref e);
    static void setFirstDelay(const Conn* c, double delay);
    static double getFirstDelay(Eref e);
    
    static void setSecondLevel(const Conn* c, double level);
    static double getSecondLevel(Eref e);
    static void setSecondWidth(const Conn* c, double width);
    static double getSecondWidth(Eref e);
    static void setSecondDelay(const Conn* c, double delay);
    static double getSecondDelay(Eref e);

    static void setBaseLevel(const Conn* c, double level);
    static double getBaseLevel(Eref e);
    static void setTrigMode(const Conn* c, int mode);
    static int getTrigMode(Eref e);
    static double getOutput(Eref e);
    static double getTrigTime(Eref e);
    static void setTrigTime(const Conn* c, double trigTime);
    static int getPreviousInput(Eref e);
    
    //////////////////////////////////////////////////////////////////
    // Message dest functions.
    //////////////////////////////////////////////////////////////////
    static void inputFunc(const Conn* c, double input);
    /// The following three are for lookup (valuefinfos - where index
    /// comes after value.
    static void setLevel(const Conn* c, double level, const int& pulseNo);
    static void setWidth(const Conn* c, double width, const int& pulseNo);
    static void setDelay(const Conn* c, double delay, const int& pulseNo);
    /// The following three are for dest messages - which have order
    /// of parameters: index, value in GENESIS.
    static void setLevelFunc(const Conn* c, int pulseNo, double level);
    static void setWidthFunc(const Conn* c, int pulseNo, double width);
    static void setDelayFunc(const Conn* c, int pulseNo, double delay);
    
    static void processFunc( const Conn* c, ProcInfo p );
    static void reinitFunc( const Conn* c, ProcInfo p );
    static void setCount(const Conn* c, int count);
    static int getCount(Eref e);
    static double getWidth(Eref e, const int& index);
    static double getDelay(Eref e, const int& index);
    static double getLevel(Eref e, const int& index);
    
  protected:
    void innerProcessFunc( const Conn* c, ProcInfo p );    
    
  private:
    vector <double> delay_;
    vector <double> level_;
    vector <double> width_;

    
    double output_;
    double baseLevel_;
    double trigTime_;
    int trigMode_;
    bool secondPulse_;
    
    int prevInput_;
    int input_;    
};

    
#endif
