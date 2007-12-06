/*******************************************************************
 * File:            StochSynchan.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-26 10:49:16
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

#ifndef _STOCHSYNCHAN_H
#define _STOCHSYNCHAN_H
#include "StochSynInfo.h"

class StochSynchan
{
    
  public:
    // Functions duplicated from SynChan
    StochSynchan(): Ek_( 0.0 ), Gk_( 0.0 ), Ik_( 0.0 ), Gbar_( 0.0 ), 
                  tau1_( 1.0e-3 ), tau2_( 1.0e-3 ),
                  normalizeWeights_( 0 )
    {
        ;
    }

    static void setGbar( const Conn& c, double Gbar );
    static double getGbar( const Element* e );

    static void setEk( const Conn& c, double Ek );
    static double getEk( const Element* e );

    static void setTau1( const Conn& c, double tau1 );
    static double getTau1( const Element* e );

    static void setTau2( const Conn& c, double tau2 );
    static double getTau2( const Element* e );

    static void setNormalizeWeights( const Conn& c, bool value );
    static bool getNormalizeWeights( const Element* e );

    static void setGk( const Conn& c, double Gk );
    static double getGk( const Element* e );

    static void setIk( const Conn& c, double Ik );
    static double getIk( const Element* e );

    static int getNumSynapses( const Element* e );

    static void setWeight(
        const Conn& c, double val, const unsigned int& i );
    static double getWeight( 
        const Element* e, const unsigned int& i );

    static void setDelay(
        const Conn& c, double val, const unsigned int& i );
    static double getDelay( 
        const Element* e, const unsigned int& i );
///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

    void innerSynapseFunc( const Conn& c, double time );
    static void synapseFunc( const Conn& c, double time );

    static void channelFunc( const Conn& c, double Vm );

    void innerProcessFunc( Element* e, ProcInfo p );
    static void processFunc( const Conn& c, ProcInfo p );
    void innerReinitFunc( Element* e,  ProcInfo p );
    static void reinitFunc( const Conn& c, ProcInfo p );

    static void activationFunc( const Conn& c, double val );
    static void modulatorFunc( const Conn& c, double val );

///////////////////////////////////////////////////
// Utility function
///////////////////////////////////////////////////
    unsigned int updateNumSynapse( const Element* e );

    ///////////////////////////////////////
    // Functions specific to StochSynchan
    ///////////////////////////////////////
    static void setReleaseP(const Conn& c, double p, const unsigned int& index);
    static double getReleaseP(const Element* e, const unsigned int& index);
    
    static double getReleaseCount( const Element* e, const unsigned int& i );
    
///////////////////////////////////////////////////
// Private fields.
///////////////////////////////////////////////////

  private:
    double Ek_;
    double Gk_;
    double Ik_;
    double Gbar_;
    double tau1_;
    double tau2_;
    int normalizeWeights_;
    double xconst1_;
    double yconst1_;
    double xconst2_;
    double yconst2_;
    double norm_;
    double activation_;
    double modulation_;
    double X_;	
    double Y_;	
    double Vm_;
    vector< StochSynInfo > synapses_;
    priority_queue< SynInfo > pendingEvents_;
};

#endif
