/*******************************************************************
 * File:            STPSynChan.h
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

#ifndef _STPSYNCHAN_H
#define _STPSYNCHAN_H

#include <queue>
#include "SynInfo.h"
#include "SynChan.h"

class STPSynChan: public SynChan
{    
  public:
    // Functions duplicated from SynChan
    STPSynChan();
    virtual ~STPSynChan(){};
    ///////////////////////////////////////
    // Functions specific to STPSynChan
    ///////////////////////////////////////
    static void setInitPr(const Conn* c, double p, const unsigned int& index);
    static double getInitPr(Eref e, const unsigned int& index);
    static double getPr(Eref e, const unsigned int& index);
    static void setInitF(const Conn* c, double p, const unsigned int& index);
    static double getInitF(Eref e, const unsigned int& index);
    static void setInitD1(const Conn* c, double p, const unsigned int& index);
    static double getInitD1(Eref e, const unsigned int& index);
    static void setInitD2(const Conn* c, double p, const unsigned int& index);
    static double getInitD2(Eref e, const unsigned int& index);
    static void set_d1(const Conn* c, double value);
    static double get_d1(Eref e);
    static void set_d2(const Conn* c, double value);
    static double get_d2(Eref e);
    static void setDeltaF(const Conn* c, double value);
    static double getDeltaF(Eref e);
    static void setTauF(const Conn* c, double value);
    static double getTauF(Eref e);
    static void setTauD1(const Conn* c, double value);
    static double getTauD1(Eref e);
    static void setTauD2(const Conn* c, double value);
    static double getTauD2(Eref e);
    static double getD1(Eref e, const unsigned int& index);
    static double getD2(Eref e, const unsigned int& index);
    static double getF(Eref e, const unsigned int & index);
    
///////////////////////////////////////////////////
// Private fields.
///////////////////////////////////////////////////

  protected:
    void innerSetInitPr(Eref e, double value, const unsigned int& index);
    double innerGetInitPr(Eref e, const unsigned int& index);
    double innerGetPr(Eref e, const unsigned int& index);
    void innerSetInitF(Eref e, double value, const unsigned int& index);
    double innerGetInitF(Eref e, const unsigned int& index);
    void innerSetInitD1(Eref e, double value, const unsigned int& index);
    double innerGetInitD1(Eref e, const unsigned int& index);
    void innerSetInitD2(Eref e, double value, const unsigned int& index);
    double innerGetInitD2(Eref e, const unsigned int& index);
    ///////////////////////////////////////////////////
    // Dest function definitions
    ///////////////////////////////////////////////////

    virtual void innerSynapseFunc( const Conn* c, double time );
    virtual void innerProcessFunc( Eref e, ProcInfo p );
    virtual void innerReinitFunc( Eref e,  ProcInfo p );

    ///////////////////////////////////////////////////
    // Utility function
    ///////////////////////////////////////////////////
    virtual unsigned int updateNumSynapse( Eref e );

    double d1_, d2_, deltaF_, tauD1_, tauD2_, tauF_, dt_tauF_, dt_tauD1_, dt_tauD2_;
    vector< double > initPr_, F_, D1_, D2_, initF_, initD1_, initD2_, amp_;
    
};

extern const Cinfo * initSynChanCinfo();
#endif
