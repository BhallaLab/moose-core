/*******************************************************************
 * File:            StochSynchan.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-12-06 12:22:43
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

#ifndef _pymoose_StochSynchan_h
#define _pymoose_StochSynchan_h
#include "PyMooseIterable.h"
#include "PyMooseBase.h"
#include "Neutral.h"
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    Wrapper for StochSynchan class of MOOSE. It was created by
//    modifiying the auto-generated wrappers manually. So do not edit
//    anything without understanding what is going on.
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
namespace pymoose
{

    class StochSynchan;
    typedef InnerPyMooseIterable<StochSynchan, unsigned int, double>  StochSynchanDoubleArray;

    class StochSynchan : public Neutral
    {    public:
        static const std::string className_;
        StochSynchan(Id id);
        StochSynchan(std::string path);
        StochSynchan(std::string name, Id parentId);
        StochSynchan(std::string name, PyMooseBase& parent);
        StochSynchan( const StochSynchan& src, std::string name, PyMooseBase& parent);
        StochSynchan( const StochSynchan& src, std::string name, Id& parent);
        StochSynchan( const StochSynchan& src, std::string path);
        StochSynchan( const Id& src, std::string name, Id& parent);
        StochSynchan( const Id& src, std::string path);
        ~StochSynchan();
        const std::string& getType();
        double __get_Gbar() const;
        void __set_Gbar(double Gbar);
        double __get_Ek() const;
        void __set_Ek(double Ek);
        double __get_tau1() const;
        void __set_tau1(double tau1);
        double __get_tau2() const;
        void __set_tau2(double tau2);
        bool __get_normalizeWeights() const;
        void __set_normalizeWeights(bool normalizeWeights);
        double __get_Gk() const;
        void __set_Gk(double Gk);
        double __get_Ik() const;
        void __set_Ik(double Ik);
        unsigned int __get_numSynapses() const;
        void __set_numSynapses(unsigned int index, unsigned int num)
        {
            /*dummy*/
        }
    
        double __get_weight(unsigned int index) const;
        void __set_weight(unsigned int index,double weight);
        double __get_delay( unsigned int index) const;
        void __set_delay(unsigned int index, double delay);
        double __get_releaseP(unsigned int index) const;
        void __set_releaseP(unsigned int index, double releaseP);
        double __get_releaseCount(unsigned int index) const;    
        void __set_releaseCount(unsigned int index, double releaseCount)
        {
            /* dummy*/
        }
    
        double __get_synapse() const;
        void __set_synapse(double synapse);
        double __get_activation() const;
        void __set_activation(double activation);
        double __get_modulator() const;
        void __set_modulator(double modulator);

        // These are for wrapping LookupFinfo members of the MOOSE class StochSynchan
        // so that they look like array type members from inside python
        StochSynchanDoubleArray * weight;    
        StochSynchanDoubleArray * delay;
        StochSynchanDoubleArray * releaseP;
        StochSynchanDoubleArray * releaseCount;
    };
}

#endif
