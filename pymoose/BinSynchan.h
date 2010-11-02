/*******************************************************************
 * File:            pymoose/BinSynchan.h
 * Description:     SWIG wrapper for BinSynchan class. It has been
 *                      created by modifying the generated class
 *                      declaration.
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-12-03 10:56:28
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

#ifndef _pymoose_BinSynchan_h
#define _pymoose_BinSynchan_h
#include "PyMooseIterable.h"
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose
{
    class BinSynchan;    
    typedef  InnerPyMooseIterable <BinSynchan, unsigned int, double> BinSynchanDoubleArray;
    typedef  InnerPyMooseIterable <BinSynchan, unsigned int, int> BinSynchanIntArray;
    class PyMooseBase;
    

    class BinSynchan : public Neutral
    {
      public:
        static const std::string className_;
        BinSynchan(Id id);
        BinSynchan(std::string path);
        BinSynchan(std::string name, Id parentId);
        BinSynchan(std::string name, PyMooseBase& parent);
        BinSynchan(const BinSynchan& src,std::string name, PyMooseBase& parent);
        BinSynchan(const BinSynchan& src,std::string name, Id& parent);
        BinSynchan(const Id& src,std::string name, Id& parent);
        BinSynchan(const BinSynchan& src,std::string path);
        BinSynchan(const Id& src,std::string path);
        ~BinSynchan();
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
        void __set_numSynapses(unsigned int index, unsigned int num){/*dummy*/}
    

        double __get_weight(unsigned int index) const;
        void __set_weight(unsigned int index, double weight);


        double __get_delay(unsigned int index) const;
        void __set_delay(unsigned int index, double delay);
    

        int __get_poolSize(unsigned int index) const;
        void __set_poolSize(unsigned int index, int size);
    

        double __get_releaseP(unsigned int index) const;
        void __set_releaseP(unsigned int index, double releaseP);
    
    

        double __get_releaseCount(unsigned int index) const;
        void __set_releaseCount(unsigned int index, double releaseCount){/*dummy*/}
    
        double __get_synapse() const;
        void __set_synapse(double synapse);
    
        double __get_activation() const;
        void __set_activation(double activation);
    
        double __get_modulator() const;
        void __set_modulator(double modulator);
    
        // Data fields: These are to wrap LookupFinfo fields in the MOOSE class
        // so that from python they look like array type members of BinSynchan.
        BinSynchanDoubleArray* weight;    
        BinSynchanDoubleArray* delay;
        BinSynchanDoubleArray* releaseP;
        BinSynchanIntArray* poolSize;    
        BinSynchanDoubleArray* releaseCount;    
    };
}
// namespace pymoose
#endif
