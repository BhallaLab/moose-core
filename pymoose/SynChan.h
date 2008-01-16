#ifndef _pymoose_SynChan_h
#define _pymoose_SynChan_h
#include "PyMooseBase.h"
namespace pymoose
{
    class SynChan : public PyMooseBase
    {    public:
        static const std::string className;
        SynChan(Id id);
        SynChan(std::string path);
        SynChan(std::string name, Id parentId);
        SynChan(std::string name, PyMooseBase* parent);
        ~SynChan();
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
        void __set_numSynapses(unsigned int numSynapses);
        double __get_weight() const;
        void __set_weight(double weight);
        double __get_delay() const;
        void __set_delay(double delay);
        double __get_IkSrc() const;
        void __set_IkSrc(double IkSrc);
        double __get_synapse() const;
        void __set_synapse(double synapse);
        double __get_activation() const;
        void __set_activation(double activation);
        double __get_modulator() const;
        void __set_modulator(double modulator);
    };
}

#endif
