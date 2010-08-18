#ifndef _pymoose_SynChan_h
#define _pymoose_SynChan_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class SynChan : public Neutral
    {      public:
        static const std::string className_;
        SynChan(Id id);
        SynChan(std::string path);
        SynChan(std::string name, Id parentId);
        SynChan(std::string name, PyMooseBase& parent);
        SynChan( const SynChan& src, std::string name, PyMooseBase& parent);
        SynChan( const SynChan& src, std::string name, Id& parent);
        SynChan( const SynChan& src, std::string path);
        SynChan( const Id& src, std::string name, Id& parent);
        SynChan( const Id& src, std::string path);
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
        unsigned int __get_numSynapses() const;
        // vector<SynInfo>& __get_synapse() const;
        double getWeight(const unsigned int& index) const;
        void setWeight(const unsigned int& index, double weight);
        double getDelay(const unsigned int& index) const;
        void setDelay(const unsigned int& index, double delay);
      protected:
        // This constructor is for allowing derived type (Table) to
        // have constructors exactly as if it was directly derived from PyMooseBase.
    
        SynChan(std::string className, std::string objectName, Id parentId);
        SynChan(std::string className, std::string path);    
        SynChan(std::string className, std::string objectName, PyMooseBase& parent);
        
    };
}

#endif
