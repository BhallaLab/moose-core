#ifndef _pymoose_SteadyState_h
#define _pymoose_SteadyState_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

class SteadyState : public Neutral
    {      public:
        static const std::string className_;
        SteadyState(Id id);
        SteadyState(std::string path);
        SteadyState(std::string name, Id parentId);
        SteadyState(std::string name, PyMooseBase& parent);
        SteadyState( const SteadyState& src, std::string name, PyMooseBase& parent);
        SteadyState( const SteadyState& src, std::string name, Id& parent);
        SteadyState( const SteadyState& src, std::string path);
        SteadyState( const Id& src, std::string name, Id& parent);
        SteadyState( const Id& src, std::string path);
        ~SteadyState();
        const std::string& getType();
            bool __get_badStoichiometry() const;
            bool __get_isInitialized() const;
            unsigned int __get_nIter() const;
            const string&  __get_status() const;
            unsigned int __get_maxIter() const;
            void __set_maxIter(unsigned int maxIter);
            double __get_convergenceCriterion() const;
            void __set_convergenceCriterion(double convergenceCriterion);
            unsigned int __get_nVarMols() const;
            unsigned int __get_rank() const;
            unsigned int __get_stateType() const;
            unsigned int __get_nNegEigenvalues() const;
            unsigned int __get_nPosEigenvalues() const;
            unsigned int __get_solutionStatus() const;
    };
}
#endif
