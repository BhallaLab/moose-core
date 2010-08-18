#ifndef _pymoose_InterSolverFlux_h
#define _pymoose_InterSolverFlux_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class InterSolverFlux : public Neutral
    {      public:
        static const std::string className_;
        InterSolverFlux(Id id);
        InterSolverFlux(std::string path);
        InterSolverFlux(std::string name, Id parentId);
        InterSolverFlux(std::string name, PyMooseBase& parent);
        InterSolverFlux( const InterSolverFlux& src, std::string name, PyMooseBase& parent);
        InterSolverFlux( const InterSolverFlux& src, std::string name, Id& parent);
        InterSolverFlux( const InterSolverFlux& src, std::string path);
        InterSolverFlux( const Id& src, std::string name, Id& parent);
	InterSolverFlux( const Id& src, std::string path);
        ~InterSolverFlux();
        const std::string& getType();
            string __get_method() const;
            void __set_method(string method);
    };
}
#endif
