#ifndef _pymoose_StochSpikeGen_h
#define _pymoose_StochSpikeGen_h
#include "SpikeGen.h"
namespace pymoose{
    class PyMooseBase;
    class SpikeGen;
    class StochSpikeGen : public SpikeGen
    {
      public:
        static const std::string className_;
        StochSpikeGen(std::string className, std::string objectName, Id parentId);
        StochSpikeGen(std::string className, std::string path);
        StochSpikeGen(std::string className, std::string objectName, PyMooseBase& parent);
        StochSpikeGen(Id id);
        StochSpikeGen(std::string path);
        StochSpikeGen(std::string name, Id parentId);
        StochSpikeGen(std::string name, PyMooseBase& parent);
        StochSpikeGen( const StochSpikeGen& src, std::string name, PyMooseBase& parent);
        StochSpikeGen( const StochSpikeGen& src, std::string name, Id& parent);
        StochSpikeGen( const StochSpikeGen& src, std::string path);
        StochSpikeGen( const Id& src, std::string name, Id& parent);
        StochSpikeGen( const Id& src, std::string path);
        ~StochSpikeGen();
        const std::string& getType();
            double __get_failureP() const;
            void __set_failureP(double failureP);
    };

}
#endif
