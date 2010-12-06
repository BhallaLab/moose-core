#ifndef _pymoose_PoissonRng_h
#define _pymoose_PoissonRng_h
#include "RandGenerator.h"
namespace pymoose
{
    class PoissonRng : public RandGenerator
    {    public:
        static const std::string className_;
        PoissonRng(Id id);
        PoissonRng(std::string path);
        PoissonRng(std::string name, Id parentId);
        PoissonRng(std::string name, PyMooseBase& parent);
        PoissonRng(const PoissonRng& src,std::string name, PyMooseBase& parent);
        PoissonRng(const PoissonRng& src,std::string name, Id& parent);
        PoissonRng(const Id& src,std::string name, Id& parent);
        PoissonRng(const PoissonRng& src,std::string path);
        PoissonRng(const Id& src,std::string path);
        ~PoissonRng();
        const std::string& getType();
    };
}

#endif
