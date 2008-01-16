#ifndef _pymoose_PoissonRng_h
#define _pymoose_PoissonRng_h
#include "RandGenerator.h"
namespace pymoose
{
    class PoissonRng : public RandGenerator
    {    public:
        static const std::string className;
        PoissonRng(Id id);
        PoissonRng(std::string path);
        PoissonRng(std::string name, Id parentId);
        PoissonRng(std::string name, PyMooseBase* parent);
        ~PoissonRng();
        const std::string& getType();
        double __get_mean() const;
        void __set_mean(double mean);
    };
}

#endif
