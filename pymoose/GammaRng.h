#ifndef _pymoose_GammaRng_h
#define _pymoose_GammaRng_h
#include "RandGenerator.h"
class GammaRng : public RandGenerator
{    public:
        static const std::string className;
        GammaRng(Id id);
        GammaRng(std::string path);
        GammaRng(std::string name, Id parentId);
        GammaRng(std::string name, PyMooseBase* parent);
        ~GammaRng();
        const std::string& getType();
        double __get_alpha() const;
        void __set_alpha(double alpha);
        double __get_theta() const;
        void __set_theta(double theta);
};
#endif
