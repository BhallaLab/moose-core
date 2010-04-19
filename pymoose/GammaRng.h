#ifndef _pymoose_GammaRng_h
#define _pymoose_GammaRng_h
#include "RandGenerator.h"
namespace pymoose
{
    class GammaRng : public RandGenerator
    {
      public:
        static const std::string className_;
        GammaRng(Id id);
        GammaRng(std::string path);
        GammaRng(std::string name, Id parentId);
        GammaRng(std::string name, PyMooseBase& parent);
        GammaRng(const GammaRng& src,std::string name, PyMooseBase& parent);
        GammaRng(const GammaRng& src,std::string name, Id& parent);
        GammaRng(const Id& src,std::string name, Id& parent);
        GammaRng(const GammaRng& src,std::string path);
        GammaRng(const Id& src,std::string path);
        ~GammaRng();
        const std::string& getType();
        double __get_alpha() const;
        void __set_alpha(double alpha);
        double __get_theta() const;
        void __set_theta(double theta);
    };
}

#endif
