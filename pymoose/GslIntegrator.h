#ifndef _pymoose_GslIntegrator_h
#define _pymoose_GslIntegrator_h
#include "Neutral.h"
namespace pymoose{
    class PyMooseBase;
    class Neutral;
    class GslIntegrator : public Neutral
    {
      public:
        static const std::string className_;
        GslIntegrator(std::string className, std::string objectName, Id parentId);
        GslIntegrator(std::string className, std::string path);
        GslIntegrator(std::string className, std::string objectName, PyMooseBase& parent);
        GslIntegrator(Id id);
        GslIntegrator(std::string path);
        GslIntegrator(std::string name, Id parentId);
        GslIntegrator(std::string name, PyMooseBase& parent);
        GslIntegrator( const GslIntegrator& src, std::string name, PyMooseBase& parent);
        GslIntegrator( const GslIntegrator& src, std::string name, Id& parent);
        GslIntegrator( const GslIntegrator& src, std::string path);
        GslIntegrator( const Id& src, std::string name, Id& parent);
        GslIntegrator( const Id& src, std::string path);
        ~GslIntegrator();
        const std::string& getType();
            bool __get_isInitiatilized() const;
            const string&  __get_method() const;
            void __set_method(string method);
            double __get_relativeAccuracy() const;
            void __set_relativeAccuracy(double relativeAccuracy);
            double __get_absoluteAccuracy() const;
            void __set_absoluteAccuracy(double absoluteAccuracy);
            double __get_internalDt() const;
            void __set_internalDt(double internalDt);
    };

}
#endif
