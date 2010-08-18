#ifndef _pymoose_KineticManager_h
#define _pymoose_KineticManager_h
#include "PyMooseBase.h"
namespace pymoose{
class KinCompt;
    class KineticManager : public KinCompt
    {
      public:
        static const std::string className_;
        KineticManager(Id id);
        KineticManager(std::string path);
        KineticManager(std::string name, Id parentId);
        KineticManager(std::string name, PyMooseBase& parent);
        KineticManager( const KineticManager& src, std::string name, PyMooseBase& parent);
        KineticManager( const KineticManager& src, std::string name, Id& parent);
        KineticManager( const KineticManager& src, std::string path);
        KineticManager( const Id& src, std::string name, Id& parent);
	KineticManager( const Id& src, std::string path);
        ~KineticManager();
        const std::string& getType();
            bool __get_autoMode() const;
            void __set_autoMode(bool autoMode);
            bool __get_stochastic() const;
            void __set_stochastic(bool stochastic);
            bool __get_spatial() const;
            void __set_spatial(bool spatial);
            string __get_method() const;
            void __set_method(string method);
            bool __get_variableDt() const;
            bool __get_singleParticle() const;
            bool __get_multiscale() const;
            bool __get_implicit() const;
            string __get_description() const;
            double __get_recommendedDt() const;
            double __get_loadEstimate() const;
            unsigned int __get_memEstimate() const;
            double __get_eulerError() const;
            void __set_eulerError(double eulerError);
    };
}
#endif
