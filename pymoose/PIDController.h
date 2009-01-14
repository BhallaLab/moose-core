#ifndef _pymoose_PIDController_h
#define _pymoose_PIDController_h
#include "PyMooseBase.h"
namespace pymoose{
    class PIDController : public PyMooseBase
    {      public:
        static const std::string className_;
        PIDController(Id id);
        PIDController(std::string path);
        PIDController(std::string name, Id parentId);
        PIDController(std::string name, PyMooseBase& parent);
        PIDController( const PIDController& src, std::string name, PyMooseBase& parent);
        PIDController( const PIDController& src, std::string name, Id& parent);
        PIDController( const PIDController& src, std::string path);
        PIDController( const Id& src, std::string name, Id& parent);
        ~PIDController();
        const std::string& getType();
            double __get_gain() const;
            void __set_gain(double gain);
            double __get_saturation() const;
            void __set_saturation(double saturation);
            double __get_command() const;
            void __set_command(double command);
            double __get_sensed() const;
            double __get_tauI() const;
            void __set_tauI(double tauI);
            double __get_tauD() const;
            void __set_tauD(double tauD);
            double __get_output() const;
    };
}
#endif
