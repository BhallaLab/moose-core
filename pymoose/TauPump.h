#ifndef _pymoose_TauPump_h
#define _pymoose_TauPump_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class TauPump : public Neutral
    {      public:
        static const std::string className_;
        TauPump(Id id);
        TauPump(std::string path);
        TauPump(std::string name, Id parentId);
        TauPump(std::string name, PyMooseBase& parent);
        TauPump( const TauPump& src, std::string name, PyMooseBase& parent);
        TauPump( const TauPump& src, std::string name, Id& parent);
        TauPump( const TauPump& src, std::string path);
        TauPump( const Id& src, std::string name, Id& parent);
        TauPump( const Id& src, std::string path);
        ~TauPump();
        const std::string& getType();
            double __get_pumpRate() const;
            void __set_pumpRate(double pumpRate);
            double __get_eqConc() const;
            void __set_eqConc(double eqConc);
            double __get_TA() const;
            void __set_TA(double TA);
            double __get_TB() const;
            void __set_TB(double TB);
            double __get_TC() const;
            void __set_TC(double TC);
            double __get_TV() const;
            void __set_TV(double TV);
    };
}
#endif
