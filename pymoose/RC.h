#ifndef _pymoose_RC_h
#define _pymoose_RC_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class RC : public Neutral
    {      public:
        static const std::string className_;
        RC(Id id);
        RC(std::string path);
        RC(std::string name, Id parentId);
        RC(std::string name, PyMooseBase& parent);
        RC( const RC& src, std::string name, PyMooseBase& parent);
        RC( const RC& src, std::string name, Id& parent);
        RC( const RC& src, std::string path);
        RC( const Id& src, std::string name, Id& parent);
        RC( const Id& src, std::string path);
        ~RC();
        const std::string& getType();
            double __get_V0() const;
            void __set_V0(double V0);
            double __get_R() const;
            void __set_R(double R);
            double __get_C() const;
            void __set_C(double C);
            double __get_state() const;
            double __get_inject() const;
            void __set_inject(double inject);
    };
}
#endif
