#ifndef _pymoose_HHGate2D_h
#define _pymoose_HHGate2D_h
#include "PyMooseBase.h"
#include "HHGate.h"
#include "Interpol2D.h"
namespace pymoose{
class HHGate;
    class HHGate2D : public HHGate
    {      public:
        static const std::string className_;
        HHGate2D(Id id);
        HHGate2D(std::string path);
        HHGate2D(std::string name, Id parentId);
        HHGate2D(std::string name, PyMooseBase& parent);
        HHGate2D( const HHGate2D& src, std::string name, PyMooseBase& parent);
        HHGate2D( const HHGate2D& src, std::string name, Id& parent);
        HHGate2D( const HHGate2D& src, std::string path);
        HHGate2D( const Id& src, std::string name, Id& parent);
	HHGate2D( const Id& src, std::string path);
        ~HHGate2D();
        Interpol2D* __get_A() const;
        Interpol2D* __get_B() const;        
        const std::string& getType();
    };
}
#endif
