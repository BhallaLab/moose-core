#ifndef _pymoose_HHGate2D_h
#define _pymoose_HHGate2D_h
#include "PyMooseBase.h"
namespace pymoose{
    class HHGate2D : public PyMooseBase
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
        const std::string& getType();
    };
}
#endif
