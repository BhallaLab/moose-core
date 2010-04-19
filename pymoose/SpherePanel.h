#ifndef _pymoose_SpherePanel_h
#define _pymoose_SpherePanel_h
#include "PyMooseBase.h"
#include "Panel.h"

namespace pymoose{
    class SpherePanel : public Panel
    {      public:
        static const std::string className_;
        SpherePanel(Id id);
        SpherePanel(std::string path);
        SpherePanel(std::string name, Id parentId);
        SpherePanel(std::string name, PyMooseBase& parent);
        SpherePanel( const SpherePanel& src, std::string name, PyMooseBase& parent);
        SpherePanel( const SpherePanel& src, std::string name, Id& parent);
        SpherePanel( const SpherePanel& src, std::string path);
        SpherePanel( const Id& src, std::string name, Id& parent);
        SpherePanel( const Id& src, std::string path);
        ~SpherePanel();
        const std::string& getType();
    };
}
#endif
