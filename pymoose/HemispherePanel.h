#ifndef _pymoose_HemispherePanel_h
#define _pymoose_HemispherePanel_h
#include "PyMooseBase.h"
#include "Panel.h"
namespace pymoose{
    class HemispherePanel : public Panel
    {      public:
        static const std::string className_;
        HemispherePanel(Id id);
        HemispherePanel(std::string path);
        HemispherePanel(std::string name, Id parentId);
        HemispherePanel(std::string name, PyMooseBase& parent);
        HemispherePanel( const HemispherePanel& src, std::string name, PyMooseBase& parent);
        HemispherePanel( const HemispherePanel& src, std::string name, Id& parent);
        HemispherePanel( const HemispherePanel& src, std::string path);
        HemispherePanel( const Id& src, std::string name, Id& parent);
	HemispherePanel( const Id& src, std::string path);
        ~HemispherePanel();
        const std::string& getType();
    };
}
#endif
