#ifndef _pymoose_CylPanel_h
#define _pymoose_CylPanel_h
#include "PyMooseBase.h"
#include "Panel.h"
namespace pymoose{
    class CylPanel : public Panel
    {      public:
        static const std::string className_;
        CylPanel(Id id);
        CylPanel(std::string path);
        CylPanel(std::string name, Id parentId);
        CylPanel(std::string name, PyMooseBase& parent);
        CylPanel( const CylPanel& src, std::string name, PyMooseBase& parent);
        CylPanel( const CylPanel& src, std::string name, Id& parent);
        CylPanel( const CylPanel& src, std::string path);
        CylPanel( const Id& src, std::string name, Id& parent);
	CylPanel( const Id& src, std::string path);
        ~CylPanel();
        const std::string& getType();
    };
}
#endif
