#ifndef _pymoose_TriPanel_h
#define _pymoose_TriPanel_h
#include "PyMooseBase.h"
#include "Panel.h"
namespace pymoose{
    class TriPanel : public Panel
    {      public:
        static const std::string className_;
        TriPanel(Id id);
        TriPanel(std::string path);
        TriPanel(std::string name, Id parentId);
        TriPanel(std::string name, PyMooseBase& parent);
        TriPanel( const TriPanel& src, std::string name, PyMooseBase& parent);
        TriPanel( const TriPanel& src, std::string name, Id& parent);
        TriPanel( const TriPanel& src, std::string path);
        TriPanel( const Id& src, std::string name, Id& parent);
        TriPanel( const Id& src, std::string path);
        ~TriPanel();
        const std::string& getType();
    };
}
#endif
