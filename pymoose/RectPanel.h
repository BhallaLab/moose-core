#ifndef _pymoose_RectPanel_h
#define _pymoose_RectPanel_h
#include "PyMooseBase.h"
#include "Panel.h"
namespace pymoose{
    class RectPanel : public Panel
    {      public:
        static const std::string className_;
        RectPanel(Id id);
        RectPanel(std::string path);
        RectPanel(std::string name, Id parentId);
        RectPanel(std::string name, PyMooseBase& parent);
        RectPanel( const RectPanel& src, std::string name, PyMooseBase& parent);
        RectPanel( const RectPanel& src, std::string name, Id& parent);
        RectPanel( const RectPanel& src, std::string path);
        RectPanel( const Id& src, std::string name, Id& parent);
        RectPanel( const Id& src, std::string path);
        ~RectPanel();
        const std::string& getType();
    };
}
#endif
