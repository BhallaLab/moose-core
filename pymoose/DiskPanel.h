#ifndef _pymoose_DiskPanel_h
#define _pymoose_DiskPanel_h
#include "PyMooseBase.h"
#include "Panel.h"

namespace pymoose{
    class DiskPanel : public Panel
    {      public:
        static const std::string className_;
        DiskPanel(Id id);
        DiskPanel(std::string path);
        DiskPanel(std::string name, Id parentId);
        DiskPanel(std::string name, PyMooseBase& parent);
        DiskPanel( const DiskPanel& src, std::string name, PyMooseBase& parent);
        DiskPanel( const DiskPanel& src, std::string name, Id& parent);
        DiskPanel( const DiskPanel& src, std::string path);
        DiskPanel( const Id& src, std::string name, Id& parent);
	DiskPanel( const Id& src, std::string path);
        ~DiskPanel();
        const std::string& getType();
    };
}
#endif
