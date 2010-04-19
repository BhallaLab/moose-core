#ifndef _pymoose_NeuroHub_h
#define _pymoose_NeuroHub_h
#include "PyMooseBase.h"
namespace pymoose{
    class NeuroHub : public PyMooseBase
    {      public:
        static const std::string className_;
        NeuroHub(Id id);
        NeuroHub(std::string path);
        NeuroHub(std::string name, Id parentId);
        NeuroHub(std::string name, PyMooseBase& parent);
        NeuroHub( const NeuroHub& src, std::string name, PyMooseBase& parent);
        NeuroHub( const NeuroHub& src, std::string name, Id& parent);
        NeuroHub( const NeuroHub& src, std::string path);
        NeuroHub( const Id& src, std::string name, Id& parent);
	NeuroHub( const Id& src, std::string path);
        ~NeuroHub();
        const std::string& getType();
    };
}
#endif
