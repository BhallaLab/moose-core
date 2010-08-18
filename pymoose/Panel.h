#ifndef _pymoose_Panel_h
#define _pymoose_Panel_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class Panel : public Neutral
    {      public:
        static const std::string className_;
        Panel(Id id);
        Panel(std::string path);
        Panel(std::string name, Id parentId);
        Panel(std::string name, PyMooseBase& parent);
        Panel( const Panel& src, std::string name, PyMooseBase& parent);
        Panel( const Panel& src, std::string name, Id& parent);
        Panel( const Panel& src, std::string path);
        Panel( const Id& src, std::string name, Id& parent);
        Panel( const Id& src, std::string path);
        
	Panel(std::string typeName, std::string objectName, Id parentId);
        Panel(std::string typeName, std::string path);
	Panel(std::string typeName, std::string objectName, PyMooseBase& parent);
        

        ~Panel();
        const std::string& getType();
            unsigned int __get_nPts() const;
            unsigned int __get_nDims() const;
            unsigned int __get_nNeighbors() const;
            unsigned int __get_shapeId() const;
            vector < double > __get_coords() const;
    };
}
#endif
