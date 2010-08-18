#ifndef _pymoose_HHChannel2D_h
#define _pymoose_HHChannel2D_h
#include "PyMooseBase.h"
#include "HHChannel.h"

namespace pymoose{
class HHChannel;
    class HHChannel2D : public HHChannel
    {      public:
        static const std::string className_;
        HHChannel2D(Id id);
        HHChannel2D(std::string path);
        HHChannel2D(std::string name, Id parentId);
        HHChannel2D(std::string name, PyMooseBase& parent);
        HHChannel2D( const HHChannel2D& src, std::string name, PyMooseBase& parent);
        HHChannel2D( const HHChannel2D& src, std::string name, Id& parent);
        HHChannel2D( const HHChannel2D& src, std::string path);
        HHChannel2D( const Id& src, std::string name, Id& parent);
	HHChannel2D( const Id& src, std::string path);
        ~HHChannel2D();
        const std::string& getType();
            string __get_Xindex() const;
            void __set_Xindex(string Xindex);
            string __get_Yindex() const;
            void __set_Yindex(string Yindex);
            string __get_Zindex() const;
            void __set_Zindex(string Zindex);
    };
}
#endif
