#ifndef _pymoose_Neutral_h
#define _pymoose_Neutral_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Neutral : public PyMooseBase
    {    public:
        static const std::string className_;
        Neutral(Id id);
        Neutral(std::string path);
        Neutral(std::string name, Id parentId);
        Neutral(std::string name, PyMooseBase& parent);
        Neutral(std::string path, std::string fileName);
        Neutral(const Neutral& src,std::string name, PyMooseBase& parent);
        Neutral(const Neutral& src,std::string name, Id& parent);
        Neutral(const Id& src,std::string name, Id& parent);
        Neutral(const Neutral& src,std::string path);
        Neutral(const Id& src,std::string path);
    
        ~Neutral();
        const std::string& getType();
        int __get_child() const;
        void __set_child(int child);
    };
}

#endif
