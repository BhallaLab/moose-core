#ifndef _pymoose_Neutral_h
#define _pymoose_Neutral_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Neutral : public PyMooseBase
    {    public:
        static const std::string className_;
        Neutral(std::string className, std::string objectName, Id parentId);
        Neutral(std::string className, std::string path);
        Neutral(std::string className, std::string objectName, PyMooseBase& parent);        
        Neutral(Id id);
        Neutral(std::string path);
        Neutral(std::string name, Id parentId);
        Neutral(std::string name, PyMooseBase& parent);
        Neutral(const Neutral& src, std::string name, PyMooseBase& parent);
        Neutral(const Neutral& src, std::string name, Id& parent);
        Neutral(const Id& src, std::string name, Id& parent);
        Neutral(const Neutral& src,std::string path);
        Neutral(const Id& src,std::string path);
    
        ~Neutral();
        const std::string& getType();
        const string&  __get_name() const;
            void __set_name(string name);
            int __get_index() const;
            const Id* __get_parent() const;
            const string&  __get_class() const;
            const vector<Id>& __get_childList() const;
            vector<Id> children(string path=".", bool ordered=true);
            unsigned int __get_node() const;
            double __get_cpu() const;
            unsigned int __get_dataMem() const;
            unsigned int __get_msgMem() const;
            const vector < string >& __get_fieldList() const;
    };
}

#endif
