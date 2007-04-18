#ifndef _pymoose_Neutral_h
#define _pymoose_Neutral_h
#include "PyMooseBase.h"
class Neutral : public PyMooseBase
{    public:
        static const std::string className;
        Neutral(Id id);
        Neutral(std::string path);
        Neutral(std::string name, Id parentId);
        Neutral(std::string name, PyMooseBase* parent);
        ~Neutral();
        const std::string& getType();
        int __get_childSrc() const;
        void __set_childSrc(int childSrc);
        int __get_child() const;
        void __set_child(int child);
};
#endif
