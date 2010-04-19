#ifndef _pymoose_Neutral_cpp
#define _pymoose_Neutral_cpp
#include "Neutral.h"
using namespace pymoose;
const std::string Neutral::className_ = "Neutral";
Neutral::Neutral(Id id):PyMooseBase(id){}
Neutral::Neutral(std::string path):PyMooseBase(className_, path){}
Neutral::Neutral(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
Neutral::Neutral(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
Neutral::Neutral(std::string path, std::string fileName):PyMooseBase(className_, path, fileName){}
Neutral::Neutral(const Neutral& src, std::string objectName,  PyMooseBase& parent):PyMooseBase(src, objectName, parent){}

Neutral::Neutral(const Neutral& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
Neutral::Neutral(const Neutral& src, std::string path):PyMooseBase(src, path)
{
}

Neutral::Neutral(const Id& src, string name, Id& parent):PyMooseBase(src, name, parent)
{
}
Neutral::Neutral(const Id& src, string path):PyMooseBase(src, path)
{
}

Neutral::~Neutral(){}
const std::string& Neutral::getType(){ return className_; }
int Neutral::__get_childSrc() const
{
    int childSrc;
    get < int > (id_(), "childSrc",childSrc);
    return childSrc;
}
void Neutral::__set_childSrc( int childSrc )
{
    set < int > (id_(), "childSrc", childSrc);
}
int Neutral::__get_child() const
{
    int child;
    get < int > (id_(), "child",child);
    return child;
}
void Neutral::__set_child( int child )
{
    set < int > (id_(), "child", child);
}
#endif
