#ifndef _pymoose_Neutral_cpp
#define _pymoose_Neutral_cpp
#include "Neutral.h"
const std::string Neutral::className = "Neutral";
Neutral::Neutral(Id id):PyMooseBase(id){}
Neutral::Neutral(std::string path):PyMooseBase(className, path){}
Neutral::Neutral(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Neutral::Neutral(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Neutral::~Neutral(){}
const std::string& Neutral::getType(){ return className; }
int Neutral::__get_childSrc() const
{
    int childSrc;
    get < int > (Element::element(id_), "childSrc",childSrc);
    return childSrc;
}
void Neutral::__set_childSrc( int childSrc )
{
    set < int > (Element::element(id_), "childSrc", childSrc);
}
int Neutral::__get_child() const
{
    int child;
    get < int > (Element::element(id_), "child",child);
    return child;
}
void Neutral::__set_child( int child )
{
    set < int > (Element::element(id_), "child", child);
}
#endif
