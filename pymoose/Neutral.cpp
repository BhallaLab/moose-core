#ifndef _pymoose_Neutral_cpp
#define _pymoose_Neutral_cpp
#include "Neutral.h"
using namespace pymoose;
const std::string Neutral::className_ = "Neutral";
Neutral::Neutral(std::string className, std::string objectName, Id parentId):PyMooseBase(className, objectName, parentId){}
Neutral::Neutral(std::string className, std::string path): PyMooseBase(className, path){}
Neutral::Neutral(std::string className, std::string objectName, PyMooseBase& parent):PyMooseBase(className, objectName, parent){}
Neutral::Neutral(Id id):PyMooseBase(id){}
Neutral::Neutral(std::string path):PyMooseBase(className_, path){}
Neutral::Neutral(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
Neutral::Neutral(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}

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

vector<Id> Neutral::children(string path=".", bool breadthFirst=true)
{
    vector<Id> childList;
    if (path.length() > 0){
        if (breadthFirst &&
            ((path[0] == '.') && ((path.length() == 2 && path[1] == '/') || (path.length() == 1)))){
            get < vector<Id> > (id_(), "childList",childList);
        } else {
            childList = getContext()->getWildcardList(path, breadthFirst);
        }
    }
    return childList;
}
string  Neutral::__get_name() const
{
return this->getField("name");
}
void Neutral::__set_name( string name )
{
    set < string > (id_(), "name", name);
}
int Neutral::__get_index() const
{
    int index;
    get < int > (id_(), "index",index);
    return index;
}
const Id* Neutral::__get_parent() const
{
    return &(getContext()->getParent(id_));
}
const string&  Neutral::__get_class() const
{
    return getContext()->className(id_);
}
const vector<Id>& Neutral::__get_childList() const
{
    return getContext()->getChildren(id_);
}
unsigned int Neutral::__get_node() const
{
    unsigned int node;
    get < unsigned int > (id_(), "node",node);
    return node;
}
double Neutral::__get_cpu() const
{
    double cpu;
    get < double > (id_(), "cpu",cpu);
    return cpu;
}
unsigned int Neutral::__get_dataMem() const
{
    unsigned int dataMem;
    get < unsigned int > (id_(), "dataMem",dataMem);
    return dataMem;
}
unsigned int Neutral::__get_msgMem() const
{
    unsigned int msgMem;
    get < unsigned int > (id_(), "msgMem",msgMem);
    return msgMem;
}
const vector < string >& Neutral::__get_fieldList() const
{
    return getContext()->getValueFieldList(id_);
}
#endif
