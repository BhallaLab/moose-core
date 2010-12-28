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

/**
   returns the children of this object.

   \param path -- a string pattern for the child path. It can be left
   empty - when all the elements directly under the current element
   will be returned. If it is a wild card, all descendants matching
   this wild card will be returned.
   In genesis/moose convention the following wildcards can be noted:

   "#" - matches all the direct discendants
   "##" - matches all descendants
   "#[TYPE={type}] - matches all direct descendants of the specified
   type, e.g., "#[TYPE=Table]" will match all elements of Table class
   directly under the current element.
   "##[TYPE={type}] - matches all descendants of of class {type}.

   Generally, "#{rule}" will return a list of direct descndants that
   match rule and "##{rule}" will recursively reaverse the element
   tree and return all descendants that match rulr.
   
   "#[Class={type}]" is synonymous with "#[TYPE={type}]"
   "##[Class={typr}]" is synonymous with "##[TYPE={type}]"
   
   \param ordered -- whether the child list should be sorted. Default
   is true. If false, the list is sorted in terms of the pointers -
   creating an apparently random list.
 */
vector<Id> Neutral::children(string path, bool ordered)
{
    vector<Id> childList;
    if (path.length() > 0){
        if (ordered &&
            ((path[0] == '.') && ((path.length() == 2 && path[1] == '/') || (path.length() == 1)))){
            get < vector<Id> > (id_(), "childList",childList);
        } else {
            string new_path = this->__get_path() + (path[0] == '/'? "": "/") + path;
            childList = getContext()->getWildcardList(path, ordered);
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
