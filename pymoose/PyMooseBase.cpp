/*******************************************************************
 * File:            PyMooseBase.cpp
 * Description:      
 * Author:          Subhasis Ray / NCBS
 * Created:         2007-03-10 18:42:50
 ********************************************************************/
#ifndef _PYMOOSE_BASE_CPP
#define _PYMOOSE_BASE_CPP

#include "PyMooseBase.h"

string PyMooseBase::separator_ = "/";
PyMooseContext* PyMooseBase::context_ = PyMooseContext::createPyMooseContext("BaseContext", "RootShell");

PyMooseBase::PyMooseBase(Id id)
{
  Element* e = Element::element(id);
  if ( e == NULL)
  {
    cerr << "ERROR: PyMooseBase::PyMooseBase(Id id) - this Id does not exist." << endl;
    id_ = PyMooseContext::BAD_ID;    
    return;
  }
  id_ = id;
}
PyMooseBase::PyMooseBase(std::string className, std::string objectName, Id parentId )
{
    id_ = context_->create(className, objectName, parentId);    
}

PyMooseBase::PyMooseBase(std::string className, std::string path)
{
    if ( path == getSeparator() || path == (getSeparator()+"root"))
    {
        cerr << "ERROR: PyMooseBase::PyMooseBase(std::string className, std::string path) -  Attempt to create predefined root element." << endl;
        return;        
    }
    assert(path.length() > 0);
    
    std::string::size_type name_start =path.rfind(getSeparator());
    if (name_start == std::string::npos)
    {
        name_start = 0;
    }
    
    std::string myName = path.substr(name_start);
    std::string parentPath = path.substr(0, name_start);
    id_ = context_->create(className, myName, context_->pathToId(parentPath));
}

PyMooseBase::PyMooseBase(std::string className, std::string objectName, PyMooseBase* parent)
{
    id_ = context_->create(className, objectName, parent->id_);
}
/**
   We do not want the objects to be really destroyed until the
   simulation is over and user asks for destruction explicitly.  So if
   we called the destroy method here, we would have had problem when
   the reference count for something in python wen to zero, e.g.,
   
   c = moose.Compartment("test")
   for i in range(10):
        d = moose.Compartment("test_"+str(i), c)

   This results in the destructor being called in each iteration of
   loop and only the 10-th child survives. Hence we move the actual
   destruction to a separate method which is supposed to be called
   explicitly at the end of simulation.
*/
PyMooseBase::~PyMooseBase()
{
    context_->destroy(id_);
//    cerr << "Destructor called for " << id_ << endl;    
    id_ = PyMooseContext::BAD_ID;    
}

const std::string& PyMooseBase::getSeparator() const
{
    return context_->separator;    
}
std::string PyMooseBase::__get_path() const
{
    return context_->getPath(id_);
}

Id PyMooseBase::__get_id() const
{
    return id_;
}

Id PyMooseBase::__get_parent() const 
{
    return context_->getParent(id_);
}

vector <Id>& PyMooseBase::__get_children() const
{
    return context_->getChildren(id_);
}
/**
  This method is for reinitializing simulation once
  PyMooseBase::endSimulation() method has been called and
  PyMooseBase::context_ has become invalid. Otherwise it does not do
  anything.
 */
void PyMooseBase::initSimulation()
{
    if (context_== NULL)
    {
        context_ = PyMooseContext::createPyMooseContext("BaseContext", "RootShell");
    }    
}
/**
   This method destroys all objects that have been created under the
   current context and deletes the context itself
*/
void PyMooseBase::endSimulation()
{
    context_->end();
    // \NOTE: not sure whether we should handle the destruction of context object inside python
//    PyMooseContext::destroyPyMooseContext(context_);    
//    context_ = NULL;    
}

// These functions break the Object-oriented design.
// But these are here for facility of the user and to handle
// orphans and the root element
Id PyMooseBase::getParent(Id id)
{
    return context_->getParent(id);
}

vector <Id>& PyMooseBase::getChildren(Id id)
{
    return context_->getChildren(id);
}

vector <Id>& PyMooseBase::le()
{
    return context_->getChildren(context_->getCwe());
}

Id PyMooseBase::pwe()
{
    return context_->getCwe();
}

Id PyMooseBase::ce(Id newElement)
{
    context_->setCwe(newElement);
    return newElement;    
}

bool PyMooseBase::destroy(Id id)
{
    return context_->destroy(id);    
}

#endif
