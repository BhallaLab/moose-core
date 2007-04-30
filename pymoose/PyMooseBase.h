/*******************************************************************
 * File:            Base.h
 * Description:      
 * Author:          Subhasis Ray / NCBS
 * Created:         2007-03-10 18:24:02
 ********************************************************************/

#ifndef _PYMOOSE_BASE_H
#define _PYMOOSE_BASE_H

#include "PyMooseContext.h"

class PyMooseBase
{
  public:  
    PyMooseBase(std::string className, std::string objectName, Id parentId);
    PyMooseBase(std::string className, std::string path);    
    PyMooseBase(std::string className, std::string objectName, PyMooseBase* parent);
    
    virtual ~PyMooseBase();
    static bool destroy(Id id);    
    static void endSimulation();    
    virtual const std::string& getType() = 0;
    const string& getSeparator() const;
    static const PyMooseContext* getContext();

    vector< Id >& __get_children() const;
    Id __get_parent() const;
    std::string path() const;   
    Id __get_id() const;
    

    bool connect(std::string field, PyMooseBase* dest, std::string destField);
    bool connect(std::string field, Id dest, std::string destField);
    static vector <Id>& le();
    static Id pwe();
    static Id ce(Id newElement);
    static Id ce(std::string path);
    static Id pathToId(std::string path);
    static string idToPath(Id id);    
    static Id getParent(Id id);
    static vector <Id>& getChildren(Id id);    
    static void initSimulation();
    
// Think about this - are we going to allow people to access objects by their ID?
// If we do, that breaks the idea of interpreter doing the object lifetime management
// gets broken. If we don't, user can create objects and re-assign to same variable,
// - and they will vanish into oblivion.
//    static PyMooseBase* getObjectById(Id id);
    
  protected:
    Id id_;
    PyMooseBase(Id id);    // This is for wrapping an existing ID inside an object
  private:
    static PyMooseBase* root_;    
    static string  separator_;
    static PyMooseContext* context_;
};
#endif // _PYMOOSE_BASE_H

