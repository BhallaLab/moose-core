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
    static PyMooseContext* getContext();

    vector< Id >& __get_children() const;
    const Id* __get_parent() const;
    const std::string path() const;   
    const Id* __get_id() const;

    bool connect(std::string field, PyMooseBase* dest, std::string destField);
    bool connect(std::string field, Id dest, std::string destField);
    vector <std::string> getMessageList(string field, bool isIncoming );
    vector <std::string>& __get_incoming_messages();
    vector <std::string>& __get_outgoing_messages();
    
    static bool exists(Id id);
    static bool exists(string path);
    static vector <Id>& le();
    static Id pwe();
    static Id ce(Id newElement);
    static Id ce(std::string path);
    static Id pathToId(std::string path, bool echo = true);
    static const string idToPath(Id id);
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
    PyMooseBase(Id id);    /// This is for wrapping an existing ID inside an object
    PyMooseBase(std::string className, std::string path, std::string fileName); /// this will use readcell - since we do not know how exactly all future classes will be loaded from file, we make it protected and those classes should provide the actual implementation
    
  private:
    static PyMooseBase* root_;    
    static string  separator_;
    static PyMooseContext* context_;
    vector <std::string> incomingMessages_;
    vector <std::string> outgoingMessages_;
};
#endif // _PYMOOSE_BASE_H

