/*******************************************************************
 * File:            Base.h
 * Description:      
 * Author:          Subhasis Ray / NCBS
 * Created:         2007-03-10 18:24:02
 ********************************************************************/

#ifndef _PYMOOSE_BASE_H
#define _PYMOOSE_BASE_H

#include "PyMooseContext.h"
namespace pymoose
{
    class PyMooseBase
    {
      public:
        PyMooseBase(std::string className, std::string objectName, Id parentId);
        PyMooseBase(std::string className, std::string path);
        PyMooseBase(std::string className, std::string objectName, PyMooseBase& parent);
        /** This does a deep copy of the source object as a child of parent and sets the name to objectName*/
        PyMooseBase(const PyMooseBase& src, std::string objectName, PyMooseBase& parent);
        PyMooseBase(const PyMooseBase& src, std::string objectName, Id& parent);
        PyMooseBase(const PyMooseBase& src, std::string path);
        PyMooseBase(const Id& src, std::string name, Id& parent);
        PyMooseBase(const Id& src, std::string path);
        
        virtual ~PyMooseBase();

        const std::string& __get_author() const;
        const std::string& __get_description() const;
        static bool destroy(Id id);    
        static void endSimulation();    
        virtual const std::string& getType() = 0;
        static const std::string& getSeparator();
        static pymoose::PyMooseContext* getContext();
        const std::string& getField(std::string name) const;
        void setField(std::string name, std::string value);
        const std::vector<std::string> getFieldList(FieldType ftype=FTYPE_ALL);
        const std::vector<Id>& neighbours(std::string msgName="*", int direction=INCOMING);
        // TODO: need a way to find the field name of the other end of
        // a message. It will be good to have the source object Id and
        // the field name.
        //        const std::map<Id, string> neighbourFields(std::string& field);
        const std::string& __get_path() const;   
        const Id* __get_id() const;
        void addField(const std::string fieldName);
//        static const std::string __get_docString() const;
        void useClock(int clockNo, string func="process");
        void useClock(Id clock, string func="process");        
        bool connect(std::string field, PyMooseBase* dest, std::string destField);
        bool connect(std::string field, Id dest, std::string destField);
        std::vector <std::string> getMessageList(string field, bool isIncoming );
        std::vector <std::string> inMessages();
        std::vector <std::string> outMessages();
    
        static bool exists(Id id);
        static bool exists(string path);
        static std::vector <Id> le();
        static Id pwe();
        static Id ce(Id newElement);
        static Id ce(std::string path);
        static Id pathToId(std::string path, bool echo = true);
        static const std::string& idToPath(Id id);
        static Id getParent(Id id);
        static std::vector < Id > getChildren(Id id);
        
        static void initSimulation();
        
// Think about this - are we going to allow people to access objects by their ID?
// If we do, that breaks the idea of interpreter doing the object lifetime management
// gets broken. If we don't, user can create objects and re-assign to same variable,
// - and they will vanish into oblivion.
//    static PyMooseBase* getObjectById(Id id);
    
      protected:
        static const std::string className_;
        Id id_;
        PyMooseBase(Id id);    /// This is for wrapping an existing ID inside an object
        PyMooseBase(std::string className, std::string path, std::string fileName); /// this will use readcell - since we do not know how exactly all future classes will be loaded from file, we make it protected and those classes should provide the actual implementation
    
      private:
        static PyMooseBase* root_;    
        static const char*  separator_;
        static pymoose::PyMooseContext* context_;
        std::vector <std::string> incomingMessages_;
        std::vector <std::string> outgoingMessages_;
    };
} // namespace pymoose
void initPyMoose();

#endif // _PYMOOSE_BASE_H

