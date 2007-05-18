/*******************************************************************
 * File:            PyMooseContext.h
 *
 * Description:
 *
 *      PyMooseContext is the class to hold the global context for
 *      pymoose. Similar to GenesisParserWrapper in genesis
 *      interpreter. This class was created to maintain the data that
 *      were not found fit for going into the PyMooseBaseClass. The
 *      base class should have only those info required by and common
 *      to the Moose classes. It should not be dirtied by the details
 *      of interaction with Shell class.
 *
 * Author:          Subhasis Ray / NCBS
 * Created:         2007-03-12 03:15:09
 ********************************************************************/
#ifndef _PYMOOSE_CONTEXT_H
#define _PYMOOSE_CONTEXT_H

typedef unsigned int Id;
#include "../basecode/header.h"
#include "../basecode/moose.h"
#include "../shell/Shell.h"

#include "PyMooseUtil.h"

class PyMooseContext
{
  public:

    static const Id BAD_ID;
    
    PyMooseContext();
    ~PyMooseContext();
    /**
       @returns current working element - much like unix pwd
    */
    Id getCwe();
    
    /**
       @param Id elementId: Id of the new working element.
       Sets the current working element - effectively unix cd
    */
    void setCwe(Id elementId);
    
    void setCwe(std::string path);
    
    /**
       @returns Id of the Shell instance
    */
    Id getShell();
    Id id();
    
    /**
       @param type : ClassName of the MOOSE object to be generated.
       @param name : Name of the instance to be generated.
       @returns id of the newly generated object.
    */
    Id create(std::string type, std::string name, Id parent);
    bool destroy(Id victim);
    void end();    
    // Receive functions
    static void recvCwe( const Conn& c, unsigned int i );
    static void recvElist( const Conn& c, vector< Id > elist );
    static void recvCreate( const Conn& c, unsigned int i );
    static void recvField( const Conn& c, string value );
    static void recvWildcardList( const Conn& c,
                                  vector< unsigned int > value );
    
    static void recvClocks( const Conn& c, vector< double > dbls);
    static void recvMessageList( 
        const Conn& c, vector< Id > elist, string s);
    
    static PyMooseContext* createPyMooseContext(string contextName, string shellName);
    static void destroyPyMooseContext(PyMooseContext* context);
    std::string getField(Id, std::string);
    void setField(Id, std::string, std::string);
    Id getParent(Id id) const;
    std::string getPath(Id id) const;
    vector <Id>& getChildren(Id id);
    vector <Id>& getChildren(std::string path);    
    Id pathToId(std::string path, bool echo = true);
    void step(double runTime);
    void step(long multiple);
    void step();
    void reset();
    void stop();
    void setClock(int clockNo, double dt, int stage);    
    vector <double>& getClocks();
    void useClock(Id tickId, std::string path, std::string func = "process");
    void addTask(std::string arg);
    void do_deep_copy( Id object, std::string new_name, Id dest);
    Id deepCopy( Id object, std::string new_name, Id dest);    
    void move( Id object, std::string new_name, Id dest);
    bool connect(Id src, std::string srcField, Id dest, std::string destField);
    void setupAlpha( std::string channel, std::string gate, vector <double> parms );
    void setupTau( std::string channel, std::string gate, vector <double> parms );
    void tweakAlpha( std::string channel, std::string gate );
    void tweakTau( std::string channel, std::string gate);
    void tabFill(Id table, int xdivs, int mode);    
        
    void setupAlpha( Id gateId, vector <double> parms );
    void setupTau( Id gateId, vector <double> parms );
    void tweakAlpha( Id gateId );
    void tweakTau( Id gateId);
    bool exists(Id id);
    bool exists(std::string path);
    

#ifdef DO_UNIT_TESTS    
    static bool testPyMooseContext(int count, bool print);
    bool testCreateDestroy(std::string className, int count, bool print);
    bool testSetGetField(std::string className, std::string fieldName, std::string value, int count, bool print);
    bool testSetGetField(int count, bool print);
#endif    // DO_UNIT_TESTS
    static const std::string separator;
    
  private:
    Id findChanGateId( std::string channel, std::string gate);
    void setupChanFunc( std::string channel, std::string gate, vector <double>& parms, const unsigned int& slot);
    void tweakChanFunc( std::string channel, std::string gate, unsigned int slot );
    void setupChanFunc( Id gateId, vector <double> parms, unsigned int slot);
    void tweakChanFunc( Id gateId, unsigned int slot );
    
    Id myId_;    
    // Present working element
    Id cwe_;
    Id returnId_;

    Id createdElm_;
    vector< Id > elist_;
    // We should avoid this - as it gets field values as std::string
    // for easier printing, whereas we can allow python to handle the
    // display
    string fieldValue_;
    vector< double > dbls_;
    // We may need the shell
    Id shell_;
    Id scheduler_;
    Id clockJob_;
};
#endif
