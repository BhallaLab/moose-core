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

#include "basecode/header.h"
#include "basecode/moose.h"
#include "shell/Shell.h"

#include "PyMooseUtil.h"

namespace pymoose
{
const float version = 3.0;
const string revision = SVN_REVISION;

/// Enumerations for field types.
   
enum FieldType { FTYPE_ALL, /// Wildcard to match all field types.
                 FTYPE_VALUE, /// Value field
                 FTYPE_LOOKUP, /// Lookup field - such as a lookup table
                 FTYPE_SOURCE, /// Source field - used for connecting to a destination field
                 FTYPE_DEST, /// Destination field - source field connects here
                 FTYPE_SHARED, /// shared field - source and
                               /// destination combined - connects to another shared field
                 FTYPE_SOLVE, /// solve field - special purpose field for solvers
                 FTYPE_THIS,  /// special purpose field
                 FTYPE_GLOBAL, /// special purpose field
                 FTYPE_DEL /// special purpose field
};
/// Enumeration for specifying the trigMode of PulseGen objects
enum { FREE_RUN, /// generate periodic pulses - no need for input
       EXT_TRIG, /// generate a pulse only if input switches from zero to nonzero
       EXT_GATE /// generate periodic pulses as long as input is high.
};

/// Enumeration for message direction
enum { OUTGOING, /// Messages for which the current object is a source.
       INCOMING, /// Messages for which current object is a destination.
       INOUT /// Union of OUTGOING and INCOMING messages.
};
    class PyMooseContext
    {
      public:
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
        Id create(std::string type, std::string name, Id parent=Id::badId());
        bool destroy(Id victim);
        void end();    
        // Receive functions
        static void recvCwe( const Conn* c, Id i );
        static void recvElist( const Conn* c, std::vector< Id > elist );
        static void recvCreate( const Conn* c, Id i );
        static void recvField( const Conn* c, std::string value );
        static void recvWildcardList( const Conn* c,
                                      std::vector< Id > value );
    
        static void recvClocks( const Conn* c, std::vector< double > dbls);
        static void recvMessageList( 
            const Conn* c, std::vector< Id > elist, std::string s);
        
        static PyMooseContext* createPyMooseContext(std::string contextName, std::string shellName);
        static void destroyPyMooseContext(PyMooseContext* context);
        void loadG(std::string script); /// load a GENESIS script file
        void runG(std::string statement); /// run a GENESIS statement
        
        const std::string& getField(Id, std::string);
        void setField(Id, std::string, std::string);
        std::vector <std::string> getMessageList(Id obj, std::string field, bool incoming);
        std::vector <std::string> getMessageList(Id obj, bool incoming);
        
        const Id& getParent(Id id) const;
        const std::string& getPath(Id id) const;
        const std::string& getName(Id id) const;
        const std::vector <Id>& getChildren(Id id);
        const std::vector <Id>& getChildren(std::string path);
        const std::vector <Id> & getWildcardList(std::string path, bool ordered);
        Id pathToId(std::string path, bool echo = true);
        /// set the seed for random number generator
        static void srandom(long seed);
        
        void step(double runTime);
        void step(long multiple);
        void step();
        void reset();
        void stop();
        void setClock(int clockNo, double dt, int stage = 0);    
        std::vector <double>& getClocks();
        void useClock(const std::string& tickName, const std::string& path, const std::string& func = "process");
        void useClock(int tickNo, const std::string& path, const std::string& func = "process");
        void addTask(std::string arg);
        void do_deep_copy( const Id& object, const Id& dest, std::string new_name);
        void copy(const Id& src, const Id& dest_parent, std::string new_name);
        Id deepCopy( const Id& object, const Id& dest, std::string new_name);    
        void move( const Id& object, const Id& dest, std::string new_name);
        void move( string src, string dest, std::string new_name);
        bool connect(const Id& src, std::string srcField, const Id& dest, std::string destField);
        void setupAlpha( std::string channel, std::string gate, std::vector <double> parms );
        void setupAlpha(std::string channel, std::string gate, double AA, double AB, double AC, double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max);
        void setupTau( std::string channel, std::string gate, std::vector <double> parms );
        void setupTau(std::string channel, std::string gate, double AA, double AB, double AC, double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max);
        void tweakAlpha( std::string channel, std::string gate );
        void tweakTau( std::string channel, std::string gate);
        void tabFill(const Id& table, int xdivs, int mode);    
        const vector <double>& getTableVector(const Id& table);
        void setupAlpha( const Id& gateId, std::vector <double> parms );
        void setupTau( const Id& gateId, std::vector <double> parms );
        void tweakAlpha( const Id& gateId );
        void tweakTau( const Id& gateId);
        void readCell(std::string filename, std::string cellpath, double cm, double rm, double ra, double erestAct, double eleak);
        void readCell(std::string filename, std::string cellpath, std::vector <double> params);        
        void readCell(std::string fileName, std::string cellPath);
        void readSBML(std::string fileName, std::string modelPath);
        void readNeuroML(std::string fileName, std::string modelPath);
        double getCurrentTime();
        bool exists(const Id& id);
        bool exists(std::string path);
        void addField(std::string objectPath, std::string fieldName);
        void addField(Id objectId, std::string fieldName);
        void createMap(std::string src, std::string dest, unsigned int nx, unsigned int ny, double dx = 1.0, double dy = 1.0, double xo = 0.0, double yo = 0.0, bool isObject = true);
        void createMap( Id src,  Id dest, std::string name, unsigned int nx, unsigned int ny, double dx = 1.0, double dy = 1.0, double xo = 0.0, double yo = 0.0);
        void createMap( Id src, Id dest, std::string name, std::vector<double> param);
        void planarConnect(std::string src, std::string dst, double probability = 1.0);
        void plannarDelay(std::string src, double delay);
        void planarWeight(std::string src, double weight);
        const std::string& className(const Id& objId) const;
        const std::string& description(const std::string className) const;
        const std::string& author(const std::string className) const;
        const std::string& doc(const std::string& className) const;
        const vector<Id>& getNeighbours(Id object, std::string fieldName="*", int direction=INCOMING);
        const vector <string>& getValueFieldList(Id id);
        const vector<string>& getFieldList(Id id, FieldType ftype=FTYPE_ALL);        
#ifdef DO_UNIT_TESTS    
        static bool testPyMooseContext(int count, bool print);
        bool testCreateDestroy(std::string className, int count, bool print);
        bool testSetGetField(std::string className, std::string fieldName, std::string value, int count, bool print);
        bool testSetGetField(int count, bool print);
#endif    // DO_UNIT_TESTS
        static const std::string separator;
        bool parallel;
      private:
        Id findChanGateId( std::string channel, std::string gate);
        void setupChanFunc( std::string channel, std::string gate, std::vector <double>& parms, const Slot& slot);
    
        void setupChanFunc(std::string channel, std::string gate, double AA, double AB, double AC, double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max, const Slot& slot);

    
        void tweakChanFunc( std::string channel, std::string gate, const Slot& slot );
    
        void setupChanFunc( const Id& gateId, std::vector <double> parms, const Slot& slot);
        void tweakChanFunc( const Id& gateId, const Slot& slot );
        bool parseCopyMove( std::string src, std::string dest,  Id s,  Id& e, Id& pa, std::string& childname );
        
        Id myId_;    
        // Present working element
        Id cwe_;
        Id returnId_;

        Id createdElm_;
        std::vector< Id > elist_;
        // We should avoid this - as it gets field values as std::std::string
        // for easier printing, whereas we can allow python to handle the
        // display
        mutable std::string fieldValue_;
        std::vector< std::string> strings_;
        std::vector< double > dbls_;
        // We may need the shell
        Id shell_;
        Id scheduler_;
        Id clockJob_;
        Element* genesisSli_;
        const Finfo* genesisParseFinfo_;
        
    };
} // namespace pymoose

#endif
