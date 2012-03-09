// pymooseutil.cpp --- 
// 
// Filename: startfinish.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Sat Mar 26 22:41:37 2011 (+0530)
// Version: 
// Last-Updated: Fri Mar  9 15:25:43 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 371
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
// This is for utility functions to initialize and finalize PyMOOSE
//
// 2011-08-25 15:27:10 (+0530) - It has now eveolved into a dumping
// ground for all utility functions for pymoose.

// Code:

#include <string>
#include <map>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../basecode/header.h"
#include "../basecode/Id.h"
#include "../shell/Shell.h"
#include "../utility/utility.h"
#include "../scheduling/Tick.h"
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"
#include "pymoose.h"

using namespace pymoose;

extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size, unsigned int numThreads,
	unsigned int method );
extern void testShell();
extern void testScheduling();
extern void testSchedulingProcess();
extern void testBuiltins();
extern void testBuiltinsProcess();

extern void testMpiScheduling();
extern void testMpiBuiltins();
extern void testMpiShell();
extern void testMsg();
extern void testMpiMsg();
extern void testKinetics();
extern void nonMpiTests(Shell *);
extern void mpiTests();
extern void processTests( Shell* );
extern Id init(int argc, char ** argv);

extern void initMsgManagers();
extern void destroyMsgManagers();
extern void speedTestMultiNodeIntFireNetwork( 
	unsigned int size, unsigned int runsteps );
#ifdef DO_UNIT_TESTS
extern void regressionTests();
#endif
extern bool benchmarkTests( int argc, char** argv );
extern int getNumCores();

int isSingleThreaded = 0;
int isInfinite = 0;
int numNodes = 1;
int numCores = 1;
int myNode = 0;
int numProcessThreads = 0;
static bool quitFlag = 0;
static Element* shellE = NULL; // This is in order to keep a handle on
                               // the original shell element - I don't
                               // know how to get back the Id of
                               // stupid shell from the Shell&.

    /// Available Finfo types
const char ** getFinfoTypes()
{
    static const char * finfoTypes[] = {"valueFinfo",
                                        "srcFinfo",
                                        "destFinfo",
                                        "lookupFinfo",
                                        "sharedFinfo",
                                        "fieldElementFinfo",
                                        0};
    return finfoTypes;
}

void pymoose::setup_runtime_env(bool verbose){
    const map<string, string>& argmap = getArgMap();
    map<string, string>::const_iterator it;
    it = argmap.find("SINGLETHREADED");
    if (it != argmap.end()){
        istringstream(it->second) >> isSingleThreaded;
    }
    it = argmap.find("INFINITE");
    if (it != argmap.end()){
        istringstream(it->second) >> isInfinite;
    }
    it = argmap.find("NUMNODES");
    if (it != argmap.end()){
        istringstream(it->second) >> numNodes;
    }
    it = argmap.find("NUMCORES");
    if (it != argmap.end()){
        istringstream(it->second) >> numCores;
    }
    it = argmap.find("NUMPTHREADS");
    if (it != argmap.end()){
        istringstream(it->second) >> numProcessThreads;
    }
    it = argmap.find("QUIT");
    if (it != argmap.end()){
        istringstream(it->second) >> quitFlag;
    }
    if (verbose){
        cout << "ENVIRONMENT: " << endl
             << "----------------------------------------" << endl
             << "   SINGLETHREADED = " << isSingleThreaded << endl
             << "   INFINITE = " << isInfinite << endl
             << "   NUMNODES = " << numNodes << endl
             << "   NUMCORES = " << numCores << endl
             << "   NUMPTHREADS = " << numProcessThreads << endl
             << "========================================" << endl;
    }
}

Shell& pymoose::getShell(int argc, char ** argv)
{
    static Shell * shell_ = NULL;
    if (shell_ != NULL){
        return *shell_;
    }
        
    // Set up the system parameters
    int _isSingleThreaded = 0;
    int _numCores = 1;
    int _numNodes = 1;
    int _isInfinite = 0;
    int _myNode = 0;
    int _numProcessThreads = 0;
    vector<string> args;
    map<string, string>::const_iterator it = getArgMap().find("SINGLETHREADED");    
    if (it != getArgMap().end()){
        args.push_back("s");
        istringstream(it->second) >> _isSingleThreaded;
    }
    it = getArgMap().find("NUMCORES");
    if ((it == getArgMap().end()) || it->second.empty()){
        _numCores = getNumCores();
    } else {
        istringstream(it->second) >> _numCores;
    }
    it = getArgMap().find("NUMNODES");
    if (it != getArgMap().end()){
        istringstream(it->second) >> _numNodes;
        args.push_back("n");
        args.push_back(it->second);
    }
    it = getArgMap().find("INFINITE");
    if (it != getArgMap().end()){
        args.push_back("i");
        istringstream(it->second) >> _isInfinite;
    }
    it = getArgMap().find("NUMPTHREADS");
    if (it != getArgMap().end()){
        args.push_back("t");
        args.push_back(it->second);
        istringstream(it->second) >> _numProcessThreads;
    } else {
        _numProcessThreads = _numCores;
    }
    if (_numProcessThreads == 0){
        _isSingleThreaded = 1;
    }
    cout << "================================================" << endl
         << "Final system parameters:" << endl
         << " SINGLETHREADED: " << _isSingleThreaded << endl
         << " NUMNODES: " << _numNodes << endl
         << " NUMCORES: " << _numCores << endl
         << " NUMPTHREADS: " << _numProcessThreads << endl
         << " INFINITE: " << _isInfinite << endl
         << "================================================" << endl;
        
    char ** argv_new = new char*[args.size()];
    for (unsigned ii = 0; ii < args.size(); ++ii){
        argv_new[ii] = new char[args[ii].length() + 1];
        strncpy(argv_new[ii], args[ii].c_str(), args[ii].length()+1);
    }
    // Utilize the main::init function which has friend access to Id
    Id shellId = init(argc, argv_new);
    shell_ = reinterpret_cast<Shell*>(shellId.eref().data());
    shellE = shellId();
    for (unsigned ii = 0; ii < args.size(); ++ii){
        delete [] argv_new[ii];
    }
    delete [] argv_new;
#ifdef DO_UNIT_TESTS        
    nonMpiTests( shell_ ); // These tests do not need the process loop.
#endif // DO_UNIT_TESTS
    if (!shell_->isSingleThreaded()){
        Qinfo::initMutex(); // Mutex used to align Parser and MOOSE threads.
        shell_->launchThreads();
    }
    if ( shell_->myNode() == 0 ) {
#ifdef DO_UNIT_TESTS
        mpiTests();
        processTests( shell_ );
        regressionTests();
#endif
        if ( benchmarkTests( argc, argv ) || quitFlag ){
            shell_->doQuit();
        }
    }
    return *shell_;
}

void pymoose::finalize()
{
    getShell().doQuit();
    if (!getShell().isSingleThreaded()){
        getShell().joinThreads();
        Qinfo::freeMutex();
    }
    // getShell().clearSetMsgs();
    Neutral* ns = reinterpret_cast<Neutral*>(shellE->dataHandler()->data(0));
    ns->destroy( shellE->id().eref(), 0, 0);
#ifdef USE_MPI
    MPI_Finalize();
#endif

}

pair < string, string > pymoose::getFieldFinfoTypePair(ObjId id, string fieldName)
{
    for (const char ** finfoType = getFinfoTypes(); *finfoType; ++finfoType){
        string ftype = pymoose::getFieldType(id, fieldName, string(*finfoType));
        if (!ftype.empty()) {
            return pair < string, string > (ftype, string(*finfoType));
        }
    }
    return pair <string, string>("", "");
}
/**
   Return the data type of the field. Look up a field of specified
   finfoType with given fieldName.  Return empty string on failure
   (either there is no field of name {fieldName} or {finfoType} is not a
   correct type of finfo, or no field of {finfoType} with name
   {fieldName} exists.
*/
string pymoose::getFieldType(ObjId id, string fieldName, string finfoType)
{
    string fieldType = "";
    string className = Field<string>::get(id, "class");
    string classInfoPath("/classes/" + className);
    Id classId(classInfoPath);
    assert (classId != Id());
    unsigned int numFinfos = Field<unsigned int>::get(ObjId(classId, 0), "num_" + finfoType);
    Id fieldId(classId.path() + "/" + finfoType);
    for (unsigned int ii = 0; ii < numFinfos; ++ii){
        string _fieldName = Field<string>::get(ObjId(fieldId, DataId(0, ii, 0)), "name");
        if (fieldName == _fieldName){                
            fieldType = Field<string>::get(ObjId(fieldId, DataId(0, ii, 0)), "type");
            return fieldType;
        }
    }
    cerr << "Error: No field named '" << fieldName << "' of type '" << finfoType << "'" << endl;
    return fieldType;
}

/**
   Return a vector of field names of specified finfo type.
 */
vector<string> pymoose::getFieldNames(ObjId id, string finfoType)
{
    vector <string> ret;
    string className = Field<string>::get(id, "class");    
    Id classId("/classes/" + className);
    assert(classId != Id());
    unsigned int numFinfos = Field<unsigned int>::get(ObjId(classId), "num_" + finfoType);
    Id fieldId(classId.path() + "/" + finfoType);
    if (fieldId == Id()){
        return ret;
    }
    for (unsigned int ii = 0; ii < numFinfos; ++ii){
        string fieldName = Field<string>::get(ObjId(fieldId, DataId(0, ii, 0)), "name");
        ret.push_back(fieldName);
    }
    return ret;
}

// 
// pymooseutil.cpp ends here
