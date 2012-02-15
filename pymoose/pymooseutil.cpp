// pymooseutil.cpp --- 
// 
// Filename: startfinish.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Sat Mar 26 22:41:37 2011 (+0530)
// Version: 
// Last-Updated: Wed Feb 15 18:46:30 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 280
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
const char * FINFOTYPES[] = {"valueFinfo",
                             "srcFinfo",
                             "destFinfo",
                             "lookupFinfo",
                             "sharedFinfo",
                             "fieldElementFinfo",
                             0};

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
    if (shell_ == NULL)
    {
     
        // Set up the system parameters
        int _isSingleThreaded = 0;
        int _numCores = 1;
        int _numNodes = 1;
        int _isInfinite = 0;
        int _myNode = 0;
        int _numProcessThreads = 0;
        string arg;

        map<string, string>::const_iterator it = getArgMap().find("SINGLETHREADED");    
        if (it != getArgMap().end()){
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
        }
        it = getArgMap().find("INFINITE");
        if (it != getArgMap().end()){
            istringstream(it->second) >> _isInfinite;
        }
        it = getArgMap().find("NUMPTHREADS");
        if (it != getArgMap().end()){
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
        // Do MPI Initialization
        // Not yet sure what should be passed on in argv
#ifdef USE_MPI
        MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
        MPI_Comm_size( MPI_COMM_WORLD, &_numNodes );
        MPI_Comm_rank( MPI_COMM_WORLD, &_myNode );
        if ( provided < MPI_THREAD_SERIALIZED && _myNode == 0 ) {
            cout << "Warning: This MPI implementation does not like multithreading: " << provided << "\n";
        }    
#endif
        Msg::initNull();
        // Initialize the shell
        Id shellId;
        vector <DimInfo> dims; // TODO: DimInfo is new in dh_branch .. confirm this is correct
        // dims.push_back(DimInfo(1));
        shellE = new Element(shellId, Shell::initCinfo(), "root", dims, 0, 1);
        Id clockId = Id::nextId();
        shell_ = reinterpret_cast<Shell*>(shellId.eref().data());
        shell_->setShellElement(shellE);
        shell_->setHardware(_numProcessThreads, _numCores, _numNodes, _myNode);
        shell_->loadBalance();
    
        // Initialize the system objects

        new Element(clockId, Clock::initCinfo(), "clock", dims, 1, 1);
        Id tickId( 2 );
        assert(tickId() != 0);
        assert( tickId.value() == 2 );
        assert( tickId()->getName() == "tick" ) ;
    
        Id classMasterId( 3 );
    
        new Element( classMasterId, Neutral::initCinfo(), "classes", dims, 1 , 1);
    
        assert ( shellId == Id() );
        assert( clockId == Id( 1 ) );
        assert( tickId == Id( 2 ) );
        assert( classMasterId == Id( 3 ) );


        /// Sets up the Elements that represent each class of Msg.
        Msg::initMsgManagers();
    
        shell_->connectMasterMsg();
    
        Shell::adopt( shellId, clockId );
        Shell::adopt( shellId, classMasterId );
    
        Cinfo::makeCinfoElements( classMasterId );
    
        while ( isInfinite ) // busy loop for debugging under gdb and MPI.
            ;
        // The following are copied from main.cpp: main()
#ifdef DO_UNIT_TESTS        
        nonMpiTests( shell_ ); // These tests do not need the process loop.
#endif    
        if (!shell_->isSingleThreaded())
            Qinfo::initMutex(); // Mutex used to align Parser and MOOSE threads.
            shell_->launchThreads();
        if ( shell_->myNode() == 0 ) {
#ifdef DO_UNIT_TESTS
            mpiTests();
            processTests( shell_ );
            regressionTests();
#endif
            // The following commented out for pymoose
            //--------------------------------------------------
            // These are outside unit tests because they happen in optimized
            // mode, using a command-line argument. As soon as they are done
            // the system quits, in order to estimate timing.
            // if ( benchmarkTests( argc, argv ) || quitFlag )
            if ( benchmarkTests( argc, argv ) || quitFlag ){
                shell_->doQuit();
            }
            // else 
            //     shell_->launchParser(); // Here we set off a little event loop to poll user input. It deals with the doQuit call too.
        }
        shell_->setCwe(0);
    } // ! if (shell_ == NULL)
    return *shell_;
}

void pymoose::finalize()
{
    if (!getShell().isSingleThreaded()){
        getShell().doQuit();
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
    for (const char ** finfoType = &FINFOTYPES[0]; *finfoType; ++finfoType){
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
