// startfinish.cpp --- 
// 
// Filename: startfinish.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Sat Mar 26 22:41:37 2011 (+0530)
// Version: 
// Last-Updated: Sat Apr  2 18:34:30 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 65
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

// Code:

#include <string>
#include <map>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../basecode/header.h"
#include "../basecode/Id.h"
#include "../basecode/ReduceBase.h"
#include "../basecode/ReduceMax.h"
#include "../shell/Shell.h"
#include "../utility/utility.h"
#include "../scheduling/Tick.h"
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"


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

extern unsigned int getNumCores();

int isSingleThreaded = 0;
int isInfinite = 0;
int numNodes = 1;
int numCores = 1;
int myNode = 0;

void setup_runtime_env(bool verbose=true){
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
    it = argmap.find("NUMCORES");
    if (it != argmap.end()){
        istringstream(it->second) >> numCores;
    }
    it = argmap.find("NUMNODES");
    if (it != argmap.end()){
        istringstream(it->second) >> numNodes;
    }
    if (verbose){
        cout << "ENVIRONMENT: " << endl
             << "   SINGLETHREADED = " << isSingleThreaded << endl
             << "   INFINITE = " << isInfinite << endl
             << "   NUMCORES = " << numCores << endl
             << "   NUMNODES = " << numNodes << endl;
    }
}

Shell& getShell()
{
    static Shell * shell_ = NULL;
    if (shell_ == NULL)
    {
     
        // Set up the system parameters
        long isSingleThreaded = 0;
        long numCores = 1;
        long numNodes = 1;
        long isInfinite = 0;
        long myNode = 0;
        string arg;

        map<string, string>::const_iterator it = getArgMap().find("SINGLETHREADED");    
        if (it != getArgMap().end()){
            istringstream(it->second) >> isSingleThreaded;
        }
        it = getArgMap().find("NUMCORES");
        if ((it == getArgMap().end()) || it->second.empty()){
            numCores = getNumCores();
        }
        it = getArgMap().find("NUMNODES");
        if (it != getArgMap().end()){
            istringstream(it->second) >> numNodes;
        }
        it = getArgMap().find("INFINITE");
        if (it != getArgMap().end()){
            istringstream(it->second) >> isInfinite;
        }

        // Do MPI Initialization
        // Not yet sure what should be passed on in argv
#ifdef USE_MPI
        int argc = 0;
        char ** argv = NULL;
        MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
        MPI_Comm_size( MPI_COMM_WORLD, &numNodes );
        MPI_Comm_rank( MPI_COMM_WORLD, &myNode );
        if ( provided < MPI_THREAD_SERIALIZED && myNode == 0 ) {
            cout << "Warning: This MPI implementation does not like multithreading: " << provided << "\n";
        }    
#endif
        Msg::initNull();
        // Initialize the shell
        Id shellId;
        vector <unsigned int> dims;
        dims.push_back(1);
        Element * shellE = new Element(shellId, Shell::initCinfo(), "root", dims, 1);
        Id clockId = Id::nextId();
        shell_ = reinterpret_cast<Shell*>(shellId.eref().data());
        shell_->setShellElement(shellE);
        shell_->setHardware(isSingleThreaded, numCores, numNodes, myNode);
        shell_->loadBalance();
    
        // Initialize the system objects

        new Element(clockId, Clock::initCinfo(), "clock", dims, 1);
        Id tickId( 2 );
        assert(tickId() != 0);
        assert( tickId.value() == 2 );
        assert( tickId()->getName() == "tick" ) ;
    
        Id classMasterId( 3 );
    
        new Element( classMasterId, Neutral::initCinfo(), "classes", dims, 1 );
    
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
        // nonMpiTests( shell_ ); // These tests do not need the process loop.
    
        if (!shell_->isSingleThreaded())
            shell_->launchThreads();
        if ( shell_->myNode() == 0 ) {
#ifdef DO_UNIT_TESTS
            // mpiTests();
            // processTests( shell_ );
            // regressionTests();
#endif
            // The following commented out for pymoose
            //--------------------------------------------------
            // These are outside unit tests because they happen in optimized
            // mode, using a command-line argument. As soon as they are done
            // the system quits, in order to estimate timing.
            // if ( benchmarkTests( argc, argv ) || quitFlag )
            //     shell->doQuit();
            // else 
            //     shell->launchParser(); // Here we set off a little event loop to poll user input. It deals with the doQuit call too.
        }
    } // ! if (shell_ == NULL)
    return *shell_;
}

void finalize()
{
    cout << "In pymoose_finalize()" << endl;
    if (!getShell().isSingleThreaded()){
        cout << "Joining threads." << endl;
        getShell().joinThreads();
    }
    cout << "Destroying Id(0)." << endl;
    Id().destroy();
    cout << "Destroying Id(1)." << endl;
    Id(1).destroy();
    cout << "Destroying Id(2)." << endl;
    Id(2).destroy();
    cout << "Destroying msgmanagers." << endl; 
    destroyMsgManagers();
#ifdef USE_MPI
    cout << "Befor MPI Finalize." << endl; 
    MPI_Finalize();
#endif
    cout << "Finished pymoose_finalize()" << endl;            

}

string getFieldType(ObjId id, string fieldName)
{
    string fieldType;
    string className = Field<string>::get(id, "class");
    string classInfoPath("/classes/" + className);
    Id classId(classInfoPath);
    assert(classId != Id());
    static vector<string> finfotypes;
    if (finfotypes.empty()){
        finfotypes.push_back("srcFinfo");
        finfotypes.push_back("destFinfo");
        finfotypes.push_back("valueFinfo");
        finfotypes.push_back("lookupFinfo");
        finfotypes.push_back("sharedFinfo");        
    }

    for (unsigned jj = 0; jj < finfotypes.size(); ++ jj){
        unsigned int numFinfos = Field<unsigned int>::get(ObjId(classId, 0), "num_" + finfotypes[jj]);
        Id fieldId(classId.path() + "/" + finfotypes[jj]);
        for (unsigned int ii = 0; ii < numFinfos; ++ii){
            string _fieldName = Field<string>::get(ObjId(fieldId, DataId(0, ii)), "name");
            if (fieldName == _fieldName){                
                fieldType = Field<string>::get(ObjId(fieldId, DataId(0, ii)), "type");
                return fieldType;
            }
        }
    }
    cerr << "Error: No such field: " << fieldName << endl;
    return fieldType;        
}

vector<string> getFieldNames(ObjId id, string finfoType)
{
    vector <string> ret;
    string className = Field<string>::get(id, "class");    
    Id classId("/classes/" + className);
    assert(classId != Id());
    unsigned int numFinfos = Field<unsigned int>::get(ObjId(classId), "num_" + finfoType);
    Id fieldId(classId.path() + "/" + finfoType);
    assert(fieldId != Id());
    for (unsigned int ii = 0; ii < numFinfos; ++ii){
        string fieldName = Field<string>::get(ObjId(fieldId, DataId(0, ii)), "name");
        ret.push_back(fieldName);
    }
    return ret;
}

// 
// startfinish.cpp ends here
