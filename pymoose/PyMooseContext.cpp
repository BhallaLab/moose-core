/*******************************************************************
 * File:            PyMooseContext.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-03-12 03:412
 ********************************************************************/
#ifndef _PYMOOSE_CONTEXT_CPP
#define _PYMOOSE_CONTEXT_CPP

#include "PyMooseContext.h"
#include "basecode/Id.h"
#include "element/Neutral.h"
#include "scheduling/Tick.h"
#include "scheduling/ClockJob.h"
#include "builtins/Interpol.h"
#include "builtins/Table.h"
#include "maindir/init.h"
#include "connections/ConnTainer.h"

#include <string>
#include <cstdio>

using namespace std;
using namespace pymoose;

extern unsigned int init(int& argc, char **& argv);
extern void initSched();
extern const Cinfo ** initCinfos();
extern const string& helpless();
extern const string& getClassDoc(const string&, const string&);
extern const string& getCommandDoc(const string&);
extern unsigned int parseNodeNum( string& name );

extern void setupDefaultSchedule(Element*, Element*, Element*);
extern Element* makeGenesisParser();
extern char* copyString(const string& s);
extern void mtseed(long);

extern const Cinfo* initShellCinfo();
extern const Cinfo* initTickCinfo();
extern const Cinfo* initClockJobCinfo();
extern const Cinfo* initTableCinfo();
extern const Cinfo* initSchedulerCinfo();



const Cinfo* initPyMooseContextCinfo()
{
	static Finfo* parserShared[] =
	{
		new SrcFinfo( "cwe", Ftype1< Id >::global(),
			"Setting cwe" ),
		new SrcFinfo( "trigCwe", Ftype0::global(),
			"Getting cwe back: First trigger a request" ),
		new DestFinfo( "recvCwe", Ftype1< Id >::global(),
					RFCAST( &PyMooseContext::recvCwe ),
					"Then receive the cwe info" ),
		new SrcFinfo( "pushe", Ftype1< Id >::global(),
			"Setting pushe. This returns with the new cwe." ),
		new SrcFinfo( "pope", Ftype0::global(),
			"Doing pope. This returns with the new cwe." ),
		new SrcFinfo( "trigLe", Ftype1< Id >::global(),
			"Getting a list of child ids: First send a request with the requested parent elm id." ),
		new DestFinfo( "recvElist", 
					Ftype1< vector< Id > >::global(), 
					RFCAST( &PyMooseContext::recvElist ),
					"Then recv the vector of child ids. This function is shared by several other messages as all "
					"it does is dump the elist into a temporary local buffer." ),
		///////////////////////////////////////////////////////////////
		// Object heirarchy manipulation functions.
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "create",
				Ftype4< string, string, int, Id >::global(),
				"Creating an object: Send out the request. "
				"args: type, name, node, parent" ),
		new SrcFinfo( "createArray",
				Ftype4< string, string, Id, vector <double> >::global(),
				"Creating an object: Recv the returned object id." ),
		new SrcFinfo( "planarconnect", Ftype3< string, string, double >::global() ),
		new SrcFinfo( "planardelay", Ftype3< string, string, vector <double> >::global() ),
		new SrcFinfo( "planarweight", Ftype3< string, string, vector<double> >::global() ),
		new SrcFinfo( "getSynCount", Ftype1< Id >::global() ),
		new DestFinfo( "recvCreate",
					Ftype1< Id >::global(),
					RFCAST( &PyMooseContext::recvCreate ) ),
		// Deleting an object: Send out the request.
		new SrcFinfo( "delete", Ftype1< Id >::global() ),

		///////////////////////////////////////////////////////////////
		// Value assignment: set and get.
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "addField", Ftype2<Id, string>::global(),
			"Create a dynamic field on the specified object" ),
		new SrcFinfo( "get", Ftype2< Id, string >::global(),
			"Getting a field value as a string: send out request:" ),
		new DestFinfo( "recvField",
					Ftype1< string >::global(),
					RFCAST( &PyMooseContext::recvField ),
					"Getting a field value as a string: Recv the value." ),
		new SrcFinfo( "set", // object, field, value 
				Ftype3< Id, string, string >::global(),
				"Setting a field value as a string: send out request:" ),
		new SrcFinfo( "file2tab", // object, filename, skiplines 
				Ftype3< Id, string, unsigned int >::global() ),

		///////////////////////////////////////////////////////////////
		// Clock control and scheduling
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "setClock", // clockNo, dt, stage
				Ftype3< int, double, int >::global(),
				"Setting values for a clock tick: setClock" ),
		new SrcFinfo( "useClock", // tickname, path, function
				Ftype3< string, string, string >::global(),
				"Assigning path and function to a clock tick: useClock" ),
		new SrcFinfo( "el", Ftype2< string, bool >::global(),
				"Getting a wildcard path of elements: send out request args are path, flag true for "
				"breadth-first list." ),
		new SrcFinfo( "resched", Ftype0::global(),
				"The return function for the wildcard past is the shared function recvElist"), // resched
		new SrcFinfo( "reinit", Ftype0::global() ), // reinit
		new SrcFinfo( "stop", Ftype0::global() ), // stop
		new SrcFinfo( "step", Ftype1< double >::global() ),
		new SrcFinfo( "requestClocks", 
					Ftype0::global(),
					"step, arg is time" ), //request clocks
		new DestFinfo( "recvClocks", 
					Ftype1< vector< double > >::global(), 
					RFCAST( &PyMooseContext::recvClocks ) ),
		new SrcFinfo( "requestCurrentTime", Ftype0::global() ),
		new SrcFinfo( "quit", Ftype0::global(), 
			"Returns time in the default return value." ),
		
		///////////////////////////////////////////////////////////////
		// Message functions
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "addMsg", 
					Ftype4< vector< Id >, string, vector< Id >, string >::global(),
					"Create a message. srcId, srcField, destId, destField" ),
		new SrcFinfo( "deleteMsg", Ftype2< Fid, int >::global(),
					"Delete a message based on number " ),
		new SrcFinfo( "deleteEdge", 
					Ftype4< Id, string, Id, string >::global(),
					"Delete a message based on src id.field and dest id.field "
					"This is how to specify an edge, so call it deleteEdge" ),
		new SrcFinfo( "listMessages", 
					Ftype3< Id, string, bool >::global(),
					"Request message list: id elm, string field, bool isIncoming" ),
		new DestFinfo( "recvMessageList",
					Ftype2< vector < Id >, string >::global(), 
					RFCAST( &PyMooseContext::recvMessageList ),
					"Receive message list and string with remote fields for msgs" ),

		///////////////////////////////////////////////////////////////
		// Object heirarchy manipulation functions.
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "copy", Ftype3< Id, Id, string >::global(),
					"This function is for copying an element tree, complete with messages, onto another." ),
		new SrcFinfo( "copyIntoArray", Ftype4< Id, Id, string, vector <double> >::global() ),
		new SrcFinfo( "move", Ftype3< Id, Id, string >::global(),
					"This function is for moving element trees." ),

		///////////////////////////////////////////////////////////////
		// Cell reader: filename cellpath
		///////////////////////////////////////////////////////////////
		
		new SrcFinfo( "readcell", 
			Ftype4< string, string, vector< double >, int >::global(),
			"filename, cellpath, parms, node" ),

		///////////////////////////////////////////////////////////////
		// Channel setup functions
		///////////////////////////////////////////////////////////////
		
		new SrcFinfo( "setupAlpha", 
					Ftype2< Id, vector< double > >::global(),
					"setupalpha" ),
		new SrcFinfo( "setupTau", 
					Ftype2< Id, vector< double > >::global(),
					"setuptau" ),
		new SrcFinfo( "tweakAlpha", Ftype1< Id >::global(),
					"tweakalpha" ),
		new SrcFinfo( "tweakTau", Ftype1< Id >::global(),
					"tweaktau" ),
		new SrcFinfo( "setupGate", 
					Ftype2< Id, vector< double > >::global() ),

		///////////////////////////////////////////////////////////////
		// SimDump facilities
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "readDumpFile", 
					Ftype1< string >::global(),
					"readDumpFile" ),
		new SrcFinfo( "writeDumpFile", 
					Ftype2< string, string >::global(),
					"writeDumpFile" ),
		new SrcFinfo( "simObjDump",
					Ftype1< string >::global(),
					"simObjDump" ),
		new SrcFinfo( "simUndump",
					Ftype1< string >::global(),
					"simundump" ),
		new SrcFinfo( "openfile", 
			Ftype2< string, string >::global() ),
		new SrcFinfo( "writefile", 
			Ftype2< string, string >::global() ),
		new SrcFinfo( "flushfile", 
			Ftype1< string >::global() ),
		new SrcFinfo( "listfiles", 
			Ftype0::global() ),
		new SrcFinfo( "closefile", 
			Ftype1< string >::global() ),
		new SrcFinfo( "readfile", 
			Ftype2< string, bool >::global() ),
		///////////////////////////////////////////////////////////////
		// Setting field values for a vector of objects
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "setVecField", // object, field, value 
			Ftype3< vector< Id >, string, string >::global(),
			"Setting a vec of field values as a string: send out request:" ),
		new SrcFinfo( "loadtab", 
			Ftype1< string >::global() ),
		new SrcFinfo( "tabop", 
			Ftype4< Id, char, double, double >::global() ),
		///////////////////////////////////////////////////////////////
		// SBML
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "readsbml", 
			Ftype3< string, string, int >::global() ),
		new SrcFinfo( "writesbml", 
			Ftype3< string, string, int >::global() ),
		///////////////////////////////////////////////////////////////
		// NeuroML
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "readneuroml", 
			Ftype3< string, string, int >::global() ),
		new SrcFinfo( "writeneuroml", 
			Ftype3< string, string, int >::global() ),
		///////////////////////////////////////////////////////////////
		// Misc
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "createGate", 
			Ftype2< Id, string >::global(),
			"Args: HHGate id, Interpol A id, Interpol B id. "
			"Request an HHGate explicitly to create Interpols, with the given "
			"ids. This is used when the gate is a global object, and so the "
			"interpols need to be globals too. Comes in use in TABCREATE in the "
			"parallel context." ),

	};
	
	static Finfo* pyMooseContextFinfos[] =
	{
		new SharedFinfo( "parser", parserShared,
				sizeof( parserShared ) / sizeof( Finfo* ),
				"This is a shared message to talk to the Shell." ),
		new DestFinfo( "readline",
			Ftype1< string >::global(),
			RFCAST( &dummyFunc ) ),
		new DestFinfo( "process",
			Ftype0::global(),
			RFCAST( &dummyFunc ) ), 
		new DestFinfo( "parse",
			Ftype1< string >::global(),
			RFCAST( &dummyFunc ) ), 
		new SrcFinfo( "echo", Ftype1< string>::global() ),

	};

	static Cinfo pyMooseContextCinfo(
		"PyMooseContext",
		"Subhasis Ray, NCBS",
                "Object to interact with MOOSE shell object",
		initNeutralCinfo(),
		pyMooseContextFinfos,
		sizeof(pyMooseContextFinfos) / sizeof( Finfo* ),
		ValueFtype1< PyMooseContext >::global()
	);

	return &pyMooseContextCinfo;
}

static const Cinfo* pyMooseContextCinfo = initPyMooseContextCinfo();
static const Slot setCweSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.cwe" );
static const Slot requestCweSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.trigCwe" );
static const Slot requestLeSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.trigLe" );
static const Slot pusheSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.pushe" );
static const Slot popeSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.pope" );
static const Slot createSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.create" );
static const Slot createArraySlot = 
	initPyMooseContextCinfo()->getSlot( "parser.createArray" );
static const Slot planarconnectSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.planarconnect" );
static const Slot planardelaySlot = 
	initPyMooseContextCinfo()->getSlot( "parser.planardelay" );
static const Slot planarweightSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.planarweight" );
static const Slot getSynCountSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.getSynCount" );
static const Slot deleteSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.delete" );
static const Slot addfieldSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.addField" );
static const Slot requestFieldSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.get" );
static const Slot setFieldSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.set" );
static const Slot file2tabSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.file2tab" );
static const Slot setClockSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.setClock" );
static const Slot useClockSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.useClock" );
static const Slot requestWildcardListSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.el" );
static const Slot reschedSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.resched" );
static const Slot reinitSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.reinit" );
static const Slot stopSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.stop" );
static const Slot stepSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.step" );
static const Slot requestClocksSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.requestClocks" );
static const Slot requestCurrentTimeSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.requestCurrentTime" );
static const Slot quitSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.quit" );

static const Slot addMessageSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.addMsg" );
static const Slot deleteMessageSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.deleteMsg" );
static const Slot deleteEdgeSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.deleteEdge" );
static const Slot listMessagesSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.listMessages" );

static const Slot copySlot = 
	initPyMooseContextCinfo()->getSlot( "parser.copy" );
static const Slot copyIntoArraySlot = 
	initPyMooseContextCinfo()->getSlot( "parser.copyIntoArray" );
static const Slot moveSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.move" );
static const Slot readCellSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.readcell" );

static const Slot setupAlphaSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.setupAlpha" );
static const Slot setupTauSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.setupTau" );
static const Slot tweakAlphaSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.tweakAlpha" );
static const Slot tweakTauSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.tweakTau" );
static const Slot setupGateSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.setupGate" );

static const Slot readDumpFileSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.readDumpFile" );
static const Slot writeDumpFileSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.writeDumpFile" );
static const Slot simObjDumpSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.simObjDump" );
static const Slot simUndumpSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.simUndump" );

static const Slot openFileSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.openfile" );
static const Slot writeFileSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.writefile" );
static const Slot flushFileSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.flushfile" );
static const Slot listFilesSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.listfiles" );
static const Slot closeFileSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.closefile" );
static const Slot readFileSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.readfile" );
static const Slot setVecFieldSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.setVecField" );
static const Slot loadtabSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.loadtab" );
static const Slot tabopSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.tabop" );
static const Slot readSbmlSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.readsbml" );
static const Slot writeSbmlSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.writesbml" );
static const Slot readNeuroMLSlot =
        initPyMooseContextCinfo()->getSlot( "parser.readneuroml" );
static const Slot writeNeuroMLSlot =
        initPyMooseContextCinfo()->getSlot( "parser.writeneuroml" );
static const Slot createGateSlot = 
	initPyMooseContextCinfo()->getSlot( "parser.createGate" );


//////////////////////////
// Static constants
//////////////////////////
const string pymoose::PyMooseContext::separator = "/";
// Unit tests
#ifdef DO_UNIT_TESTS
	extern void testBasecode();
	extern void testNeutral();
	extern void testSparseMatrix();
	extern void testShell();
	extern void testInterpol();
	extern void testTable();
	extern void testWildcard();
	extern void testSched();
	extern void testSchedProcess();
	extern void testBiophysics();
	extern void testHSolve();
	extern void testKinetics();
//	extern void testAverage();
#ifdef USE_MPI
	extern void testPostMaster();
#endif
#endif

// char* copyString( const string& s )
// {
//     char* ret = ( char* ) calloc ( s.length() + 1, sizeof( char ) );
//     strcpy( ret, s.c_str() );
//     return ret;
// }

//////////////////////////////////////////////////////////////////
// PyMooseContext Message recv functions
//////////////////////////////////////////////////////////////////

void pymoose::PyMooseContext::recvCwe( const Conn* c, Id cwe )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c->data() );
    gpw->cwe_ = cwe;
}

void PyMooseContext::recvElist( const Conn* c, vector< Id > elist )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c->data() );
    gpw->elist_ = elist;
}

void PyMooseContext::recvCreate( const Conn* c, Id e )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c->data() );
    gpw->createdElm_ = e;
}

void PyMooseContext::recvField( const Conn* c, string value )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c->data() );
    gpw->fieldValue_ = value;
}


void PyMooseContext::recvWildcardList(
    const Conn* c, vector< Id > value )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c->data() );
    gpw->elist_ = value;
}


void PyMooseContext::recvClocks( 
    const Conn* c, vector< double > dbls)
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c->data() );
    gpw->dbls_ = dbls;
}

void PyMooseContext::recvMessageList( 
    const Conn* c, vector< Id > elist, string s)
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c->data() );
    gpw->elist_ = elist;
    gpw->fieldValue_ = s;
}

//////////////////////////////////////////////////////////////////////
// Member functions
//////////////////////////////////////////////////////////////////////
/**
   The constructor is used by whatever wrapper connects this with the
   shell.  Cannot do much here. This constructor would have been
   private but for it is used by some other class (cinfo?) for
   creating the actual object instance (@see createPyMooseContext()
   method - we use send method for creating objects).
*/

PyMooseContext::PyMooseContext():parallel(false)
{
    genesisSli_ = NULL;
    genesisParseFinfo_ = NULL;    
}

PyMooseContext::~PyMooseContext()
{
    end();    
}

/**
   load a genesis script.
   script - filename/full path for loading the script file
*/
void PyMooseContext::loadG(string script)
{
    // Disconnect the PyMooseContext object temporarily and connect
    // GenesisParserWrapper to the shell
    bool ret;
    ret = this->shell_.eref().dropAll("parser"); // disconnect the context from shell
    assert(ret);
    ret = this->shell_.eref().add("parser", this->genesisSli_, "parser", ConnTainer::Default); // connect genesis parser
    assert(ret);
        
    string stmt = "include "+script;
    set<string>( this->genesisSli_, this->genesisParseFinfo_, stmt);
    set<string>( this->genesisSli_, this->genesisParseFinfo_, "\n");
    ret = this->shell_.eref().dropAll("parser"); // disconnect genesis parser
    assert(ret);
    ret = this->shell_.eref().add( "parser", this->myId_(), "parser", ConnTainer::Default); // reconnect context
    assert(ret);    
}
/**
   Run a genesis statement.
   stmt - statement to be executed.
*/
void PyMooseContext::runG(string stmt)
{
    bool ret;
    ret = this->shell_.eref().dropAll("parser"); // disconnect the context from shell
    assert(ret);
    ret = this->myId_.eref().dropAll("parser"); // disconnect the context from shell
    assert(ret);
    ret = this->shell_.eref().add("parser", this->genesisSli_, "parser", ConnTainer::Default); // connect genesis parser
    assert(ret);
    
    set<string>( this->genesisSli_, this->genesisParseFinfo_, stmt);
    set<string>( this->genesisSli_, this->genesisParseFinfo_, "\n");
    
    ret = this->shell_.eref().dropAll("parser"); // disconnect genesis parser
    assert(ret);
    ret = this->genesisSli_->id().eref().dropAll("parser");
    assert(ret);
    ret = this->shell_.eref().add( "parser", this->myId_(), "parser", ConnTainer::Default); // reconnect context
    assert(ret);    
}

/**
   @returns current working element - much like unix pwd
*/
Id PyMooseContext::getCwe()
{
    return cwe_;        
}
/**
   @param Id elementId: Id of the new working element.
   Sets the current working element - effectively unix cd
*/
// TODO: complete
void PyMooseContext::setCwe(Id elementId)
{
    if ((elementId.bad()) || (elementId() == NULL))
    {
        cerr << "ERROR:  PyMooseContext::setCwe(Id elementId) - Element with id " << elementId << " does not exist" << endl;
        return;        
    }
    
    send1< Id > (myId_(), setCweSlot, elementId);
    // Todo: if success
    cwe_ = elementId;    
}

void PyMooseContext::setCwe(string path)
{
    Id newElementId(path);
    if (!newElementId.bad() )
    {
        setCwe(newElementId);        
    }
    else 
    {
        cerr << "Error: Invalid path specified" << endl;
    }    
}

/**
   Get a field value.  We do not want to use this function at
   all. Python should be able to represent the fields directly in
   their own data type.

   We need a way to peep inside the objects witout going through the shell.
   We want shell only for creation and deletion.
   
   @param objectId - Id of the owner object
   @param fieldName - name of the field to be retrieved
   @returns the string representation of the field value
*/
const string& PyMooseContext::getField(Id objectId, string fieldName)
{
    send2<Id, string >(myId_(), requestFieldSlot, objectId, fieldName);
    return fieldValue_;
}

/**
   set a field value. Same comment here as for getField
*/
void PyMooseContext::setField(Id object, string fieldName, string fieldValue)
{
    send3< Id, string, string >(myId_(), setFieldSlot, object, fieldName, fieldValue);
}

   
/**
   Returns Id of the element containing this object
*/
Id PyMooseContext::id()
{
    return myId_;
}

/**
   @returns Element pointer for the Shell instance
*/
Id PyMooseContext::getShell()
{
    return shell_;
}

/**
 * 'breakMain()' is defined in main/main.cpp.
 * It is just a blank function, useful when debugging.
 * 
 * This is how to use it under PyMOOSE:
 *   1) Insert following lines at start of your python script:
 *          import os
 *          print os.getpid()  # This will print PID to stdout.
 *          raw_input()        # This will cause process to wait for keyboard I/P.
 *      
 *      If debugging some MOOSE initialization code (like Cinfo init.), then
 *      place these lines before 'import moose'. Otherwise easier to place them
 *      afterwards.
 *   2) Run python script. It will print PID and wait for keyboard input. Let it
 *      wait.
 *   3) Attach GDB to the process using 'gdb --pid <pid>'
 *   4) Set breakpoint at 'breakMain': 'b breakMain'. If the moose module has
 *      not yet been imported in the python process, answer 'y' to the question
 *      regarding setting the breakpoint in a future library load.
 *   5) Continue process under GDB.
 *   6) Return to python, let program begin by hitting 'Enter'.
 *   7) Once the breakpoint hits, return to GDB to set any further breakpoints.
 *   8) Continue process under GDB. Now you can proceed with debugging in the
 *      usual way.
 * 
 * Note: Using 'breakMain' is not needed if one needs to set breakpoints at
 * places that will not get hit in startup code: Unit tests, for example. Also,
 * one can avoid the trouble of attaching gdb by simply loading python under
 * gdb directly ('gdb python'), and then loading the python script using one of
 * the available ways.
 */
extern void breakMain();

/**
   This method should be used instead of a constructor. The constructor is kept public
   for only the moose-core's use.
   @param string shellName - name of the shell to be embedded in this context
   @param string contextElement - name of this context
   @returns PyMooseContext* pointer to the newly created PyMooseContext object.
*/
PyMooseContext* PyMooseContext::createPyMooseContext(string contextName, string shellName)
{
    static PyMooseContext* context = 0;
    if (context)
    {
        return context;
    }
    
    Element* shell;
    bool ret;
    int argc = 0;
    char** argv = NULL;
    init(argc, argv); // No clue why this strange signature with int&
                      // and char**&
#ifdef DO_UNIT_TESTS
	// if ( mynode == 0 )
	if ( 1 )
	{
		testBasecode();
		testNeutral();
		testSparseMatrix();
		testShell();
		testWildcard();
		testInterpol();
		testTable();
		testSched();
		testSchedProcess();
		testBiophysics();
		testHSolve();
		testKinetics();
//		testAverage();
	}
#endif

    // From maindir/main.cpp: parser requires to be created before the clock job
    Element * genesisSli = makeGenesisParser();
    
    // Call to an empty function, useful for setting breakpoints.
    breakMain();

    Id shellId = Id::shellId();
    
    ret = genesisSli->id().eref().dropAll("parser");
    assert(ret);
    ret = shellId.eref().dropAll("parser"); // disconnect genesis parser
    assert(ret);

    Element* contextElement = Neutral::create( "PyMooseContext",contextName, shellId, Id::scratchId());
    context = static_cast<PyMooseContext*> (contextElement->data(0) );
    context->shell_ = shellId;    
    context->genesisSli_ = genesisSli;
    context->myId_ = contextElement->id();
    context->setCwe(Element::root()->id() ); // set initial element = root
    Eref ref = shellId.eref();
    ret = ref.add( "parser", contextElement, "parser", ConnTainer::Default);// reconnect context
    assert(ret);
    
    Id cj("/sched/cj");
    Id t0("/sched/cj/t0");
    Id t1("/sched/cj/t1");
    FuncVec::sortFuncVec();           
    setupDefaultSchedule(t0(), t1(), cj());
    context->genesisParseFinfo_ = context->genesisSli_->findFinfo("parse");
    assert(context->genesisParseFinfo_!=0);
    
    
    return context;        
}


void PyMooseContext::destroyPyMooseContext(PyMooseContext* context)
{
    if (context == NULL)
    {
        cerr << "ERROR: destroyPyMooseContext(PyMooseContext* context) - NULL pointer passed to method." << endl;
        return;
    }
    
    context->end();
    
    Element* ctxt = context->myId_();
    
    if ( ctxt == NULL)
    {
        cerr << "ERROR: destroyPyMooseContext(PyMooseContext* context) - Element with id " << context->myId_ << " does not exist" << endl;
        return;
    }
    
    
    Id shellId = context->shell_;
    if (shellId.bad() )
    {
        cerr << "ERROR:  PyMooseContext::destroyPyMooseContext(PyMooseContext* context) - Shell id turns out to be invalid." << endl;
        
    }    
    set(  ctxt, "destroy" );
    Element* shell = shellId();
    if (shell == NULL)
    {
        cerr <<  "ERROR:  PyMooseContext::destroyPyMooseContext(PyMooseContext* context) - Shell id turns out to be invalid." << endl;
        return;
    }
    else 
    {
        set ( shell, "destroy" );   
    }
}

/**
   @param className : Class name of the MOOSE object to be generated.
   @param name : Name of the instance to be generated.
   @param parent - Id of the element under which this should be created
   @returns id of the newly generated object.
*/

Id PyMooseContext::create(string className, string name, Id parent)
{
    if ( parent == Id::badId() )
    {
        parent = cwe_;
    }
    
    if ( name.length() < 1 ) {
        cerr << "Error: invalid object name : " << name << endl;
        return Id();
    }
    if ( !Cinfo::find( className ) )
    {
        cerr << "Error: could not find any class of name " << className << endl;
        return Id();
    }    
    int childNode = 0; // TODO: PyMOOSE does not yet have the parallel computing facilities.
    send4 < string, string, int, Id > (
        myId_(), createSlot, className, name, childNode, parent );
//    cerr << "PyMooseContext::create - Created Id = " << createdElm_ << " " << className << "::" << name << endl;
    
    return createdElm_;
}

bool PyMooseContext::destroy( Id victim)
{
    if ( victim != Id() )
    {
        send1< Id >(myId_(), deleteSlot, victim );
        return true;
    }
    else 
    {
        return false;
    }    
}
/**
   This method cleans up all resources allocated for a simulation.
*/
void PyMooseContext::end()
{
    // TODO: this has become obsolete - update with the new implementation of Id system
//     while (createdElm_ > myId_)
//     {
//         if (Element::element(createdElm_) != NULL)
//             destroy(createdElm_);
//         cerr << "Destroyed " << createdElm_ << endl;
        
//         --createdElm_;        
//     }    
}

const Id& PyMooseContext::getParent( Id e ) const
{
    static Id ret = Id::badId();
    Element* elm = e();
    if (elm == NULL)
    {
        cerr << "ERROR: PyMooseContext::getParent( Id e ) - Element with id " << e << " does not exist." << endl;        
    }    
    else if (e != Element::root()->id() )
    {
        get< Id >( elm, "parent", ret );
    }
    else 
    {
        cerr << "WARNING: PyMooseContext::getParent( Id e ) - 'root' object does not have any parent" << endl;
        ret = e;        
    }
    
    return ret;
}

const string& PyMooseContext::getPath(Id id) const
{
	fieldValue_ = Shell::eid2path( id );
	return fieldValue_;
}

Id PyMooseContext::pathToId(string path, bool echo)
{
    Id returnValue = Shell::path2eid(path, separator, parallel);
    
    if (( returnValue.bad() ) && echo)
    {
        cerr << "ERROR: PyMooseContext::pathToId(string path) - Could not find the object '" << path << "'"<< endl;
    }

    return returnValue;    
}

bool PyMooseContext::exists(const Id& id)
{
    Element* e = id();
    return e != NULL;    
}

bool PyMooseContext::exists(string path)
{
    Id id(path);    
    return !id.bad();    
}

void PyMooseContext::addField(string objectPath, string fieldname)
{
    send2<Id, string>(myId_(), addfieldSlot, Id(objectPath), fieldname);
}

void PyMooseContext::addField(Id objectId, string fieldname)
{
    send2<Id, string>(myId_(), addfieldSlot, objectId, fieldname);
}

const vector < Id >& PyMooseContext::getChildren(Id id)
{
    elist_.clear();
    if ( id() != NULL )
    {
        send1< Id >( myId_(), requestLeSlot, id );
    }
    else 
    {
        cerr << "ERROR: PyMooseContext::getChildren(Id id) - Element with Id " << id << " does not exist" << endl;
    }
    
    return elist_;
}

const vector < Id >& PyMooseContext::getChildren(string path)
{
    elist_.clear();    
    Id id(path);        
    if ( id.bad() )
    {
        send2< string, bool >( myId_(), requestWildcardListSlot, path, true);
        return elist_;
    }
    send1< Id >( myId_(), requestLeSlot, id );
    return elist_;
}

const vector<Id> & PyMooseContext::getWildcardList(string path, bool ordered=true)
{
    elist_.clear();
    send2<string, bool>(myId_(), requestWildcardListSlot, path, ordered);
    return elist_;
}
void PyMooseContext::srandom(long seed)
{
    mtseed(seed);
}


/*
  A set of overloaded functions to step the clocks.
  The three versions are required to account for
  genesis parsers multiple decisions depending on argument list.
*/
/**
   The most basic versions: step by the amount specified as rutime.
   corresponds to :
   step 0.005 -t
   in genesis parser
*/
void PyMooseContext::step(double runtime )
{
    Element* e = myId_();

    if ( runtime < 0 ) {
        cout << "Error: " << runtime << " : negative time is illegal\n";
        return;
    }
    send1< double >( e, stepSlot, runtime );
}
/**
   step by mult multiple of the smallest clock duration.
   corresponds to:
   step 5
   in genesis parser
*/
void PyMooseContext::step(long mult)
{
    double runtime;
    
    send0( myId_(), requestClocksSlot ); 
    assert( dbls_.size() > 0 );
    // This fills up a vector of doubles with the clock duration.
    // Find the shortest dt.
    double min = 1.0e10;
    vector< double >::iterator i;
    for ( i = dbls_.begin(); i != dbls_.end(); i++ )
        if ( min > *i )
            min = *i ;
    runtime = min * mult;
    step(runtime);     
}
/**
   step by the smallest clock duration only
   corresponds to genesis parser command:
   step 
*/
void PyMooseContext::step(void)
{
    step((long)1);    
}

void PyMooseContext::setClock(int clockNo, double dt, int stage)
{
    send3< int, double, int >(myId_(), setClockSlot, clockNo, dt, stage);
}


vector <double> & PyMooseContext::getClocks()
{
    send0( myId_(), requestClocksSlot );
    return dbls_;        
}

void PyMooseContext::useClock(const string& tickName, const string& path, const string& func)
{
    send3< string, string, string>(myId_(), useClockSlot, tickName, path, func);
}

void PyMooseContext::useClock(int tickNo, const string& path, const string& func)
{
    static const int tickNameLen = 63;
    char tickName[tickNameLen+1];
    snprintf(tickName, (size_t)(tickNameLen), "t%d",tickNo);
    this->useClock( tickName, path, func);
}

void PyMooseContext::reset()
{
    send0(myId_(), reschedSlot);
    send0(myId_(), reinitSlot);    
}

void PyMooseContext::stop()
{
    send0( myId_(), stopSlot );
}

void PyMooseContext::addTask(string arg)
{
    //Do nothing - but give a message to inform that
    cerr << "void PyMooseContext::addTask(string arg) - empty function.\n";
}

void PyMooseContext::copy(const Id& src, const Id& dest_parent, std::string new_name)
{
    send3< Id, Id, string >( myId_(), copySlot, src, dest_parent, new_name);
}


/**
   This just does the copying without returning anything.
   Corresponds to the procedural technique used in Genesis shell
*/
void PyMooseContext::do_deep_copy( const Id& object, const Id& dest, string new_name)
{
    send3< Id, Id, string >  ( myId_(), copySlot, object, dest, new_name);
}
/**
   This is the object oriented version. It returns Id of new copy.
*/
Id PyMooseContext::deepCopy( const Id& object, const Id& dest, string new_name)
{
    if ( new_name.find(PyMooseContext::separator) != string::npos )
    {
        cerr << "Error: object name may not contain "<< PyMooseContext::separator << endl;
        Id id;
        return id;    
    }
    
    do_deep_copy( object, dest,  new_name);
    
    string path = getPath(dest);
    size_t len = path.length();
    
    if ( len > 0 ){
        if ( path.substr(len-1) != PyMooseContext::separator )
        {
            path += PyMooseContext::separator;            
        }
    }
    else
    {
        path += PyMooseContext::separator;
    }
    
    path += new_name;
    Id id(path);
    return id;
}

/**
   move object into the element specified by dest and rename the object to new_name
*/
void PyMooseContext::move( const Id& object, const Id& dest, string new_name)
{
    send3< Id, Id, string >(
        myId_(), moveSlot, object, dest, new_name );
}
void PyMooseContext::move( string object, string dest, string new_name)
{
    Id src(object), dst(dest);
    send3< Id, Id, string >(
        myId_(), moveSlot, src, dst, new_name );
}

bool PyMooseContext::connect(const Id& src, string srcField, const Id& dest, string destField)
{
    vector< Id > srcList( 1, src );
    vector< Id > destList( 1, dest );
    if ( !src.bad() && !dest.bad() )
    {
        send4< vector<Id>, string, vector <Id>, string >( myId_(), addMessageSlot,
			srcList, srcField, destList, destField );
        return 1;
    }
    return 0;
}
/**
   createmap
*/
void PyMooseContext::createMap(Id src, Id parent, string name, vector<double> params)
{
    send4< Id, Id, string, vector <double> >( myId_(), copyIntoArraySlot, src, parent, name, params );
}
void PyMooseContext::createMap(Id src, Id parent, string name, unsigned int nx, unsigned int ny, double dx, double dy, double xo, double yo)
{
    vector < double > params;
    params.push_back(nx);
    params.push_back(ny);
    params.push_back(dx);
    params.push_back(dy);
    params.push_back(xo);
    params.push_back(yo);    
    send4< Id, Id, string, vector <double> >( myId_(), copyIntoArraySlot, src, parent, name, params );
}

    
void PyMooseContext::createMap(string src, string dest, unsigned int nx, unsigned int ny, double dx, double dy, double xo, double yo, bool copy)
{
    
        
    vector <double> params;
    params.push_back(nx);
    params.push_back(ny);
    params.push_back(dx);
    params.push_back(dy);
    params.push_back(xo);
    params.push_back(yo);    
    Id parent(dest);
    if (parent.bad())
    {
        string headpath = Shell::head(dest, "/");
        Id head(headpath);
        if (head.bad())
        {
            cout << "Error: '" << headpath << "'" << " is not defined for destination path " << dest << "." << endl;
            return;            
        }
        send3< string, string, Id >( myId_(), createSlot, "Neutral", Shell::tail(dest, "/"), head);
    }
    if (!copy)
    {
        string className = src;
        if ( !Cinfo::find( className ))
        {
            cout << "Error: Unknown class: " << className << endl;
            return;
        }
        string name = ( className == "Neutral")? "proto": src;
        if (name.length() < 1)
        {
            cout << "Error: Invalid object name: " << name << endl;
            return;
        }
        send4 < string, string, Id, vector <double> >( myId_(), createArraySlot, className, name, parent, params );
    }
    else
    {
        Id e;
        Id pa;
        string name;        
	if ( parseCopyMove( src, dest, myId_, e, parent, name ) ) {
            send4< Id, Id, string, vector <double> >( myId_(), copyIntoArraySlot, e, pa, name, params );
        }
    }
    
}
/**
 * This function figures out destination and name for moves
 * and copies. The rules are:
 * 1. Target can be existing element, in which case name is retained.
 *  In this case the childname is set to the empty string "".
 * 2. Target might not exist. In this case the parent of the target
 * 	must exist, and the object e is going to be renamed. This new
 * 	name is returned in the childname.
 * Modified from the function with same name in GenesisParserWrapper.cpp
 */
bool PyMooseContext::parseCopyMove( string src, string dest, Id s,
		Id& e,  Id& pa, string& childname )
{
    e = Id( src );
    if ( !e.zero() && !e.bad() ) {
        childname = "";
        pa = Id( dest );
        if ( pa.bad() ) { // Possibly we are renaming it too.
            string pastr = dest;
            if ( pastr.find( "/" ) == string::npos ) {
                pastr = ".";
            } else {
                pastr = Shell::head( dest, "/" );
            }
            if ( pastr == "" )
                pastr = ".";
            // pa = GenesisParserWrapper::path2eid( pastr, s );
            pa = Id( pastr );
            if ( pa.bad() ) { // Nope, even that doesn't work.
                cout << "Error: Parent element " << dest << 
                    " not found\n";
                return 0;
            }
            childname = Shell::tail( dest, "/" );
        }
        return 1;
    } else {
        cout << "Error: source element " <<
            src << " not found\n";
    }

    return 0;
}
/**
   Get list of messages for object "obj" from/to field "field".
*/
vector <string> PyMooseContext::getMessageList(Id obj, string field, bool incoming)
{
    elist_.clear();
    // The return message puts the elements in elist_ and the 
    // target field names in fieldValue_
    send3 < Id, string, bool > (myId_(), listMessagesSlot, obj, field, incoming);
    vector <string> fieldlist;
    vector <string> list;
    
    separateString( fieldValue_, fieldlist, ", ");
    
    if (elist_.size() > 0)
    {
        assert(elist_.size() == fieldlist.size());
        for ( unsigned int i = 0; i < elist_.size(); ++i )
        {
            list.push_back("["+elist_[i].path()+"]."+fieldlist[i]);
        }              
    }
    return list;    
}
/**
   Get list of messages from/to object "obj" for all fields.
*/
vector <string> PyMooseContext::getMessageList(Id obj, bool incoming)
{
    vector <string> msgList;
    if ( obj.bad() ) {
        cout << "Error: PyMooseContext::getMessageList" << ": unknown element." << endl;
        return msgList;
    }
    string direction = incoming? " <- ":" -> ";
    vector <string> fieldList;    
    send2<Id, string>(myId_(), requestFieldSlot, obj, "fieldList" );
    separateString(fieldValue_, fieldList, ", ");
    for ( unsigned int i = 0; i < fieldList.size(); i++)
    {
        if (fieldList[i] == "fieldList")
        {
            continue;
        }
        vector <string> tmpList = getMessageList(obj, fieldList[i], incoming);
        for ( unsigned int j = 0; j < tmpList.size(); ++j)
            {
                string msgInfo = "["+obj.path()+"]."+ fieldList[i] + direction + tmpList[j];
                msgList.push_back(msgInfo);
            }
    }
    return msgList;    
}

void PyMooseContext::planarConnect(string src, string dst, double probability)
{
    send3<string, string, double> (myId_(), planarconnectSlot, src, dst, probability);    
}

void PyMooseContext::plannarDelay(string src, double delay)
{
    send2< string, double > ( myId_(), planardelaySlot, src, delay);    
}

void PyMooseContext::planarWeight(string src, double weight)
{
    send2< string, double >( myId_(), planarweightSlot, src, weight);
}

/*
  The following functions are for manipulating HHChannels.  They
  should ideally be part of HHChannel/HHGate class only.  But I had to
  put them here as there is no better way to access the slots of the
  shell.

  So HHChannel / HHGate should have functions that wrap these fnctions.

  The functions are findChanGateId
  
*/

Id PyMooseContext::findChanGateId( string channel, string gate)
{
    string path = "";
    if (( gate.at(0) == 'X' )||( gate.at(0) == 'x' ) )
        path = channel + "/xGate";
    else if (( gate.at(0) == 'Y' ) || ( gate.at(0) == 'y' ) )
        path = channel + "/yGate";
    else if (( gate.at(0)   == 'Z' )||( gate.at(0)   == 'z' ) )
        path = channel + "/zGate";
    Id gateId(path);
    if ( gateId.bad() ) // Don't give up, it might be a tabgate
        gateId = Id( channel);
    if ( gateId.bad() ) { // Now give up
        cout << "Error: findChanGateId: unable to find channel/gate '" << channel << "/" << gate << endl;        
    }
    return gateId;
}

/**
   this is a common interface for setting up channel function - used
   by setupAlpha, setupTau.
   
   parameter channel is the path to the channel containing the gate.
   The actual gate used is xGate or yGate or xGate when the gate
   parameter starts with 'x', 'y' or 'z' respectively.
*/
void PyMooseContext::setupChanFunc(string channel, string gate, vector <double>& parms, const Slot& slot)
{    
    if (parms.size() < 10 ) {
        cerr << "Error: PyMooseContext::setupChanFunc() -  We need a vector for these items: AA AB AC AD AF BA BB BC BD BF size min max (length should be at least 10)" << endl;
        return;
    }

    Id gateId = findChanGateId(channel, gate );
    if (gateId.bad())
        return;

    
    double size = 3000.0;
    double min = -0.100;
    double max = 0.050;
    if (parms.size() < 11)
    {
        parms.push_back(size);
    }
    if (parms.size() < 12 )
    {
        parms.push_back(min);
    }
    if ( parms.size() < 13 )
    {
        parms.push_back(max);        
    }    
    
    send2< Id, vector< double > >( myId_(), slot, gateId, parms );
}

/**
   this is a common interface for setting up channel function - used
   by setupAlpha, setupTau.
   
   parameter channel is the path to the channel containing the gate.
   The actual gate used is xGate or yGate or xGate when the gate
   parameter starts with 'x', 'y' or 'z' respectively.
*/
void PyMooseContext::setupChanFunc(string channel, string gate, double AA, double AB, double AC, double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max, const Slot& slot)
{
    Id gateId = findChanGateId(channel, gate );
    if (gateId.bad() )
        return;
    vector<double> params;
    params.push_back(AA);
    params.push_back(AB);
    params.push_back(AC);
    params.push_back(AD);
    params.push_back(AF);
    params.push_back(BA);
    params.push_back(BB);
    params.push_back(BC);
    params.push_back(BD);
    params.push_back(BF);
    params.push_back(size);
    params.push_back(min);
    params.push_back(max);
    send2< Id, vector< double > >( myId_(), slot, gateId, params );
}

void PyMooseContext::setupAlpha( string channel, string gate, vector <double> parms ) 
{
    setupChanFunc( channel, gate, parms, setupAlphaSlot );
}

void PyMooseContext::setupAlpha(string channel, string gate, double AA, double AB, double AC, double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max)
{
    setupChanFunc(channel, gate, AA, AB, AC, AD, AF, BA, BB, BC, BD, BF, size, min, max, setupAlphaSlot);    
}

void PyMooseContext::setupTau( string channel, string gate, vector <double> parms ) 
{
    setupChanFunc( channel, gate, parms, setupTauSlot );
}
void PyMooseContext::setupTau(string channel, string gate, double AA, double AB, double AC, double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max)
{
    setupChanFunc(channel, gate, AA, AB, AC, AD, AF, BA, BB, BC, BD, BF, size, min, max, setupTauSlot);    
}
void PyMooseContext::tweakChanFunc( string  channel, string gate, const Slot& slot )
{
    Id gateId = findChanGateId( channel, gate );
    if ( gateId.bad() )
        return;
    send1< Id >( myId_(), slot, gateId );
}

void PyMooseContext::tweakAlpha( string channel, string gate ) 
{
    tweakChanFunc( channel, gate, tweakAlphaSlot );
}

void PyMooseContext::tweakTau( string channel, string gate)
{
    tweakChanFunc( channel, gate, tweakTauSlot );
}

//=================================================================
// These are a set of overloaded versions of the above
// - where we take advantage of the fact that python users
// have more control over the moose objects ;-) and they can
// do better programming than the genesis parser allows
//=================================================================

void PyMooseContext::setupChanFunc(const Id& gateId, vector <double> parms, const Slot& slot)
{
    
    if (parms.size() < 10 ) {
        cerr << "Error: PyMooseContext::setupChanFunc() -  We need a vector for these items: AA AB AC AD AF BA BB BC BD BF size min max (length should be at least 10)" << endl;
        return;
    }
    if (gateId.bad() )
        return;
    
    double size = 3000.0;
    double min = -0.100;
    double max = 0.050;
    if (parms.size() < 11)
    {
        parms.push_back(size);
    }
    if (parms.size() < 12 )
    {
        parms.push_back(min);
    }
    if ( parms.size() < 13 )
    {
        parms.push_back(max);        
    }    
    
    send2< Id, vector< double > >( myId_(), slot, gateId, parms );
}

void PyMooseContext::setupAlpha(const Id& gateId, vector <double> parms )
{
    setupChanFunc( gateId, parms, setupAlphaSlot );
}

void PyMooseContext::setupTau(const Id& gateId, vector <double> parms )
{
    setupChanFunc( gateId, parms, setupTauSlot );
}

void PyMooseContext::tweakChanFunc(const Id& gateId, const Slot& slot )
{
    if ( gateId.bad() )
        return;
    send1< Id >( myId_(), slot, gateId );
}

void PyMooseContext::tweakAlpha( const Id& gateId )
{
    tweakChanFunc( gateId, tweakAlphaSlot );
}
void PyMooseContext::tweakTau( const Id& gateId)
{
    tweakChanFunc( gateId, tweakTauSlot );
}

/**
   was marked deprecated in MOOSE, yet kept
*/
void PyMooseContext::tabFill(const Id& table, int xdivs, int mode)
{
    char argstr[32];
    snprintf(argstr, 32, "%d,%d", xdivs, mode);
    
    send3< Id, string, string >( myId_(),
                                 setFieldSlot, table, "tabFill", argstr );
}

const vector <double>& PyMooseContext::getTableVector(const Id& table)
{
    get< vector < double > > (table(), "tableVector", dbls_);
    return dbls_;
}
void PyMooseContext::readCell(string filename, string cellpath)
{
    std::string command = "readcell " + filename + " " + cellpath;
#ifndef NDEBUG
    cout << "PyMooseContext::readCell -- running GENESIS command: " << command << endl;
#endif
    this->runG(command);
}

/**
   params should have these entries:
   CM, RM, RA, EREST_ACT, ELEAK


   -- these functions have been deprecated because of the bug with
   ReadCell::read() - which requires the GenesisParserWrapper to be
   connected to Shell. But PyMooseContext has to be connected to
   Shell in order for the readCell message to reach it.
*/
void PyMooseContext::readCell(string filename, string cellpath, vector <double> params)
{
    cout << "void PyMooseContext::readCell(string filename, string cellpath, vector <double> params) -  deprecated." << endl;
    return;
    static const int node = 0;
    send4< string, string, vector < double >, int >( 
        myId_(), 
        readCellSlot, filename, cellpath , params, node);
}

void PyMooseContext::readCell(string filename, string cellpath, double cm, double rm, double ra, double erestAct, double eleak)
{
    cout << "void PyMooseContext::readCell(string filename, string cellpath, double cm, double rm, double ra, double erestAct, double eleak) - deprecated." << endl;
    return;
    
    static const int node = 0; // this parameter was added in the shell
                        // function to handle distributed network. do
                        // not know what to do with this for PyMOOSE
    
    vector <double> params;
    params.push_back(cm);
    params.push_back(rm);
    params.push_back(ra);
    params.push_back(erestAct);
    params.push_back(eleak);
    send4< string, string, vector < double >, int >( 
        myId_(), 
        readCellSlot, filename, cellpath , params, node);
}

void PyMooseContext::readSBML(string fileName, string modelPath)
{
    int node = parseNodeNum(modelPath);
    send3< string, string, int >(myId_(), readSbmlSlot, fileName, modelPath, node );
}

void PyMooseContext::readNeuroML(string fileName, string modelPath)
{
    int node = parseNodeNum(modelPath);
#ifdef USE_NEUROML    
    send3< string, string, int >(myId_(), readNeuroMLSlot, fileName, modelPath, node);
#else
    cout << "void PyMooseContext::readNeuroML(string fileName, string modelPath) -- this version of MOOSE was built without neuroML support." << endl;
#endif
}

const string& PyMooseContext::className(const Id& objId) const
{
    return  objId()->className();
}

const string& PyMooseContext::getName(const Id objId) const
{
    return objId()->name();
}

const string& PyMooseContext::description(const string className) const
{
    const Cinfo* cinfo = Cinfo::find(className);
    if (cinfo){
        fieldValue_ = cinfo->description();
    }
    else fieldValue_ = className + ": No such MOOSE class exists.";
    return fieldValue_;
}
const string& PyMooseContext::author(const string className) const
{
    const Cinfo* cinfo = Cinfo::find(className);
    if (cinfo){
        fieldValue_ = cinfo->author();
    }
    else fieldValue_ = className + ": No such MOOSE class exists.";
    return fieldValue_;
}/**
   returns help on a specied class or a specific field of a class.
*/
const string& PyMooseContext::doc(const string& target) const
{
    string field = "";
    string className(target);
    string::size_type field_start = target.find_first_of(".");
    if ( field_start != string::npos) {
        // May we need to go recursively?
        // Assume for the time being that only one level of field
        // documentation is displayed. No help for channel.xGate.A
        // kind of stuff.
        field = target.substr(field_start+1); 
        className = target.substr(0, field_start);
    }
        
    fieldValue_ = getClassDoc(className, field);
    if (!fieldValue_.empty())
    {
        return fieldValue_;
    }
    // check if it is old-style class name
    // fallback to looking for a file with same name in documentation directory
    fieldValue_ = getCommandDoc(target);
    return fieldValue_;
}
/**
   Returns the list of Ids of elements connected as destination of
   message from this element.
*/
// think if this is sane - returning a local vector - swig somehow
// takes care of it, but shouldn't we be passing a vector reference as
// argument which will be filled in?
const vector<Id>& PyMooseContext::getNeighbours(Id src, string finfoName, int direction)
{
    vector<string> fieldList;
    Element* element = src();
    this->elist_.clear();
    if (finfoName.length() == 0){
        return this->elist_;
    } else if (finfoName.length() == 1 && finfoName[0] == '*'){
        const Cinfo* cinfo = element->cinfo();
        vector<const Finfo*> finfoList;
        cinfo->listFinfos(finfoList);
        for ( int i = 0; i < finfoList.size(); ++i ){
            fieldList.push_back(finfoList[i]->name());
        }
    } else {
        fieldList.push_back(finfoName);
    }
    for ( int i = 0; i < fieldList.size(); ++i){
        Conn* conn = element->targets(fieldList[i], src.index());
        while (conn->good()){
            if (conn->isDest() == direction || direction == INOUT){
                Eref target = conn->target();
                this->elist_.push_back(target.id());
            }
            conn->increment();
        }      
    }
    return this->elist_;
}
/*

map<Id, string> PyMooseContext::neighbourFields(Id src, const string& field)
{
    map<Id, string> neighbourFieldMap;
    Element* element = src();
    if(field.length() == 0){
        Conn* conn = element->targets(field, src.index());
        while (conn->good());// TODO finish later
    }
    
}
*/

double PyMooseContext::getCurrentTime()
{
    send0( myId_(), requestCurrentTimeSlot );
    double ret = atof(fieldValue_.c_str());	
    return static_cast< float >( ret );    
}


/**
   This function just picks up the ValueFinfo fields. As opposed to
   getFieldList with FieldType = FTYPE_VALUE, this version gets all value
   fields including the user-added ones (by addField call).
*/
const vector <string>& PyMooseContext::getValueFieldList(Id id)
{
    
    send2 <Id, string>(myId_(), requestFieldSlot, id, "fieldList");
    vector< string >::iterator i;
    strings_.clear();
    separateString( fieldValue_, strings_, ", " );
    return strings_;
}

/**
   This gets the field names in a vector according to field type. This
   uses the statically initialized field names, so in case of FTYPE_VALUE
   fields, it does not retrieve the fields added later via addField
   call.
 */
const vector<string>& PyMooseContext::getFieldList(Id id, FieldType ftype)
{
    const Cinfo* cinfo = id()->cinfo();
    vector<const Finfo*> finfoList;
    cinfo->listFinfos(finfoList);
    strings_.clear();
    switch (ftype){
        case FTYPE_VALUE:
            for (int i = 0; i < finfoList.size(); ++i)
            {
                const ValueFinfo* finfo = dynamic_cast<const ValueFinfo*>(finfoList[i]);
                if (finfo){
                    strings_.push_back(finfo->name());
                }
            }
            break;
        case FTYPE_LOOKUP:
            for (int i = 0; i < finfoList.size(); ++i)
            {
                const LookupFinfo* finfo = dynamic_cast<const LookupFinfo*>(finfoList[i]);
                if (finfo){
                    strings_.push_back(finfo->name());
                }
            }
            break;
        case FTYPE_SOURCE:
            for (int i = 0; i < finfoList.size(); ++i)
            {
                const SrcFinfo* finfo = dynamic_cast<const SrcFinfo*>(finfoList[i]);
                if (finfo){
                    strings_.push_back(finfo->name());
                }
            }
            break;
        case FTYPE_DEST:
            for (int i = 0; i < finfoList.size(); ++i)
            {
                const DestFinfo* finfo = dynamic_cast<const DestFinfo*>(finfoList[i]);
                if (finfo){
                    strings_.push_back(finfo->name());
                }
            }
            break;
        case FTYPE_SHARED:
            for (int i = 0; i < finfoList.size(); ++i)
            {
                const SharedFinfo* finfo = dynamic_cast<const SharedFinfo*>(finfoList[i]);
                if (finfo){
                    strings_.push_back(finfo->name());
                }
            }
            break;
            /* The following are special finfos - some with incomplete
             information, e.g., apparently SolveFinfo does not
             implement name().
            */
        // 
        // case FTYPE_SOLVE:
        //     for (int i = 0; i < finfoList.size(); ++i)
        //     {
        //         const SolveFinfo* finfo = dynamic_cast<const SolveFinfo*>(finfoList[i]);
        //         if (finfo){
        //             fieldList.push_back(finfo->name());
        //         }
        //     }
        //     break;
        // case FTYPE_THIS:
        //     for (int i = 0; i < finfoList.size(); ++i)
        //     {
        //         const ThisFinfo* finfo = dynamic_cast<const ThisFinfo*>(finfoList[i]);
        //         if (finfo){
        //             fieldList.push_back(finfo->name());
        //         }
        //     }
        //     break;
        // case FTYPE_GLOBAL:
        //     for (int i = 0; i < finfoList.size(); ++i)
        //     {
        //         const GlobalMarkerFinfo* finfo = dynamic_cast<const GlobalMarkerFinfo*>(finfoList[i]);
        //         if (finfo){
        //             fieldList.push_back(finfo->name());
        //         }
        //     }
        //     break;
        // case FTYPE_DEL:
        //     for (int i = 0; i < finfoList.size(); ++i)
        //     {
        //         const DeletionFinfo* finfo = dynamic_cast<const DeletionMarkerFinfo*>(finfoList[i]);
        //         if (finfo){
        //             fieldList.push_back(finfo->name());
        //         }
        //     }
        //     break;
            
        case FTYPE_ALL:
            for (int i = 0; i < finfoList.size(); ++i)
            {
                strings_.push_back(finfoList[i]->name());
            }
            break;
        default:
            cout << "ERROR: Unknown Finfo type." << endl;
    }
    return strings_;
}

#ifdef DO_UNIT_TESTS
/**
   These are the unit tests
   Each method takes a count and a print argument.
   It runs the tests count times and if print is true, prints a message.
*/

bool PyMooseContext::testSetGetField(int count, bool doPrint)
{
    bool testResult = true;
    
    int i = 0;
    string setValue;    
    string getValue;
    double eps = 1e-5; // usual error (for string conversion) on PC is 1e-6
    
    Id obj = create( "Compartment", "TestSetGetCompartment", Element::root()->id() );
    
    for ( i = 0; i < count; ++i )
    {
        double d = (i+1)/3.0e-6;
        double val;
        
        setValue = toString < double > (d);
        
        setField(obj, "Rm", setValue);
        getValue = getField(obj, "Rm" );

        
        val = atof(getValue.c_str() );        
        testResult = testResult && ((d > val)? (1-val/d) < eps : (1 - d/val) < eps);
        if (testResult == false)
        {
            cerr << "TEST:: PyMooseContext::testSetGetField((int count, bool doPrint) - set value " << setValue << ", retrieved value " << getValue << ". Actual set value: "<< d << ", retrieved value: "<< val << ", error: " << 1-val/d << ", allowed error: " << eps << " - FAILED" << endl ;            
        }
    }
    destroy(obj);    
    return testResult;
}

bool PyMooseContext::testSetGetField(string className, string fieldName, string fieldValue, int count, bool doPrint)
{

    string retrievedVal;
    bool testResult = true;
    Id obj = create(className,"TestSetGetField", getCwe() );    
    for ( int i = 0; i < count; ++i)
    {
        setField(obj, fieldName, fieldValue);
        retrievedVal = getField(obj, fieldName);
        
        if (doPrint)
        {
            cerr << "TEST::  PyMooseContext::testSetGetField() - value retrieved " << retrievedVal << endl;
        }
        testResult = testResult && ( fieldValue == retrievedVal);

    }
    destroy(obj);
    
    return testResult;    
}

bool PyMooseContext::testCreateDestroy(string className, int count, bool doPrint)
{
    vector < Id > idList;
    int i;
    Id instanceId;
    
    for (i = 0; i < count; ++i)
    {
        instanceId = create(className, "test"+toString < int > (i), Element::root()->id() );
        idList.push_back(instanceId);
    }
       
    while (idList.size() > 0)
    {
        instanceId = (Id)(idList.back() );
        idList.pop_back();
        destroy(instanceId);
        --i;        
    }
    if (doPrint)
    {
        cerr << "TEST::  PyMooseContext::testCreateDestroy(string className, int count, bool doPrint) - create and destroy " << count << " " << className << " instances ... " << ((i == 0)? "SUCCESS":"FAILED" ) << endl;
    }
    
    return ( i == 0 );     
}

bool PyMooseContext::testPyMooseContext(int testCount, bool doPrint)
{
    bool overallResult = true;    
    bool testResult;
    PyMooseContext *context = createPyMooseContext( "TestContext", "TestShell" );
    
    testResult = context->testCreateDestroy( "Compartment", testCount, doPrint);
    overallResult = overallResult && testResult;
    
    if(doPrint){
        cerr << "TEST::  PyMooseContext::testPyMooseContext(int testCount, bool doPrint) - testing create and destroy ... " << (testResult?"SUCCESS":"FAILED" ) << endl;
    }
    testResult = context->testSetGetField(testCount, doPrint);
    overallResult = overallResult && testResult;
    
    if(doPrint){
        cerr << "TEST:: PyMooseContext::testPyMooseContext(int testCount, bool doPrint) - testing set and get ... " << (testResult?"SUCCESS":"FAILED" ) << endl;
    }
    
    delete context;
    
    return overallResult;
}


#endif // DO_UNIT_TESTS

#endif // _PYMOOSE_CONTEXT_CPP
