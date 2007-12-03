/*******************************************************************
 * File:            PyMooseContext.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-03-12 03:412
 ********************************************************************/
#ifndef _PYMOOSE_CONTEXT_CPP
#define _PYMOOSE_CONTEXT_CPP

#include "PyMooseContext.h"
#include "../basecode/Id.h"
#include "../element/Neutral.h"
#include "../scheduling/Tick.h"
#include "../scheduling/ClockJob.h"
#include "../builtins/Interpol.h"
#include "../builtins/Table.h"
#include "../maindir/init.h"
using namespace std;

extern int mooseInit(string configFile);

extern const Cinfo* initShellCinfo();
extern const Cinfo* initTickCinfo();
extern const Cinfo* initClockJobCinfo();
extern const Cinfo* initTableCinfo();
extern const Cinfo* initSchedulerCinfo();

const Cinfo* initPyMooseContextCinfo()
{
    static Finfo* contextShared[] =
	{
            // Current working element
            
            // 1. Setting cwe
            new SrcFinfo( "cwe", Ftype1< Id >::global() ),
            // 2. Getting cwe back: First trigger a request
            new SrcFinfo( "trigCwe", Ftype0::global() ),
            // Send out cwe info
//            new DestFinfo(
            // 3. Then receive the cwe info
            new DestFinfo( "recvCwe",
                           Ftype1< Id >::global(),
                           RFCAST( &PyMooseContext::recvCwe ) ),

            // Getting a list of child ids:

            // 4. first send a request with the requested parent elm
            // id.
            new SrcFinfo( "trigLe", Ftype1< Id >::global() ),
            // 5. Then receive the vector of child ids. This function
            // is shared by several other messages as all it does is
            // dump the elist into a temporary local buffer.
            new DestFinfo( "recvElist", 
                           Ftype1< vector< Id > >::global(), 
                           RFCAST( &PyMooseContext::recvElist ) ),

            ///////////////////////////////////////////////////////////////
            // Object hierarchy manipulation functions
            ///////////////////////////////////////////////////////////////
            // Creating an object
            // 6. send out the request.
            new SrcFinfo( "create",
                          Ftype3< string, string, Id >::global() ),
            new SrcFinfo( "createArray",
                          Ftype4< string, string, Id, vector <double> >::global() ),
            new SrcFinfo( "planarconnect", Ftype3< string, string, double >::global() ),
            new SrcFinfo( "planardelay", Ftype2< string, double >::global() ),
            new SrcFinfo( "planarweight", Ftype2< string, double >::global() ),
	
            // 7. receive the returned object id.
            new DestFinfo( "recvCreate",
                           Ftype1< Id >::global(),
                           RFCAST( &PyMooseContext::recvCreate ) ),
            // Deleting an object
            // 8. send out the request.
            new SrcFinfo( "delete", Ftype1< Id >::global() ),

            ///////////////////////////////////////////////////////////////
            // Value assignment: set and get.
            ///////////////////////////////////////////////////////////////
            // Getting a field value as a std::string
            // 9. send out request:
            new SrcFinfo( "get", Ftype2< Id, string >::global() ),
            // 10.receive the value.
            new DestFinfo( "recvField",
                           Ftype1< string >::global(),
                           RFCAST( &PyMooseContext::recvField ) ),
            // Setting a field value as a std::string
            // 11. send out request:
            new SrcFinfo( "set", // object, field, value 
                          Ftype3< Id, string, string >::global() ),


            ///////////////////////////////////////////////////////////////
            // Clock control and scheduling
            ///////////////////////////////////////////////////////////////
            // 12. Setting values for a clock tick: setClock
            new SrcFinfo( "setClock", // clockNo, dt, stage
                          Ftype3< int, double, int >::global() ),
            // 13. Assigning path and function to a clock tick: useClock
            new SrcFinfo( "useClock", // tick id, path, function
                          Ftype3< Id, vector< Id >, string >::global() ),

            // 14. Getting a wildcard path of elements: send out request
            // args are path, flag true for breadth-first list.
            new SrcFinfo( "el", Ftype2< string, bool >::global() ),
            // The return function for the wildcard past is the shared
            // function recvElist

            new SrcFinfo( "resched", Ftype0::global() ), // 15.resched
            new SrcFinfo( "reinit", Ftype0::global() ), // 16. reinit
            new SrcFinfo( "stop", Ftype0::global() ), // 17. stop
            new SrcFinfo( "step", Ftype1< double >::global() ),
            // 18. step, arg is time
            new SrcFinfo( "requestClocks", 
                          Ftype0::global() ), // 19. request clocks
            new DestFinfo( "recvClocks", 
                           Ftype1< vector< double > >::global(), 
                           RFCAST( &PyMooseContext::recvClocks ) ), // 20. receive the response
		
            ///////////////////////////////////////////////////////////////
            // Message info functions
            ///////////////////////////////////////////////////////////////
            // 21. Request message list: id elm, string field, bool isIncoming
            new SrcFinfo( "listMessages", 
                          Ftype3< Id, string, bool >::global() ),
            // 22. Receive message list and string with remote fields for msgs
            new DestFinfo( "recvMessageList",
                           Ftype2< vector < Id >, string >::global(), 
                           RFCAST( &PyMooseContext::recvMessageList ) ),

            ///////////////////////////////////////////////////////////////
            // Object heirarchy manipulation functions.
            ///////////////////////////////////////////////////////////////
            // 23. This function is for copying an element tree, complete with
            // messages, onto another.
            new SrcFinfo( "copy", Ftype3< Id, Id, string >::global() ),
            new SrcFinfo( "copyIntoArray", Ftype4< Id, Id, string, vector <double> >::global() ),
	
            // 24. This function is for moving element trees.
            new SrcFinfo( "move", Ftype3< Id, Id, string >::global() ),

            ///////////////////////////////////////////////////////////////
            // 25. Cell reader: filename cellpath
            ///////////////////////////////////////////////////////////////
            new SrcFinfo( "readcell", Ftype2< string, string >::global() ),

            ///////////////////////////////////////////////////////////////
            // Channel setup functions
            ///////////////////////////////////////////////////////////////
            // 26. setupalpha
            new SrcFinfo( "setupAlpha", 
                          Ftype2< Id, vector< double > >::global() ),
            // 27. setuptau
            new SrcFinfo( "setupTau", 
                          Ftype2< Id, vector< double > >::global() ),
            // 28. tweakalpha
            new SrcFinfo( "tweakAlpha", Ftype1< Id >::global() ),
            // 29. tweaktau
            new SrcFinfo( "tweakTau", Ftype1< Id >::global() ),

            ///////////////////////////////////////////////////////////////
            // SimDump facilities
            ///////////////////////////////////////////////////////////////
            // 30. readDumpFile
            new SrcFinfo( "readDumpFile", 
                          Ftype1< string >::global() ),
            // 31. writeDumpFile
            new SrcFinfo( "writeDumpFile", 
                          Ftype2< string, string >::global() ),
            // 32. simObjDump
            new SrcFinfo( "simObjDump",
                          Ftype1< string >::global() ),
            // 33. simundump
            new SrcFinfo( "simUndump",
                          Ftype1< string >::global() ),
            new SrcFinfo( "openfile", 
                          Ftype2< string, string >::global() ),
            new SrcFinfo( "writefile", 
                          Ftype2< string, string >::global() ),
            new SrcFinfo( "listfiles", 
                          Ftype0::global() ),
            new SrcFinfo( "closefile", 
                          Ftype1< string >::global() ),
            new SrcFinfo( "readfile", 
                          Ftype2< string, bool >::global() ),
	
            ///////////////////////////////////////////////////////////////
            // Setting field values for a vector of objects
            ///////////////////////////////////////////////////////////////
            // 34. Setting a vec of field values as a string: send out request:
            new SrcFinfo( "setVecField", // object, field, value 
                          Ftype3< vector< Id >, string, string >::global() ),
            new SrcFinfo( "loadtab", 
                          Ftype1< string >::global() ),

	};
	
    static Finfo* pyMooseContextFinfos[] =
	{
            new SharedFinfo( "parser", contextShared,
                             sizeof( contextShared ) / sizeof( Finfo* ) ),
	};

    static Cinfo pyMooseContextCinfo(
        "PyMooseContext",
        "Upinder S. Bhalla, NCBS, 2004-2007",
        "Object to handle the old Genesis parser",
        initNeutralCinfo(),
        pyMooseContextFinfos,
        sizeof(pyMooseContextFinfos) / sizeof( Finfo* ),
        ValueFtype1< PyMooseContext >::global()
	);
    return &pyMooseContextCinfo;
}

// These static initializations are required to ensure proper sequence
// of static object creation
static const Cinfo* shellCinfo = initShellCinfo();
static const Cinfo* tickCinfo = initTickCinfo();
static const Cinfo* clockJobCinfo = initClockJobCinfo();
static const Cinfo* tableCinfo = initTableCinfo();

static const Cinfo* pyMooseContextCinfo = initPyMooseContextCinfo();

static const unsigned int setCweSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.cwe" );

static const unsigned int requestCweSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.trigCwe" );

static const unsigned int requestLeSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.trigLe" );

static const unsigned int createSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.create" );
static const unsigned int createArraySlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.createArray" );
static const unsigned int planarconnectSlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.planarconnect" );
static const unsigned int planardelaySlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.planardelay" );
static const unsigned int planarweightSlot = 
    initPyMooseContextCinfo()->getSlotIndex( "parser.planarweight" );

static const unsigned int deleteSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.delete" );
static const unsigned int requestFieldSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.get" );
static const unsigned int setFieldSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.set" );

static const unsigned int setClockSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.setClock" );
static const unsigned int useClockSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.useClock" );
static const unsigned int requestWildcardListSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.el" );
static const unsigned int reschedSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.resched" );
static const unsigned int reinitSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.reinit" );
static const unsigned int stopSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.stop" );
static const unsigned int stepSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.step" );
static const unsigned int requestClocksSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.requestClocks" );
static const unsigned int listMessagesSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.listMessages" );
static const unsigned int copySlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.copy" );
static const unsigned int copyIntoArraySlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.copyIntoArray" );
static const unsigned int moveSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.move" );
static const unsigned int readCellSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.readCell" );

static const unsigned int setupAlphaSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.setupAlpha" );
static const unsigned int setupTauSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.setuptau" );
static const unsigned int tweakAlphaSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.tweakAlpha" );
static const unsigned int tweakTauSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.tweakTau" );
static const unsigned int readDumpFileSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.readDumpFile" );
static const unsigned int writeDumpFileSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.writeDumpFile" );
static const unsigned int simObjDumpSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.simObjDump" );
static const unsigned int simUndumpSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser.simUndump" );


static const unsigned int openFileSlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.openfile" );
static const unsigned int writeFileSlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.writefile" );
static const unsigned int listFilesSlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.listfiles" );
static const unsigned int closeFileSlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.closefile" );
static const unsigned int readFileSlot = 
	initPyMooseContextCinfo()->getSlotIndex( "parser.readfile" );
static const unsigned int setVecFieldSlot = 
    initPyMooseContextCinfo()->getSlotIndex( "parser.setVecField" );


static const unsigned int loadtabSlot = 
    initPyMooseContextCinfo()->getSlotIndex( "parser.loadtab" );
//////////////////////////
// Static constants
//////////////////////////
const string PyMooseContext::separator = "/";


// void PyMooseContext::processFunc( const Conn& c )
// {
//     PyMooseContext* data =
// 	static_cast< PyMooseContext* >( c.targetElement()->data() );

//     data->Process();
// }

char* copyString( const string& s )
{
    char* ret = ( char* ) calloc ( s.length() + 1, sizeof( char ) );
    strcpy( ret, s.c_str() );
    return ret;
}
void setupDefaultSchedule( 
	Element* t0, Element* t1, 
	Element* t2, Element* t3, 
	Element* t4, Element* t5,
	Element* cj)
{
	set< double >( t0, "dt", 1e-5 );
	set< double >( t1, "dt", 1e-5 );
	set< int >( t1, "stage", 1 );
	set< double >( t2, "dt", 5e-3 );
	set< double >( t3, "dt", 5e-3 );
	set< int >( t3, "stage", 1 );
	set< double >( t4, "dt", 5e-3 );
	set< double >( t5, "dt", 1.0 );
	set( cj, "resched" );
	set( cj, "reinit" );
}

//////////////////////////////////////////////////////////////////
// PyMooseContext Message recv functions
//////////////////////////////////////////////////////////////////

void PyMooseContext::recvCwe( const Conn& c, Id cwe )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c.targetElement()->data() );
    gpw->cwe_ = cwe;
}

void PyMooseContext::recvElist( const Conn& c, vector< Id > elist )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c.targetElement()->data() );
    gpw->elist_ = elist;
}

void PyMooseContext::recvCreate( const Conn& c, Id e )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c.targetElement()->data() );
    gpw->createdElm_ = e;
}

void PyMooseContext::recvField( const Conn& c, string value )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c.targetElement()->data() );
    gpw->fieldValue_ = value;
}


void PyMooseContext::recvWildcardList(
    const Conn& c, vector< Id > value )
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c.targetElement()->data() );
    gpw->elist_ = value;
}


void PyMooseContext::recvClocks( 
    const Conn& c, vector< double > dbls)
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c.targetElement()->data() );
    gpw->dbls_ = dbls;
}

void PyMooseContext::recvMessageList( 
    const Conn& c, vector< Id > elist, string s)
{
    PyMooseContext* gpw = static_cast< PyMooseContext* >
        ( c.targetElement()->data() );
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

PyMooseContext::PyMooseContext()
{
}

PyMooseContext::~PyMooseContext()
{
    end();    
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
string PyMooseContext::getField(Id objectId, string fieldName)
{
    send2<Id, string >(myId_(), requestFieldSlot, objectId, fieldName);
    
    if ( fieldValue_.length() == 0 ) // Nothing came back
        return 0;
    return copyString( fieldValue_.c_str() );
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
   This method should be used instead of a constructor. The constructor is kept public
   for only the moose-core's use.
   @param string shellName - name of the shell to be embedded in this context
   @param string contextElement - name of this context
   @returns PyMooseContext* pointer to the newly created PyMooseContext object.
*/
PyMooseContext* PyMooseContext::createPyMooseContext(string contextName, string shellName)
{
    static const Cinfo* shellCinfo = initShellCinfo();
    static const Cinfo* tickCinfo = initTickCinfo();
    static const Cinfo* clockJobCinfo = initClockJobCinfo();
    static const Cinfo* tableCinfo = initTableCinfo();
    static const Cinfo* pyMooseContextCinfo = initPyMooseContextCinfo();

    
    Element* shell;
    bool ret;
    // Call the global initialization function
    mooseInit("config.xml");
    cout << "Trying to find shell with name " << shellName << endl;
    Id shellId(shellName);
    
    //lookupGet<Id, string > (Element::root(), "lookupChild", shellId, shellName );
    
    if (shellId.bad() )
    {
        cerr << "Warning: shell does not exist, trying to create a new one!" << endl;
        const Cinfo* c = Cinfo::find("Shell");
        assert(c!=0);
        const Finfo* childSrc = Element::root()->findFinfo("childSrc" );
        assert( childSrc != 0 );
        shell = c->create( Id(1), shellName);
        assert(shell != 0 );
        ret = childSrc->add( Element::root(), shell, 
		shell->findFinfo( "child" ) );
        assert(ret);
        shellId = shell->id();
    
    }
    else
    {
        shell = shellId();
    }
    
    
    Element* contextElement = Neutral::create( "PyMooseContext",contextName, shell);
    
    const Finfo* shellFinfo, *contextFinfo;
    shellFinfo = shell->findFinfo( "parser" );
    assert(shellFinfo!=NULL);
    
    contextFinfo = contextElement->findFinfo( "parser" );
    assert(contextFinfo!=NULL);
    ret = shellFinfo->add( shell, contextElement, contextFinfo);

    assert(ret);
    
    PyMooseContext* context = static_cast<PyMooseContext*> (contextElement->data() );
    context->shell_ = shellId;    
    context->myId_ = contextElement->id();
    context->setCwe(Element::root()->id() ); // set initial element = root
//    context->scheduler_ = Neutral::create( "Neutral", "sched", Element::root() )->id();    
//    context->clockJob_ = Neutral::create( "ClockJob", "cj", context->scheduler_() )->id();
    lookupGet<Id, string > (Element::root(), "lookupChild", context->scheduler_, "sched" );
    if ( context->scheduler_.bad() )
    {
        cerr << "Scheduler not found" << endl;
    }
    
    lookupGet<Id, string > (Element::root(), "lookupChild", context->clockJob_, "cj" );
    Element* cj =  context->clockJob_();
//     Element* t0 = Neutral::create( "Tick", "t0", cj );
//     Element* t1 = Neutral::create( "Tick", "t1", cj );

    
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
        
    send3 < string, string, Id > (
        myId_(), createSlot, className, name, parent );
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

const Id PyMooseContext::getParent( Id e ) const
{
    Id ret = Id::badId();
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

const string PyMooseContext::getPath(Id id) const
{
    return id.path();
}

Id PyMooseContext::pathToId(string path, bool echo)
{
    Id returnValue(path);
    
    if (( returnValue.bad() ) && echo)
    {
        cerr << "ERROR: PyMooseContext::pathToId(string path) - Could not find the object '" << path << "'"<< endl;
    }

    return returnValue;    
}

bool PyMooseContext::exists(Id id)
{
    Element* e = id();
    return e != NULL;    
}

bool PyMooseContext::exists(string path)
{
    Id id(path);    
    return !id.bad();    
}

vector < Id >& PyMooseContext::getChildren(Id id)
{
    elist_.resize(0);
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

vector < Id >& PyMooseContext::getChildren(string path)
{
    elist_.resize(0);    
    Id id(path);
    
    /// \todo: Use better test for a bad path than this.
    if ( id.bad() )
    {
        cerr << "ERROR:  PyMooseContext::getChildren(string path) - This path seems to be invalid" << endl;
        return elist_;
    }
    send1< Id >( myId_(), requestLeSlot, id );
    return elist_;
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

void PyMooseContext::setClock(int clockNo, double dt, int stage = 0)
{
    send3< int, double, int >(myId_(), setClockSlot, clockNo, dt, stage);
}


vector <double> & PyMooseContext::getClocks()
{
    send0( myId_(), requestClocksSlot );
    return dbls_;        
}

void PyMooseContext::useClock(Id tickId, string path, string func)
{
    Element * e = myId_();
    send2< string, bool >( e, requestWildcardListSlot, path, 0 );
    send3< Id, vector< Id >, string >(
        myId_(),
        useClockSlot, 
        tickId, elist_,  func );
}

void PyMooseContext::useClock(int tickNo, std::string path, std::string func)
{
    char tickName[40];
    snprintf(tickName, (size_t)(39), "/sched/cj/t%d",tickNo);
    Id tickId(tickName);
    if (tickId.bad())
    {
        cerr << "useClock: Invalid clock number " << tickNo << endl;
        return;        
    }
    this->useClock( tickId, path, func);    
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
    //Do nothing
}
/**
   This just does the copying without returning anything.
   Corresponds to the procedural technique used in Genesis shell
*/
void PyMooseContext::do_deep_copy( Id object, string new_name, Id dest)
{
    send3< Id, Id, string >  ( myId_(), copySlot, object, dest, new_name);
}
/**
   This is the object oriented version. It returns Id of new copy.
*/
Id PyMooseContext::deepCopy( Id object, string new_name, Id dest)
{
    do_deep_copy( object,  new_name, dest);
    
    string path = getPath(dest) + PyMooseContext::separator+new_name;
    Id id(path);
    return id;
}

/**
   move object into the element specified by dest and rename the object to new_name
*/
void PyMooseContext::move( Id object, string new_name, Id dest)
{
    send3< Id, Id, string >(
        myId_(), moveSlot, object, dest, new_name );
}

bool PyMooseContext::connect(Id src, string srcField, Id dest, string destField)
{
    if ( !src.bad() && !dest.bad() ) {
        Element* se = src( );
        Element* de = dest( );
        const Finfo* sf = se->findFinfo( srcField );
        if ( !sf ) return false;
        const Finfo* df = de->findFinfo( destField );
        if ( !df ) return false;
        return (bool)(se->findFinfo( srcField )->add( se, de, de->findFinfo( destField ) ));
    }
    return false;    
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
void PyMooseContext::setupChanFunc(string channel, string gate, vector <double>& parms, const unsigned int& slot)
{    
    if (parms.size() < 10 ) {
        cerr << "Error: PyMooseContext::setupChanFunc() -  We need a vector for these items: AA AB AC AD AF BA BB BC BD BF size min max (length should be at least 10)" << endl;
        return;
    }

    Id gateId = findChanGateId(channel, gate );
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

/**
   this is a common interface for setting up channel function - used
   by setupAlpha, setupTau.
   
   parameter channel is the path to the channel containing the gate.
   The actual gate used is xGate or yGate or xGate when the gate
   parameter starts with 'x', 'y' or 'z' respectively.
*/
void PyMooseContext::setupChanFunc(string channel, string gate, double AA, double AB, double AC, double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max, const unsigned int& slot)
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
void PyMooseContext::tweakChanFunc( string  channel, string gate, unsigned int slot )
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

void PyMooseContext::setupChanFunc(Id gateId, vector <double> parms, unsigned int slot)
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

void PyMooseContext::setupAlpha( Id gateId, vector <double> parms )
{
    setupChanFunc( gateId, parms, setupAlphaSlot );
}

void PyMooseContext::setupTau( Id gateId, vector <double> parms )
{
    setupChanFunc( gateId, parms, setupTauSlot );
}

void PyMooseContext::tweakChanFunc( Id gateId, unsigned int slot )
{
    if ( gateId.bad() )
        return;
    send1< Id >( myId_(), slot, gateId );
}

void PyMooseContext::tweakAlpha( Id gateId )
{
    tweakChanFunc( gateId, tweakAlphaSlot );
}
void PyMooseContext::tweakTau( Id gateId)
{
    tweakChanFunc( gateId, tweakTauSlot );
}
//========================

void PyMooseContext::tabFill(Id table, int xdivs, int mode)
{
    string argstr = xdivs + "," + mode;
    send3< Id, string, string >( myId_(), setFieldSlot, table, "tabFill", argstr );
}

void PyMooseContext::readCell( string filename, string cellpath )
{
    cerr << "PyMooseContext::readCell( " << filename << ", " << cellpath << ")" << endl;    
    send2< string, string >( shell_(), 
                             readCellSlot, filename, cellpath );
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
