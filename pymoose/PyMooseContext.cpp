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
#include "../element/Neutral.h"
#include "../scheduling/Tick.h"
#include "../scheduling/ClockJob.h"
#include "../builtins/Interpol.h"
#include "../builtins/Table.h"

using namespace std;

extern const Cinfo* initShellCinfo();
extern const Cinfo* initTickCinfo();
extern const Cinfo* initClockJobCinfo();
extern const Cinfo* initTableCinfo();

extern const unsigned int BAD_ID;

//extern vector <string> separateString(string s, vector <string> v, string separator);

const Cinfo* initPyMooseContextCinfo()
{
    /**
     * This is a shared message to talk to the Shell.
     */
    static TypeFuncPair contextTypes[] =
	{
            // Setting cwe
            TypeFuncPair( Ftype1< unsigned int >::global(), 0 ),
            // Getting cwe back: First trigger a request
            TypeFuncPair( Ftype0::global(), 0 ),
            // Then receive the cwe info
            TypeFuncPair( Ftype1< unsigned int >::global(),
                          RFCAST( &PyMooseContext::recvCwe ) ),

            // Getting a list of child ids: First send a request with
            // the requested parent elm id.
            TypeFuncPair( Ftype1< unsigned int >::global(), 0 ),
            // Then recv the vector of child ids.
            TypeFuncPair( Ftype1< vector< unsigned int > >::global(), 
                          RFCAST( &PyMooseContext::recvElist ) ),
            ///////////////////////////////////////////////////////////////
            // Object heirarchy manipulation functions.
            ///////////////////////////////////////////////////////////////		
            // Creating an object: Send out the request.
            TypeFuncPair( 
                Ftype3< string, string, unsigned int >::global(), 0 ),
            // Creating an object: Recv the returned object id.
            TypeFuncPair( Ftype1< unsigned int >::global(),
                          RFCAST( &PyMooseContext::recvCreate ) ),
            // Deleting an object: Send out the request.
            TypeFuncPair( Ftype1< unsigned int >::global(), 0 ),
            ///////////////////////////////////////////////////////////////
            // Value assignment: set and get.
            ///////////////////////////////////////////////////////////////
	
            // Getting a field value as a string: send out request:
            TypeFuncPair( 
                Ftype2< unsigned int, string >::global(), 0 ),
            // Getting a field value as a string: Recv the value.
            TypeFuncPair( Ftype1< string >::global(),
                          RFCAST( &PyMooseContext::recvField ) ),
            // Setting a field value as a string: send out request:
            TypeFuncPair( // object, field, value 
                Ftype3< unsigned int, string, string >::global(), 0 ),
            
            ///////////////////////////////////////////////////////////////
            // Clock control and scheduling
            ///////////////////////////////////////////////////////////////
	
            // Setting values for a clock tick: setClock
            TypeFuncPair( // clockNo, dt, stage
                Ftype3< int, double, int >::global(), 0 ),
            // Assigning path and function to a clock tick: useClock
            TypeFuncPair( // tick id, path, function
                Ftype3< unsigned int, vector< unsigned int >, string >::global(), 0 ),

            // Getting a wildcard path of elements: send out request
            // args are path, flag true for breadth-first list.
            TypeFuncPair( Ftype2< string, bool >::global(), 0 ),
            // The return function for the wildcard past is the shared
            // function recvElist
   
            TypeFuncPair( Ftype0::global(), 0), // resched
            TypeFuncPair( Ftype0::global(), 0), // reinit
            TypeFuncPair( Ftype0::global(), 0), // stop
            TypeFuncPair( Ftype1<double>::global(), 0),
            // step, arg is time
            TypeFuncPair( Ftype0::global(), 0),
            TypeFuncPair( Ftype1< vector < double > >::global(), RFCAST(&PyMooseContext::recvClocks )),
            ///////////////////////////////////////////////////////////////
            // Message info functions
            ///////////////////////////////////////////////////////////////
            // Request message list: id elm, string field, bool isIncoming
            TypeFuncPair( Ftype3< Id, string, bool >::global(), 0 ),
            // Receive message list and string with remote fields for msgs
            TypeFuncPair( Ftype2< vector < unsigned int >, string >::global(), 
                          RFCAST( &PyMooseContext::recvMessageList ) ),
            ///////////////////////////////////////////////////////////////
            // Object heirarchy manipulation functions.
            ///////////////////////////////////////////////////////////////
            // This function is for copying an element tree, complete with
            // messages, onto another.
            TypeFuncPair( Ftype3< unsigned int, unsigned int, string >::global(),  0 ),
            // This function is for moving element trees.
            TypeFuncPair( Ftype3< unsigned int, unsigned int, string >::global(),  0 ),
            
            ///////////////////////////////////////////////////////////////
            // Cell reader: filename cellpath
            ///////////////////////////////////////////////////////////////
            TypeFuncPair( Ftype2< string, string >::global(),  0 ),
            ///////////////////////////////////////////////////////////////
            // Channel setup functions
            ///////////////////////////////////////////////////////////////
            // setupalpha
            TypeFuncPair( Ftype2< unsigned int, vector< double > >::global(),  0 ),
            // setuptau
            TypeFuncPair( Ftype2< unsigned int, vector< double > >::global(),  0 ),
            // tweakalpha
            TypeFuncPair( Ftype1< unsigned int >::global(),  0 ),
            // tweaktau
            TypeFuncPair( Ftype1< unsigned int >::global(),  0 ),
            ///////////////////////////////////////////////////////////////
            // SimDump facilities - MAY NOT BE REQUIRED SO MUCH FOR PYMOOSE
            ///////////////////////////////////////////////////////////////
            // readDumpFile
            TypeFuncPair( Ftype1< string >::global(),  0 ),
            // writeDumpFile
            TypeFuncPair( Ftype2< string, string >::global(),  0 ),
            // simObjDump
            TypeFuncPair( Ftype1< string >::global(),  0 ),
            // simundump
            TypeFuncPair( Ftype1< string >::global(),  0 ),

            ///////////////////////////////////////////////////////////////
            // Setting field values for a vector of objects
            ///////////////////////////////////////////////////////////////
            // Setting a field value as a string: send out request:
            TypeFuncPair( // object, field, value 
                Ftype3< vector< unsigned int >, string, string >::global(), 0 ),

	};

    static Finfo* pyMooseContextFinfos[] =
	{
            new SharedFinfo( "parser", contextTypes,
                             sizeof( contextTypes ) / sizeof( TypeFuncPair ) ),
            //          new DestFinfo( "process",
//                            Ftype0::global(),
//                            RFCAST( &PyMooseContext::processFunc ) ), 
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

static const Cinfo* pyMooseContextCinfo = initPyMooseContextCinfo();
static const Cinfo* shellCinfo = initShellCinfo();
static const Cinfo* tickCinfo = initTickCinfo();
static const Cinfo* clockJobCinfo = initClockJobCinfo();
static const Cinfo* tableCinfo = initTableCinfo();

static const unsigned int setCweSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 0;
static const unsigned int requestCweSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 1;
static const unsigned int requestLeSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 2;
static const unsigned int createSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 3;
static const unsigned int deleteSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 4;
static const unsigned int requestFieldSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 5;
static const unsigned int setFieldSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 6;

static const unsigned int setClockSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 7;
static const unsigned int useClockSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 8;
static const unsigned int requestWildcardListSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 9;
static const unsigned int reschedSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 10;
static const unsigned int reinitSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 11;
static const unsigned int stopSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 12;
static const unsigned int stepSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 13;
static const unsigned int requestClocksSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 14;
static const unsigned int listMessagesSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 15;
static const unsigned int copySlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 16;
static const unsigned int moveSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 17;
static const unsigned int readCellSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 18;

static const unsigned int setupAlphaSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 19;
static const unsigned int setupTauSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 20;
static const unsigned int tweakAlphaSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 21;
static const unsigned int tweakTauSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 22;
static const unsigned int readDumpFileSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 23;
static const unsigned int writeDumpFileSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 24;
static const unsigned int simObjDumpSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 25;
static const unsigned int simUndumpSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 26;

static const unsigned int setVecFieldSlot = 
initPyMooseContextCinfo()->getSlotIndex( "parser" ) + 27;


//////////////////////////
// Static constants
//////////////////////////
const std::string PyMooseContext::separator = "/";
//const Id PyMooseContext::BAD_ID = BAD_ID;
const Id PyMooseContext::BAD_ID = ~0;


// void PyMooseContext::processFunc( const Conn& c )
// {
// 	PyMooseContext* data =
// 	static_cast< PyMooseContext* >( c.targetElement()->data() );

// 	data->Process();
// }

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
    const Conn& c, vector< unsigned int > value )
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
    if ( Element::element(elementId) == NULL)
    {
        cerr << "ERROR:  PyMooseContext::setCwe(Id elementId) - Element with id " << elementId << " does not exist" << endl;
        return;        
    }
    
    send1< Id > (Element::element(myId_), setCweSlot, elementId);
    // Todo: if success
    cwe_ = elementId;    
}

void PyMooseContext::setCwe(std::string path)
{
    Id newElementId = pathToId(path);
    if (newElementId != BAD_ID)
    {
        cwe_ = newElementId;
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
   @returns the std::string representation of the field value
*/
std::string PyMooseContext::getField(Id objectId, std::string fieldName)
{
    send2<Id, std::string >(Element::element(myId_), requestFieldSlot, objectId, fieldName);
    return fieldValue_;
}

/**
   set a field value. Same comment here as for getField
*/
void PyMooseContext::setField(Id object, std::string fieldName, std::string fieldValue)
{
    send3< Id, std::string, std::string >(Element::element(myId_), setFieldSlot, object, fieldName, fieldValue);
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
PyMooseContext* PyMooseContext::createPyMooseContext(std::string shellName, std::string contextName)
{    
//    set< string, string >( Element::root(), "create", "Shell", shellName);    
//    Element * shellElement = Element::lastElement();
    Element* shellElement = Neutral::create( "Shell", "shell", Element::root() );

//     unsigned int shellId;
//     cerr << "lookupGet returned: " << lookupGet< unsigned int, string >( Element::root(), "lookupChild", shellId, "shell" ) << endl;
//     assert( shellId != BAD_ID );
//     Element* shellElement = Element::element( shellId );

    set< string, string >( shellElement, "create", "PyMooseContext", contextName);
    Element* contextElement = Element::lastElement();
    const Finfo* shellFinfo, *contextFinfo;
    shellFinfo = shellElement->findFinfo( "parser" );
    assert(shellFinfo!=NULL);
    
    contextFinfo = contextElement->findFinfo( "parser" );
    assert(contextFinfo!=NULL);
                    
    assert( shellFinfo->add( shellElement, contextElement, contextFinfo) != 0 );
    PyMooseContext* context = static_cast<PyMooseContext*> (contextElement->data());
    context->shell_ = shellElement->id();
    context->myId_ = contextElement->id();
    set<std::string, std::string>( Element::root(), "create", "Neutral", "sched");
    context->scheduler_ = Element::lastElement()->id();
    cout << "PyMooseContext::createPyMooseContext() - scheduler id: " << context->scheduler_ << endl;
    
    set<std::string, std::string>( Element::element(context->scheduler_), "create", "ClockJob", "cj");
    context->clockJob_ =  Element::lastElement()->id();
    cout << "PyMooseContext::createPyMooseContext() - clockjob id: " << context->clockJob_ << endl;
//    set<std::string, std::string>( Element::element(context->clockJob_), "create", "Tick", "t0");
//    context->tick0_ =   Element::lastElement()->id();
//    cout << "PyMooseContext::createPyMooseContext() - tick id: " << context->tick0_ << endl;

    context->setCwe(0); // make initial element = root        

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
    
    Element* ctxt = Element::element(context->myId_);
    
    if ( ctxt == NULL)
    {
        cerr << "ERROR: destroyPyMooseContext(PyMooseContext* context) - Element with id " << context->myId_ << " does not exist" << endl;
        return;
    }
    
    
    unsigned int shellId = context->shell_;
    if (shellId == BAD_ID)
    {
        cerr << "ERROR:  PyMooseContext::destroyPyMooseContext(PyMooseContext* context) - Shell id turns out to be invalid." << endl;
        
    }    
    set(  ctxt, "destroy");
    Element* shell = Element::element(shellId);
    if (shell == NULL)
    {
        cerr <<  "ERROR:  PyMooseContext::destroyPyMooseContext(PyMooseContext* context) - Shell id turns out to be invalid." << endl;
        return;
    }
    else 
    {
        set ( shell, "destroy");   
    }
}

/**
   @param className : Class name of the MOOSE object to be generated.
   @param name : Name of the instance to be generated.
   @param parent - Id of the element under which this should be created
   @returns id of the newly generated object.
*/

Id PyMooseContext::create(std::string className, std::string name, Id parent)
{
    if ( name.length() < 1 ) {
        cerr << "Error: invalid object name : " << name << endl;
        return 0;
    }
    if ( !Cinfo::find( className ) )
    {
        cerr << "Error: could not find any class of name " << className << endl;
        return 0;
    }    
        
    send3 < string, string, unsigned int > (
        Element::element(myId_), createSlot, className, name, parent );
    
    return createdElm_;
}

bool PyMooseContext::destroy( Id victim)
{
    Element* e = Element::element(victim);
    
    if (( victim > myId_ ) && (e != NULL))
    {
        send1< Id >( Element::element(myId_), deleteSlot, victim );
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
    while (createdElm_ > myId_)
    {
        if (Element::element(createdElm_) != NULL)
            destroy(createdElm_);
        cerr << "Destroyed " << createdElm_ << endl;
        
        --createdElm_;        
    }
    
}

Id PyMooseContext::getParent( Id e ) const
{
    unsigned int ret = BAD_ID;
    Element* elm = Element::element( e );
    if (elm == NULL)
    {
        cerr << "ERROR: PyMooseContext::getParent( Id e ) - Element with id " << e << " does not exist." << endl;        
    }    
    else if (e != 0)
    {
        get< unsigned int >( elm, "parent", ret );
    }
    else 
    {
        cerr << "WARNING: PyMooseContext::getParent( Id e ) - 'root' object does not have any parent" << endl;
    }
    
    return ret;
}

std::string PyMooseContext::getPath(Id id) const
{
    std::string path = "";
       
    if ( id == 0 )
    {
        return separator;
    }
    while (id!=0)
    {
        Element* e = Element::element( id );
        if ( e == NULL)
        {
            cerr << "Error: PyMooseContext::getPath(Id id) - Invalid id specified" << endl;
            return "";        
        }
        
        path = separator + e->name() + path;
        id = getParent(id);
    }    
    return path;
}

Id PyMooseContext::pathToId(std::string path, bool echo)
{
    Id returnValue = 0;
    if ( path == separator || path == separator + "root")
    {
        return 0;
    }
    if( path == "" || path == ".")
    {
        return cwe_;
    }
    if (path == "..")
    {
        return (cwe_ == 0)? 0 : Shell::parent(cwe_);
    }
    vector <std::string > nodes;
    unsigned int start;
    if (path.substr(0,separator.length()) == separator)
    {
        start = 0;
        separateString(path.substr(separator.length()), nodes, separator);
    }
    else if (path.substr(separator.length(),4) == "root")
    {
        separateString(path.substr(separator.length()+4), nodes, separator);
    }
    else
    {
        start = cwe_;
        separateString(path, nodes, separator);
    }
    returnValue = Shell::traversePath(start, nodes);
    if (( returnValue == BAD_ID) && echo)
    {
        cerr << "ERROR: PyMooseContext::pathToId(std::string path) - Could not find the object '" << path << "'"<< endl;
    }
    return returnValue;    
}

bool PyMooseContext::exists(Id id)
{
    Element* e = Element::element(id);
    return e != NULL;    
}

bool PyMooseContext::exists(std::string path)
{
    return pathToId(path, false) != BAD_ID;    
}

vector <Id>& PyMooseContext::getChildren(Id id)
{
    elist_.resize(0);
    if ( Element::element(id) != NULL )
    {
        send1< Id >( Element::element( myId_ ), requestLeSlot, id );
    }
    else 
    {
        cerr << "ERROR: PyMooseContext::getChildren(Id id) - Element with Id " << id << " does not exist" << endl;
    }
    
    return elist_;
}

vector <Id>& PyMooseContext::getChildren(std::string path)
{
    elist_.resize(0);    
    Id id = pathToId(path);
    
    /// \todo: Use better test for a bad path than this.
    if ( id == BAD_ID )
    {
        cerr << "ERROR:  PyMooseContext::getChildren(std::string path) - This path seems to be invalid" << endl;
        return elist_;
    }
    send1< Id >( Element::element( myId_ ), requestLeSlot, id );
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
    Element* e = Element::element( myId_ );

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
    
    send0( Element::element(myId_), requestClocksSlot ); 
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
    send3< int, double, int >(Element::element(myId_), setClockSlot, clockNo, dt, stage);
}


vector <double> & PyMooseContext::getClocks()
{
    send0( Element::element(myId_), requestClocksSlot );
    return dbls_;        
}

void PyMooseContext::useClock(Id tickId, std::string path, std::string func)
{
    Element * e = Element::element(myId_);
    send2< string, bool >( e, requestWildcardListSlot, path, 0 );
    send3< unsigned int, vector< unsigned int >, string >(
        Element::element( myId_ ),
        useClockSlot, 
        tickId, elist_,  func );
}

void PyMooseContext::reset()
{
    send0(Element::element(myId_), reschedSlot);
    send0(Element::element(myId_), reinitSlot);    
}

void PyMooseContext::stop()
{
    send0( Element::element( myId_ ), stopSlot );
}

void PyMooseContext::addTask(std::string arg)
{
    //Do nothing
}
/**
   This just does the copying without returning anything.
   Corresponds to the procedural technique used in Genesis shell
*/
void PyMooseContext::do_deep_copy( Id object, std::string new_name, Id dest)
{
    send3< Id, Id, std::string >  ( Element::element(myId_), copySlot, object, dest, new_name);
}
/**
   This is the object oriented version. It returns Id of new copy.
*/
Id PyMooseContext::deepCopy( Id object, std::string new_name, Id dest)
{
    do_deep_copy( object,  new_name, dest);
    
    std::string path = getPath(dest) + PyMooseContext::separator+new_name;
    return pathToId(path);
}

void PyMooseContext::move( Id object, std::string new_name, Id dest)
{
    send3< Id, Id, string >(
        Element::element( myId_), moveSlot, object, dest, new_name );
}

bool PyMooseContext::connect(Id src, std::string srcField, Id dest, std::string destField)
{
    if ( src != BAD_ID && dest != BAD_ID ) {
        Element* se = Element::element( src );
        Element* de = Element::element( dest );
        const Finfo* sf = se->findFinfo( srcField );
        if ( !sf ) return false;
        const Finfo* df = de->findFinfo( destField );
        if ( !df ) return false;
        return (bool)(se->findFinfo( srcField )->add( se, de, de->findFinfo( destField )));
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

Id PyMooseContext::findChanGateId( std::string channel, std::string gate)
{
    std::string path = "";
    if ( gate.at(0) == 'X' )
        path = channel + "/xGate";
    else if ( gate.at(0) == 'Y' )
        path = channel + "/yGate";
    else if ( gate.at(0)   == 'Z' )
        path = channel + "/zGate";
    Id gateId = pathToId( path);
    if ( gateId == BAD_ID ) // Don't give up, it might be a tabgate
        gateId = pathToId( channel);
    if ( gateId == BAD_ID ) { // Now give up
        cout << "Error: findChanGateId: unable to find channel/gate '" << channel << "/" << gate << endl;
        return BAD_ID;
    }
    return gateId;
}

void PyMooseContext::setupChanFunc(std::string channel, std::string gate, vector <double>& parms, const unsigned int& slot)
{
    
    if (parms.size() < 10 ) {
        cerr << "Error: PyMooseContext::setupChanFunc() -  We need a vector for these items: AA AB AC AD AF BA BB BC BD BF size min max (length should be at least 10)" << endl;
        return;
    }

    Id gateId = findChanGateId(channel, gate );
    if (gateId == BAD_ID )
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
    
    send2< Id, vector< double > >( Element::element( myId_ ), slot, gateId, parms );
}

void PyMooseContext::setupAlpha( std::string channel, std::string gate, vector <double> parms ) 
{
    setupChanFunc( channel, gate, parms, setupAlphaSlot );
}

void PyMooseContext::setupTau( std::string channel, std::string gate, vector <double> parms ) 
{
    setupChanFunc( channel, gate, parms, setupTauSlot );
}

void PyMooseContext::tweakChanFunc( std::string  channel, std::string gate, unsigned int slot )
{
    Id gateId = findChanGateId( channel, gate );
    if ( gateId == BAD_ID )
        return;
    send1< Id >( Element::element( myId_ ), slot, gateId );
}

void PyMooseContext::tweakAlpha( std::string channel, std::string gate ) 
{
    tweakChanFunc( channel, gate, tweakAlphaSlot );
}
void PyMooseContext::tweakTau( std::string channel, std::string gate)
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
    if (gateId == BAD_ID )
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
    
    send2< Id, vector< double > >( Element::element( myId_ ), slot, gateId, parms );
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
    if ( gateId == BAD_ID )
        return;
    send1< Id >( Element::element( myId_ ), slot, gateId );
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
    std::string argstr = xdivs + "," + mode;
    send3< Id, string, string >( Element::element( myId_ ), setFieldSlot, table, "tabFill", argstr );
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
    std::string setValue;    
    std::string getValue;
    double eps = 1e-5; // usual error (for std::string conversion) on PC is 1e-6
    
    Id obj = create("Compartment", "TestSetGetCompartment", 0);
    
    for ( i = 0; i < count; ++i )
    {
        double d = (i+1)/3.0e-6;
        double val;
        
        setValue = toString < double > (d);
        
        setField(obj, "Rm", setValue);
        getValue = getField(obj, "Rm");

        
        val = atof(getValue.c_str());        
        testResult = testResult && ((d > val)? (1-val/d) < eps : (1 - d/val) < eps);
        if (testResult == false)
        {
            cerr << "TEST:: PyMooseContext::testSetGetField((int count, bool doPrint) - set value " << setValue << ", retrieved value " << getValue << ". Actual set value: "<< d << ", retrieved value: "<< val << ", error: " << 1-val/d << ", allowed error: " << eps << " - FAILED" << endl ;            
        }
    }
    destroy(obj);    
    return testResult;
}

bool PyMooseContext::testSetGetField(std::string className, std::string fieldName, std::string fieldValue, int count, bool doPrint)
{

    std::string retrievedVal;
    bool testResult = true;
    Id obj = create(className,"TestSetGetField", getCwe());    
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

bool PyMooseContext::testCreateDestroy(std::string className, int count, bool doPrint)
{
    vector <Id> idList;
    int i;
    Id instanceId;
    
    for (i = 0; i < count; ++i)
    {
        instanceId = create(className, "test"+toString < int > (i), 0);
        idList.push_back(instanceId);
    }
       
    while (idList.size() > 0)
    {
        instanceId = (Id)(idList.back());
        idList.pop_back();
        destroy(instanceId);
        --i;        
    }
    if (doPrint)
    {
        cerr << "TEST::  PyMooseContext::testCreateDestroy(std::string className, int count, bool doPrint) - create and destroy " << count << " " << className << " instances ... " << ((i == 0)? "SUCCESS":"FAILED") << endl;
    }
    
    return ( i == 0 );     
}

bool PyMooseContext::testPyMooseContext(int testCount, bool doPrint)
{
    bool overallResult = true;    
    bool testResult;
    PyMooseContext *context = createPyMooseContext("TestContext", "TestShell");
    
    testResult = context->testCreateDestroy("Compartment", testCount, doPrint);
    overallResult = overallResult && testResult;
    
    if(doPrint){
        cerr << "TEST::  PyMooseContext::testPyMooseContext(int testCount, bool doPrint) - testing create and destroy ... " << (testResult?"SUCCESS":"FAILED") << endl;
    }
    testResult = context->testSetGetField(testCount, doPrint);
    overallResult = overallResult && testResult;
    
    if(doPrint){
        cerr << "TEST:: PyMooseContext::testPyMooseContext(int testCount, bool doPrint) - testing set and get ... " << (testResult?"SUCCESS":"FAILED") << endl;
    }
    
    delete context;
    
    return overallResult;
}
#endif // DO_UNIT_TESTS

#endif // _PYMOOSE_CONTEXT_CPP
