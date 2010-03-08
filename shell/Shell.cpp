/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "IdManager.h"
#include "../element/Neutral.h"
#include "../element/Wildcard.h"
#include "Shell.h"
#include "ReadCell.h"
#include "SimDump.h"
#include "Ftype3.h"
#include "../randnum/Probability.h"
#include "../randnum/Uniform.h"
#include "../randnum/Exponential.h"
#include "../randnum/Normal.h"
#include "math.h"
#include "sstream"
#ifdef USE_SBML
#include "sbml_IO/SbmlReader.h"
#include "sbml_IO/SbmlWriter.h"
#endif
#ifdef USE_NEUROML
#include "neuroML_IO/NeuromlReader.h"
#include "neuroML_IO/NeuromlWriter.h"
#endif
extern void pollPostmaster(); // Defined in maindir/init.cpp
//////////////////////////////////////////////////////////////////////
// Shell static initializers
//////////////////////////////////////////////////////////////////////

unsigned int Shell::myNode_ = 0;
unsigned int Shell::numNodes_ = 1;
bool Shell::running_ = 1;
const unsigned int Shell::maxNumOffNodeRequests = 100;

//////////////////////////////////////////////////////////////////////
// Shell MOOSE object creation stuff
//////////////////////////////////////////////////////////////////////

const Cinfo* initShellCinfo()
{
	static Finfo* parserShared[] =
	{
		new DestFinfo( "cwe", Ftype1< Id >::global(),
						RFCAST( &Shell::setCwe ),
						"Setting cwe" ),
		new DestFinfo( "trigCwe", Ftype0::global(), 
						RFCAST( &Shell::trigCwe ),
						"Getting cwe back: First handle a request" ),
		new SrcFinfo( "cweSrc", Ftype1< Id >::global(),
						" Then send out the cwe info" ),

		new DestFinfo( "pushe", Ftype1< Id >::global(),
						RFCAST( &Shell::pushe ),
						"doing pushe: pushing current element onto stack and using argument "
						"for new cwe.It sends back the cweSrc." ),
		new DestFinfo( "pope", Ftype0::global(), 
						RFCAST( &Shell::pope ),
						"Doing pope: popping element off stack onto cwe. It sends back the cweSrc." ),

		new DestFinfo( "trigLe", Ftype1< Id >::global(), 
						RFCAST( &Shell::trigLe ),
						"Getting a list of child ids: First handle a request with the requested parent elm id." ),
		new SrcFinfo( "leSrc", Ftype1< vector< Id > >::global(),
						"Then send out the vector of child ids." ),
		
		new DestFinfo( "create",
				Ftype4< string, string, int, Id >::global(),
				RFCAST( &Shell::staticCreate ),
				"Creating an object type, name, node, parent." ),
		new DestFinfo( "createArray",
				Ftype4< string, string, Id, vector<double> >::global(),
				RFCAST( &Shell::staticCreateArray ),
				"Creating an array of objects" ),
		new DestFinfo( "planarconnect",
				Ftype3< string, string, double >::global(),
				RFCAST( &Shell::planarconnect ) ),
		new DestFinfo( "planardelay",
				Ftype3< string, string, vector <double> >::global(),
				RFCAST( &Shell::planardelay ) ),
		new DestFinfo( "planarweight",
				Ftype3< string, string, vector<double> >::global(),
				RFCAST( &Shell::planarweight ) ),
		new DestFinfo( "getSynCount",
				Ftype1< Id >::global(),
				RFCAST( &Shell::getSynCount2 ) ),
		new SrcFinfo( "createSrc", Ftype1< Id >::global(),
				"The create func returns the id of the created object." ),
		new DestFinfo( "delete", Ftype1< Id >::global(), 
				RFCAST( &Shell::staticDestroy ),
				"Deleting an object" ),
		new DestFinfo( "addField",
				Ftype2< Id, string >::global(),
				RFCAST( &Shell::addField ),
				"Create a dynamic field on specified object." ),
		new DestFinfo( "get",
				Ftype2< Id, string >::global(),
				RFCAST( &Shell::getField ),
				"Getting a field value as a string: handling request" ),
		new SrcFinfo( "getSrc", Ftype1< string >::global(),
				"Getting a field value as a string: Sending value back." ),

		new DestFinfo( "set",
				Ftype3< Id, string, string >::global(),
				RFCAST( &Shell::setField ),
				"Setting a field value as a string: handling request" ),
		new DestFinfo( "file2tab",
				Ftype3< Id, string, unsigned int >::global(),
				RFCAST( &Shell::file2tab ),
				"Assigning a file into a table. ElementId, filename, skiplines" ),

		new DestFinfo( "setClock",
				Ftype3< int, double, int >::global(),
				RFCAST( &Shell::setClock ),
				"Handle requests for setting values for a clock tick.args are clockNo, dt, stage" ),

		new DestFinfo( "useClock",
				Ftype3< string, string, string >::global(),
				RFCAST( &Shell::useClock ),
				"Handle requests to assign a path to a given clock tick.args are tickname, path, function" ),
		
		new DestFinfo( // args are path, flag true for breadth-first list
				"el",
				Ftype2< string, bool >::global(),
				RFCAST( &Shell::getWildcardList ),
				"Getting a wildcard path of elements: handling request" ),
		// Getting a wildcard path of elements: Sending list back.
		// This goes through the exiting list for elists set up in le.
		//TypeFuncPair( Ftype1< vector< Id > >::global(), 0 ),

		////////////////////////////////////////////////////////////
		// Running simulation set
		////////////////////////////////////////////////////////////
		new DestFinfo( "resched",
				Ftype0::global(), RFCAST( &Shell::resched ) ),
		new DestFinfo( "reinit",
				Ftype0::global(), RFCAST( &Shell::reinit ) ),
		new DestFinfo( "stop",
				Ftype0::global(), RFCAST( &Shell::stop ) ),
		new DestFinfo( "step",
				Ftype1< double >::global(), // Arg is runtime
				RFCAST( &Shell::step ) ),
		new DestFinfo( "requestClocks",
				Ftype0::global(), &Shell::requestClocks ),
		new SrcFinfo( "returnClocksSrc",
			Ftype1< vector< double > >::global(),
			"Sending back the list of clocks times" ),
		new DestFinfo( "requestCurrTime",
				Ftype0::global(), RFCAST( &Shell::requestCurrTime ) ),
		new DestFinfo( "quit", Ftype0::global(), RFCAST( &Shell::quit ),
			"Returns it in the default string return value.Call this to end the simulation." ),	
		////////////////////////////////////////////////////////////
		// Message functions
		////////////////////////////////////////////////////////////

		new DestFinfo( "addMessage",
				Ftype4< vector< Id >, string, vector< Id >, string >::global(),
				RFCAST( &Shell::addMessage ) ),
		new DestFinfo( "deleteMessage",
				Ftype2< Fid, int >::global(),
				RFCAST( &Shell::deleteMessage ) ),
		new DestFinfo( "deleteEdge",
				Ftype4< Id, string, Id, string >::global(),
				RFCAST( &Shell::deleteEdge ) ),

		new DestFinfo( "listMessages",
				Ftype3< Id, string, bool >::global(),
				RFCAST( &Shell::listMessages ),
				"Handle request for message list: id elm, string field, bool isIncoming" ),
		new SrcFinfo( "listMessagesSrc",
			Ftype2< vector < Id >, string >::global(),
			"Return message list and string with remote fields for msgs" ),

		////////////////////////////////////////////////////////////
		// Object heirarchy manipulation functions
		////////////////////////////////////////////////////////////
		new DestFinfo( "copy",
			Ftype3< Id, Id, string >::global(),
			RFCAST( &Shell::copy ) ),
		new DestFinfo( "copyIntoArray",
			Ftype4< Id, Id, string, vector <double> >::global(),
			RFCAST( &Shell::copyIntoArray ) ),
		new DestFinfo( "move",
			Ftype3< Id, Id, string >::global(),
			RFCAST( &Shell::move ) ),
		////////////////////////////////////////////////////////////
		// Cell reader
		////////////////////////////////////////////////////////////
		
		new DestFinfo( "readcell",
			Ftype4< string, string, vector< double >, int >::global(),
			RFCAST( &Shell::readCell ),
			"Args are: file cellpath globalParms node" ),
		////////////////////////////////////////////////////////////
		// Channel setup functions
		////////////////////////////////////////////////////////////
		new DestFinfo( "setupAlpha",
			Ftype2< Id, vector< double > >::global(), 
					RFCAST( &Shell::setupAlpha ) ),
		new DestFinfo( "setupTau",
			Ftype2< Id, vector< double > >::global(), 
					RFCAST( &Shell::setupTau ) ),
		new DestFinfo( "tweakAlpha",
			Ftype1< Id >::global(), RFCAST( &Shell::tweakAlpha ) ),
		new DestFinfo( "tweakTau",
			Ftype1< Id >::global(), RFCAST( &Shell::tweakTau ) ),
		new DestFinfo( "setupGate",
			Ftype2< Id, vector< double > >::global(), 
					RFCAST( &Shell::setupGate ) ),
		////////////////////////////////////////////////////////////
		// SimDump facility
		////////////////////////////////////////////////////////////
		new DestFinfo(	"readDumpFile",
			Ftype1< string >::global(), // arg is filename
					RFCAST( &Shell::readDumpFile ) ),
		new DestFinfo(	"writeDumpFile",
			// args are filename, path to dump
			Ftype2< string, string >::global(), 
					RFCAST( &Shell::writeDumpFile ) ),
		new DestFinfo(	"simObjDump",
			// arg is a set of fields for the desired class
			// The list of fields is a space-delimited list and 
			// can be extracted using separateString.
			Ftype1< string >::global(), RFCAST( &Shell::simObjDump ) ),
		new DestFinfo(	"simUndump",
					// args is sequence of args for simundump command.
			Ftype1< string >::global(), RFCAST( &Shell::simUndump ) ),
		new DestFinfo( "openfile",
				Ftype2< string, string >::global(),
				RFCAST( &Shell::openFile ) ),
		new DestFinfo( "writefile",
				Ftype2< string, string >::global(),
				RFCAST( &Shell::writeFile ) ),
		new DestFinfo( "flushfile",
				Ftype1< string >::global(),
				RFCAST( &Shell::flushFile ) ),
		new DestFinfo( "listfiles",
				Ftype0::global(),
				RFCAST( &Shell::listFiles ) ),
		new DestFinfo( "closefile",
				Ftype1< string >::global(),
				RFCAST( &Shell::closeFile ) ),	
		new DestFinfo( "readfile",
				Ftype2< string, bool >::global(),
				RFCAST( &Shell::readFile) ),	
		////////////////////////////////////////////////////////////
		// field assignment for a vector of objects
		////////////////////////////////////////////////////////////
		new DestFinfo( "setVecField",
				Ftype3< vector< Id >, string, string >::global(),
				RFCAST( &Shell::setVecField ),
				"Setting a field value as a string: handling request" ),
		new DestFinfo( "loadtab",
				Ftype1< string >::global(),
				RFCAST( &Shell::loadtab ) ),	
		new DestFinfo( "tabop",
				Ftype4< Id, char, double, double >::global(),
				RFCAST( &Shell::tabop ) ),
		////////////////////////////////////////////////////////////
		// SBML
		////////////////////////////////////////////////////////////
		new DestFinfo( "readsbml",
				Ftype3< string, string, int >::global(),
				RFCAST( &Shell::readSbml ) ),	
		new DestFinfo( "writesbml",
				Ftype3< string, string, int >::global(),
				RFCAST( &Shell::writeSbml ) ),		
		////////////////////////////////////////////////////////////
		// NeuroML
		////////////////////////////////////////////////////////////
		new DestFinfo( "readNeuroML",
				Ftype3< string, string, int >::global(),
				RFCAST( &Shell::readNeuroml ) ),	
		new DestFinfo( "writeNeuroML",
				Ftype3< string, string, int >::global(),
				RFCAST( &Shell::writeNeuroml ) ),		
		////////////////////////////////////////////////////////////
		// Misc
		////////////////////////////////////////////////////////////
		new DestFinfo( "createGate", 
			Ftype2< Id, string >::global(),
			RFCAST( &Shell::createGateMaster ),
			"Args: Channel id, gate type (X, Y or Z) "
			"Request an HHChannel to create a gate on it." ),
	};
	static Finfo* serialShared[] =
	{
		new DestFinfo( "rawAdd", // Addmsg as a raw string.
			Ftype1< string >::global(),
			RFCAST( &Shell::rawAddFunc )
		),
		new DestFinfo( "rawCopy", // Copy an entire object sent as a string
			Ftype1< string >::global(),
			RFCAST( &Shell::rawCopyFunc )
		),
		new DestFinfo( "rawTest", // Test function
			Ftype1< string >::global(),
			RFCAST( &Shell::rawTestFunc )
		),
	};

#ifdef USE_MPI
	static Finfo* parallelShared[] = 
	{
		/////////////////////////////////////////////////////
		// get stuff
		/////////////////////////////////////////////////////
		new SrcFinfo( "getSrc",
			// objId, field, requestId
			Ftype3< Id, string, unsigned int >::global() ),
		new DestFinfo( "get",
			// objId, field, requestId
			Ftype3< Id, string, unsigned int >::global(),
			RFCAST( &Shell::parGetField )
			),
		new SrcFinfo( "returnGetSrc",
			// return string, requestId
			Ftype2< string, unsigned int >::global()
		),
		new DestFinfo( "returnGet",
			Ftype2< string, unsigned int >::global(),
			RFCAST( &Shell::handleReturnGet )
		),
		/////////////////////////////////////////////////////
		// addfield
		/////////////////////////////////////////////////////
		new SrcFinfo( "addfieldSrc",
			Ftype2< Id, string >::global(),
			"Args: Id, fieldname"
		),
		new DestFinfo( "addfield",
			Ftype2< Id, string >::global(),
			RFCAST( &Shell::localAddField ),
			"Args: Id, fieldname"
		),
		/////////////////////////////////////////////////////
		// set stuff
		/////////////////////////////////////////////////////
		new SrcFinfo( "setSrc",
			// objId, field, value
			Ftype3< Id, string, string >::global() ),
		new DestFinfo( "set",
			// objId, field, value
			Ftype3< Id, string, string >::global(),
			RFCAST( &Shell::localSetField )
		),

		/////////////////////////////////////////////////////
		// Creationism, and its converse. Note that the createSrc
		// should only come
		// from the master node as it is the only one that knows
		// the next child index.
		/////////////////////////////////////////////////////
		new SrcFinfo( "createSrc", 
			// type, name, parentId, newObjId.
			Ftype4< string, string, Nid, Nid >::global()
		),
		new DestFinfo( "create",
				Ftype4< string, string, Nid, Nid >::global(),
				RFCAST( &Shell::parCreateFunc ),
				"Creating an object" ),
		new SrcFinfo( "createArraySrc", 
			// type, name, parentId, newObjId.
			Ftype4< string, string, pair< Nid, Nid >, 
				vector< double > >::global()
		),
		new DestFinfo( "createArray",
				Ftype4< string, string, pair< Nid, Nid >, 
					vector< double > >::global(),
				RFCAST( &Shell::parCreateArrayFunc ),
				" Creating an object" ),
		new SrcFinfo( "deleteSrc", 
			// type, name, parentId, newObjId.
			Ftype1< Id >::global()
		),
		new DestFinfo( "delete",
				Ftype1< Id >::global(),
				RFCAST( &Shell::staticDestroy ),
				" Creating an object" ),

		new SrcFinfo( "copySrc", 
			Ftype4<
				Nid,
				Nid,
				string,
				IdGenerator
			>::global(),
			"Send copy request to remote (src) node. " 
			"Args: srcId, parentId, name, IdGenerator for children." ),
		
		new DestFinfo( "copy",
			Ftype4<
				Nid,
				Nid,
				string,
				IdGenerator
			>::global(),
			RFCAST( &Shell::parCopy ),
			"Copying an object. "
			"Args: srcId, parentId, name, IdGenerator for children." ),
		
		new SrcFinfo( "copyIntoArraySrc", 
			Ftype5<
				Nid,
				Nid,
				string,
				vector< double >,
				IdGenerator
			>::global(),
			"Send array copy request to remote (src) node. "
			"Args: srcId, parentId, name, array parameters, IdGenerator for children." ),
		
		new DestFinfo( "copyIntoArray",
			Ftype5<
				Nid,
				Nid,
				string,
				vector< double >,
				IdGenerator
			>::global(),
			RFCAST( &Shell::parCopyIntoArray ),
			"Copying into an array. "
			"Args: srcId, parentId, name, array parameters, IdGenerator for children." ),
		
		new SrcFinfo( "readcellSrc",
			Ftype5<
				string,
				string,
				vector< double >,
				Nid,
				IdGenerator
			>::global(),
			"Args are: file, cellpath, globalParms, parentId, IdGenerator for children." ),
		
		new DestFinfo( "readcell",
			Ftype5<
				string,
				string,
				vector< double >,
				Nid,
				IdGenerator
			>::global(),
			RFCAST( &Shell::localReadCell ),
			"Args are: file, cellpath, globalParms, parentId, IdGenerator for children." ),
		
		/////////////////////////////////////////////////////
		// Msg stuff
		/////////////////////////////////////////////////////
		new SrcFinfo( "addLocalSrc",
			Ftype4< Id, string, Id, string >::global(),
			"Args: srcObjId, srcField, destObjId, destField" ),
		new SrcFinfo( "addParallelSrcSrc",
			Ftype4< Nid, string, Nid, string >::global(),
			"Args: srcObjId, srcField, destObjId, destField" ),
		new SrcFinfo( "addParallelDestSrc",
			Ftype5< Nid, unsigned int, string, Nid, string >::global(),
			"Args: srcObjId, srcSize, srcField, destObjId, destField" ),
		new DestFinfo( "addLocal",
			Ftype4< Id, string, Id, string >::global(),
			RFCAST( &Shell::addLocal ) ),
		new DestFinfo( "addParallelSrc",
			Ftype4< Nid, string, Nid, string >::global(),
			RFCAST( &Shell::addParallelSrc ) ),
		new DestFinfo( "addParallelDest",
			Ftype5< Nid, unsigned int, string, Nid, string >::global(),
			RFCAST( &Shell::addParallelDest ) ),

		/////////////////////////////////////////////////////
		// Msg completion reporting stuff
		/////////////////////////////////////////////////////
		new SrcFinfo( "parMsgErrorSrc",
			Ftype3< string, Id, Id >::global() ),
		new SrcFinfo( "parMsgOkSrc",
			Ftype2< Id, Id >::global() ),
		new DestFinfo( "parMsgError",
			Ftype3< string, Id, Id >::global(),
			RFCAST( &Shell::parMsgErrorFunc ) ),
		new DestFinfo( "parMsgOk",
			Ftype2< Id, Id >::global(),
			RFCAST( &Shell::parMsgOkFunc ) ),

		///////////////////////////////////////////////////////////
		// This little section deals with requests to traverse the
		// path string looking for an Id on a remote node.
		///////////////////////////////////////////////////////////
		new SrcFinfo( "parTraversePathSrc",
			Ftype3< Id, vector< string >, unsigned int >::global(),
			"Args are: Start, names, requestId" ),
		new SrcFinfo( "parTraversePathReturnSrc",
			Ftype2< Nid, unsigned int >::global(),
			"args are: foundId, requestId" ),
		new DestFinfo( "parTraversePath",
			Ftype3< Id, vector< string >, unsigned int >::global(),
			RFCAST( &Shell::handleParTraversePathRequest ) ),
		new DestFinfo( "parTraversePathReturn",
			Ftype2< Nid, unsigned int >::global(),
			RFCAST( &Shell::handleParTraversePathReturn ) ),

		///////////////////////////////////////////////////////////
		// This section is the converse: Look for the path of an
		// eid on a remote node.
		///////////////////////////////////////////////////////////
		new SrcFinfo( "requestPathSrc",
			Ftype2< Nid, unsigned int >::global() ),
		new DestFinfo( "requestPath",
			Ftype2< Nid, unsigned int >::global(),
			RFCAST( &Shell::handlePathRequest ) ),
		new SrcFinfo( "returnPathSrc",
			Ftype2< string, unsigned int >::global() ),
		new DestFinfo( "returnPath",
			Ftype2< string, unsigned int >::global(),
			RFCAST( &Shell::handlePathReturn ) ),

		///////////////////////////////////////////////////////////
		// This section deals with requests for le and wildcards 
		///////////////////////////////////////////////////////////
		new SrcFinfo( "requestLeSrc",
			Ftype2< Nid, unsigned int >::global(),
			"args are: parent, requestId" ),
		new DestFinfo( "requestLe",
			Ftype2< Nid, unsigned int >::global(),
			RFCAST( &Shell::handleRequestLe ) ),
		new SrcFinfo( "returnLeSrc",
			Ftype2< vector< Nid >, unsigned int >::global(),
			"args are: elist, requestId" ),
		new DestFinfo( "returnLe",
			Ftype2< vector< Nid >, unsigned int >::global(),
			RFCAST( &Shell::handleReturnLe ) ),

		new SrcFinfo( "requestParWildcardSrc",
			Ftype3< string, bool, unsigned int >::global(),
			" args are: path, ordered, requestId" ),
		new DestFinfo( "requestParWildcard",
			Ftype3< string, bool, unsigned int >::global(),
			RFCAST( &Shell::handleParWildcardList ) ),

		///////////////////////////////////////////////////////////////
		// This section handles scheduling and execution control
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "reschedSrc", Ftype0::global() ),
		new SrcFinfo( "reinitSrc", Ftype0::global() ),
		new SrcFinfo( "stopSrc", Ftype0::global() ),
		new SrcFinfo( "stepSrc", Ftype1< double >::global(),
			"Arg is runtime" ),
		new DestFinfo( "resched",
				Ftype0::global(), RFCAST( &Shell::resched ) ),
		new DestFinfo( "reinit",
				Ftype0::global(), RFCAST( &Shell::reinit ) ),
		new DestFinfo( "stop",
				Ftype0::global(), RFCAST( &Shell::stop ) ),
		new DestFinfo( "step",
				Ftype1< double >::global(), // Arg is runtime
				RFCAST( &Shell::step ) ),
		new SrcFinfo( "setClockSrc",
				Ftype3< int, double, int >::global(),
				"Handle requests for setting values for a clock tick.args are clockNo, dt, stage" ),
		new DestFinfo( "setClock",
				Ftype3< int, double, int >::global(),
				RFCAST( &Shell::setClock ) ),
		new SrcFinfo( "useClockSrc",
				Ftype3< string, string, string >::global(),
				"Handle requests to assign a path to a given clock tick.args are tickname, path, function" ),
		new DestFinfo( "useClock",
				Ftype3< string, string, string >::global(),
				RFCAST( &Shell::localUseClock ) ),
		new SrcFinfo( "quitSrc", Ftype0::global(),
				"Called to terminate simulation." ),
		new DestFinfo( "quit", Ftype0::global(), 
			RFCAST( &Shell::innerQuit ) ),
		
		///////////////////////////////////////////////////////////////
		// Id management
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "requestIdBlockSrc",
			Ftype3< unsigned int, unsigned int, unsigned int >::global(),
			"Here a slave node requests the master node for a block of main "
			"Ids that it can use locally. "
			"Args: size of block, requesting node, request Id" ),
		new DestFinfo( "requestIdBlock",
			Ftype3< unsigned int, unsigned int, unsigned int >::global(),
			RFCAST( &Shell::handleRequestNewIdBlock ),
			"Here the master node intercepts requests from slave nodes for "
			"a new Id block. "
			"Args: size of block, requesting node, request Id" ),
		new SrcFinfo( "returnIdBlockSrc",
			Ftype2< unsigned int, unsigned int >::global(),
			"Respond to requests from slave nodes for a new Id block. "
			"Args: Base id in alloted block, request Id" ),
		new DestFinfo( "returnIdBlock",
			Ftype2< unsigned int, unsigned int >::global(),
			RFCAST( &Shell::handleReturnNewIdBlock ),
			"Receive an Id block from master node. "
			"Args: Base id in alloted block, request Id" ),
		
		///////////////////////////////////////////////////////////////
		// Misc
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "createGateSrc",
			Ftype3< Id, string, IdGenerator >::global(),
			"Args: HHChannel id, Gate type (X/Y/Z), IdGenerator for children. "
			"This requests a global HHChannel to create an HHGate, and its "
			"Interpols (i.e., rate lookup tables). The IdGenerator is used for "
			"assigning ids to these 3 objects. This is necessary for aligning "
			"ids of the children across nodes, since they will be globals too." ),
		new DestFinfo( "createGate",
			Ftype3< Id, string, IdGenerator >::global(),
			RFCAST( &Shell::createGateWorker ),
			"Args: HHChannel id, Gate type (X/Y/Z), IdGenerator for children. "
			"Handles request sent from the createGateSrc Finfo." ),
		
		new SrcFinfo( "setupAlphaSrc",
			Ftype2< Id, vector< double > >::global() ),
		new DestFinfo( "setupAlpha",
			Ftype2< Id, vector< double > >::global(), 
			RFCAST( &Shell::setupAlpha ) ),
		new SrcFinfo( "setupTauSrc",
			Ftype2< Id, vector< double > >::global() ),
		new DestFinfo( "setupTau",
			Ftype2< Id, vector< double > >::global(), 
			RFCAST( &Shell::setupTau ) ),
		new SrcFinfo( "tweakAlphaSrc",
			Ftype1< Id >::global() ),
		new DestFinfo( "tweakAlpha",
			Ftype1< Id >::global(),
			RFCAST( &Shell::tweakAlpha ) ),
		new SrcFinfo( "tweakTauSrc",
			Ftype1< Id >::global() ),
		new DestFinfo( "tweakTau",
			Ftype1< Id >::global(),
			RFCAST( &Shell::tweakTau ) ),
		new SrcFinfo( "setupGateSrc",
			Ftype2< Id, vector< double > >::global() ),
		new DestFinfo( "setupGate",
			Ftype2< Id, vector< double > >::global(), 
			RFCAST( &Shell::setupGate ) ),
		
		new SrcFinfo( "file2tabSrc",
			Ftype3< Nid, string, unsigned int >::global(),
			"Args: Interpol id, filename, # of lines to skip" ),
		new DestFinfo( "file2tab",
			Ftype3< Nid, string, unsigned int >::global(),
			RFCAST( &Shell::localFile2tab ),
			"Args: Interpol id, filename, # of lines to skip" ),
	};
#endif // USE_MPI

	/**
	 * This handles parallelization: communication between shells on
	 * different nodes, which is the major mechanism for setting up
	 * multinode simulations
	static Finfo* slaveShared[] = 
	{
		masterShared[1], masterShared[0],  // note order flip.
		masterShared[2], masterShared[3],
		masterShared[4], masterShared[5], 
		masterShared[6], masterShared[7],
		masterShared[8], masterShared[9], 
		masterShared[10], masterShared[11],
		masterShared[12], masterShared[13],
		masterShared[14], masterShared[15],
		masterShared[16], masterShared[17],
	};
	 */

	static Finfo* shellFinfos[] =
	{
		new ValueFinfo( "cwe", ValueFtype1< Id >::global(),
				GFCAST( &Shell::getCwe ),
				RFCAST( &Shell::setCwe ) ),

		new ValueFinfo( "numNodes", ValueFtype1< int >::global(),
				GFCAST( &Shell::getNumNodes ),
				&dummyFunc ),

		new ValueFinfo( "myNode", ValueFtype1< int >::global(),
				GFCAST( &Shell::getMyNode ),
				&dummyFunc ),

		new DestFinfo( "xrawAdd", // Addmsg as a raw string.
			Ftype1< string >::global(),
			RFCAST( &Shell::rawAddFunc )
		),
		new DestFinfo( "add", // Simple addmsg.
			Ftype4< Id, string, Id, string >::global(),
			RFCAST( &Shell::addSingleMessage ) ),
		new DestFinfo( "poll", // Infinite loop, meant for slave nodes
			Ftype0::global(),
			RFCAST( &Shell::pollFunc )
		),
		new DestFinfo( "createGate", 
			Ftype2< Id, string >::global(),
			RFCAST( &Shell::createGateMaster ),
			"Args: Channel id, gate type (X, Y or Z). "
			"Request an HHChannel to create a gate on it." ),
		new SrcFinfo( "pollSrc", 
			// # of steps. 
			// This talks to /sched/pj:step to poll the postmasters
			Ftype1< int >::global()
		),

		new SharedFinfo( "parser", parserShared, 
				sizeof( parserShared ) / sizeof( Finfo* ),
				"This is a shared message to talk to the GenesisParser and perhaps to other parsers "
				"like the one for SWIG and Python" ), 
		new SharedFinfo( "serial", serialShared,
				sizeof( serialShared ) / sizeof( Finfo* ),
				"This handles serialized data, typically between nodes. The arguments are a single "
				"long string.Takes care of low-level operations such as message set up or the gory "
				"details of copies across nodes." ), 
#ifdef USE_MPI
		new SharedFinfo( "parallel", parallelShared,
				sizeof( parallelShared ) / sizeof( Finfo* ),
				"This handles parallelization: communication between shells on different nodes, "
				"which is the major mechanism for setting up multinode simulations " ),  
#endif // USE_MPI
		/*
		new SharedFinfo( "slave", slaveShared,
				sizeof( slaveShared ) / sizeof( Finfo* ) ), 
		*/
	};

	static string doc[] =
	{
		"Name", "Shell",
		"Author","Upi Bhalla, NCBS",
		"Description", "Shell object. Manages general simulator commands.",
	};
	static Cinfo shellCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),
		initNeutralCinfo(),
		shellFinfos,
		sizeof( shellFinfos ) / sizeof( Finfo* ),
		ValueFtype1< Shell >::global()
	);

	return &shellCinfo;
}

static const Cinfo* shellCinfo = initShellCinfo();


static const Slot cweSlot = initShellCinfo()->getSlot( "parser.cweSrc" );
static const Slot elistSlot = initShellCinfo()->getSlot( "parser.leSrc" );

// Returns the id of the created object
static const Slot createSlot =
	initShellCinfo()->getSlot( "parser.createSrc" );
static const Slot getFieldSlot =
	initShellCinfo()->getSlot( "parser.getSrc" );
static const Slot clockSlot =
	initShellCinfo()->getSlot( "parser.returnClocksSrc" );
static const Slot listMessageSlot =
	initShellCinfo()->getSlot( "parser.listMessagesSrc" );
static const Slot pollSlot =
	initShellCinfo()->getSlot( "pollSrc" );

#ifdef USE_MPI
static const Slot createGateSlot =
	initShellCinfo()->getSlot( "parallel.createGateSrc" );
static const Slot setupAlphaSlot =
	initShellCinfo()->getSlot( "parallel.setupAlphaSrc" );
static const Slot setupTauSlot =
	initShellCinfo()->getSlot( "parallel.setupTauSrc" );
static const Slot tweakAlphaSlot =
	initShellCinfo()->getSlot( "parallel.tweakAlphaSrc" );
static const Slot tweakTauSlot =
	initShellCinfo()->getSlot( "parallel.tweakTauSrc" );
static const Slot setupGateSlot =
	initShellCinfo()->getSlot( "parallel.setupGateSrc" );
static const Slot file2tabSlot =
	initShellCinfo()->getSlot( "parallel.file2tabSrc" );
static const Slot rCreateSlot =
	initShellCinfo()->getSlot( "parallel.createSrc" );
static const Slot rCreateArraySlot =
	initShellCinfo()->getSlot( "parallel.createArraySrc" );
static const Slot rGetSlot = initShellCinfo()->getSlot( "parallel.getSrc" );
static const Slot rSetSlot = initShellCinfo()->getSlot( "parallel.setSrc" );
static const Slot rAddSlot = initShellCinfo()->getSlot( "parallel.addLocalSrc" );

static const Slot parReadCellSlot =
	initShellCinfo()->getSlot( "parallel.readcellSrc" );

static const Slot requestLeSlot =
	initShellCinfo()->getSlot( "parallel.requestLeSrc" );

static const Slot requestWildcardSlot =
	initShellCinfo()->getSlot( "parallel.requestParWildcardSrc" );

static const Slot parGetFieldSlot =
	initShellCinfo()->getSlot( "parallel.getSrc" );

static const Slot parAddFieldSlot =
	initShellCinfo()->getSlot( "parallel.addfieldSrc" );

static const Slot parSetFieldSlot =
	initShellCinfo()->getSlot( "parallel.setSrc" );

static const Slot parDeleteSlot =
	initShellCinfo()->getSlot( "parallel.deleteSrc" );

static const Slot parReschedSlot =
	initShellCinfo()->getSlot( "parallel.reschedSrc" );

static const Slot parReinitSlot =
	initShellCinfo()->getSlot( "parallel.reinitSrc" );

static const Slot parStopSlot =
	initShellCinfo()->getSlot( "parallel.stopSrc" );

static const Slot parStepSlot =
	initShellCinfo()->getSlot( "parallel.stepSrc" );

static const Slot parSetClockSlot =
	initShellCinfo()->getSlot( "parallel.setClockSrc" );

static const Slot parQuitSlot =
	initShellCinfo()->getSlot( "parallel.quitSrc" );

#endif

void printNodeInfo( const Conn* c );

//////////////////////////////////////////////////////////////////////
// Initializer
//////////////////////////////////////////////////////////////////////

Shell::Shell()
	: cwe_( Id() ), recentElement_( Id() )
{

#ifdef DO_UNIT_TESTS
        post_ = 0;
#endif // DO_UNIT_TESTS


	simDump_ = new SimDump;
	// At this point the initMPI should have initialized numNodes
	offNodeData_.resize( maxNumOffNodeRequests );
	freeRidStack_.resize( maxNumOffNodeRequests );
	for ( unsigned int i = 0; i < maxNumOffNodeRequests; i++ )
		freeRidStack_[i] = maxNumOffNodeRequests - i - 1;
}

//////////////////////////////////////////////////////////////////////
// General path to eid conversion utilities
//////////////////////////////////////////////////////////////////////

/**
 * This needs to be on the Shell for future use, because we may
 * need to look up remote nodes
 */
Id Shell::parent( Id eid )
{
	Eref eref (eid(), eid.index());
	Id ret;
	// Check if eid is on local node, otherwise go to remote node
	// ret = Neutral::getParent(e)
	if ( get< Id >( eref , "parent", ret ) )
		return ret;
	return Id::badId();
}

/**
 * Returns the id of the element at the end of the specified path.
 * On failure, returns badId()
 * It is a static func as a utility for parsers.
 * It takes a pre-separated vector of names.
 * It ignores names that are just . or /
 *
 * For off-node children, if we a) insist that there always be a proxy
 * and b) change proxies to also store names, we can do this easily.
 *
 */
Id Shell::traversePath( Id start, vector< string >& names )
{
	Id ret = localTraversePath( start, names );
	if ( ret.bad() ) {
		if ( numNodes_ > 1 &&
			( start == Id() || start == Id::shellId() ) ) {
			// Possibly element is rooted off-node
			return parallelTraversePath( start, names );
		}
	}
	return ret;
}

/**
 * This function is strictly local: it does not invoke any outgoing
 * calls and returns Id::bad if it cannot find the object.
 * Called by target nodes to avoid infinite recursion.
 */
Id Shell::localTraversePath( Id start, vector< string >& names )
{
	assert( !start.bad() );
	if ( !start.isGlobal() && start.node() != Shell::myNode() )
		return Id::badId();

	vector< string >::iterator i;
	for ( i = names.begin(); i != names.end(); i++ ) {
		if ( *i == "." || *i == "/" ) {
			continue;
		} else if ( *i == ".." ) {
			start = parent( start );
		} else {
			Id ret;
			//Neutral::getChildByName(e, *i);
			lookupGet< Id, string >( start.eref(), "lookupChild", ret, *i );
			//if ( ret.zero() || ret.bad() ) cout << "Shell:traversePath: The id is bad" << endl;
			if ( ret.zero() || ret.bad() ){
				return Id::badId();
			}
			start = ret;
		}
	}
	return start;
}


// Requires a path argument without a starting space
// Perhaps this should be in the interpreter?
// Need to deal with case where element is on another node.
// Normal situation is all on current node.
// Simple situation is if a child is offnode, but parents are here.
// 	Then we just traverse through the path till we go offnode, and keep
// 	follwing it there..
// Harder situation is when offnode child path begins at a global like
// /root or /shell. Then all nodes have to be asked about the path.
//
// In the latter 2 cases we send the query to the one node, or to all
// nodes, via the shell as usual. Then we set up a busy poll loop till 
// the answer comes back.
Id Shell::innerPath2eid( 
		const string& path, const string& separator, bool isLocal ) const
{
	if ( path == separator || path == "/root" )
			return Id();

	if ( path == "" || path == "." )
			return cwe_;

	if ( path == "^" )
			return recentElement_;

	if ( path == ".." ) {
			if ( cwe_.zero() )
				return cwe_;
			return parent( cwe_ );
	}

	vector< string > names;

	Id start; // Initializes to zero
	if ( path.substr( 0, separator.length() ) == separator ) {
		separateString( path.substr( separator.length() ), names, separator );
	} else if ( path.substr( 0, 5 ) == "/root" ) {
		separateString( path.substr( 5 ), names, separator );
	} else {
		start = cwe_;
		separateString( path, names, separator );
	}
	if ( isLocal )
		return localTraversePath( start, names );
	else
		return traversePath( start, names );
}

// This is the static version of the function.
Id Shell::path2eid( const string& path, const string& separator, 
	bool isLocal )
{
	/*
	Id shellId;
	bool ret = lookupGet< Id, string >(
				Element::root(), "lookupChild", shellId, "shell" );
	assert( ret );
	assert( !shellId.bad() );
	Shell* s = static_cast< Shell* >( shellId()->data() );
	*/
	Eref ShellE = Id::shellId().eref();
	assert( ShellE.e != 0 );
	Shell* s = static_cast< Shell* >( ShellE.data() );
	assert( s != 0 );
	return s->innerPath2eid( path, separator, isLocal );
}

string Shell::localEid2Path( Id eid ) 
{
	if ( eid.eref()->className() == "proxy" )
		return "proxy";
	if ( eid.zero() )
		return string( "/" );
	if ( !eid.good() )
		return string( "bad" );

	static const string slash = "/";
	string n( "" );
	while ( !eid.zero() ) {
		Id pid = parent( eid );
		if (pid()->elementType() == "Simple" && eid()->elementType() == "Array"){
			ostringstream s1;
			s1 << eid()->name() << "[" << eid.index() << "]";
			n = slash + s1.str() + n;
		}
		else {
			n = slash + eid()->name() + n;
		}
		eid = pid;
	}
	return n;
}

/**
 * Returns that component of path that precedes the last separator.
 * If there is nothing there, or no separator, returns an empty string.
 */
string Shell::head( const string& path, const string& separator )
{
	string::size_type pos = path.rfind( separator );
	if ( pos == string::npos )
			return "";

	return path.substr( 0, pos );
}

/**
 * Returns that component of path that follows the last separator.
 * If there is nothing there, or no separator, returns the entire path.
 */
string Shell::tail( const string& path, const string& separator )
{
	string::size_type pos = path.rfind( separator );
	if ( pos == string::npos )
			return path;

	return path.substr( pos + separator.length() );
}

//////////////////////////////////////////////////////////////////////
// Special low-level operations that Shell handles using raw
// serialized strings from PostMaster.
//////////////////////////////////////////////////////////////////////

void Shell::rawAddFunc( const Conn* c, string s )
{
	Element* post = c->source().e;
	assert( post->className() == "PostMaster" );
	unsigned int mynode;
	unsigned int remotenode;
	get< unsigned int >( post, "localNode", mynode );
	get< unsigned int >( post, "remoteNode", remotenode );
	// cout << ".";
	// cout << "Shell::rawAddFunc( " << s << " ), on " << mynode << ", from " << remotenode << "\n";
	vector< string > svec;
	separateString( s, svec, " " );
	unsigned int j = 0; // This is for breakpointing for parallel debug
	while ( j > 0 ) // for breakpointing in parallel debug.
		;
	// svec seq is : srcid, targetId, targetField, srcType
	Id destId = Id::str2Id( svec[1] );
	if ( destId.bad() ) {
		cout << "Error: Shell::rawAddFunc: msgdest is a bad elm on " << mynode << " from " << remotenode << " with str " << s << "\n";
		return;
	} 
	Element* dest = destId();
	if ( dest == 0 ) {
		cout << "Error: Shell::rawAddFunc: msgdest ptr for id " << destId << " is empty on " << mynode << " from " << remotenode << " with str " << s << "\n";
		return;
	} 
	if ( dest->className() == "PostMaster" ) { //oops, off node.
		cout << "Error: Shell::rawAddFunc: msgdest is off node on " << mynode << " from " << remotenode << " with str " << s << "\n";
		return;
	}
	const Finfo *destField = dest->findFinfo( svec[2] );
	if ( destField == 0 ) {
		cout << "Error: Shell::rawAddFunc: targetField does not exist on dest on " << mynode << " from " << remotenode << " with str " << s << "\n";
		return;
	}

	string typeSig = "";
	val2str< const Ftype* >( destField->ftype()->baseFtype(), typeSig );
	if ( typeSig != svec[3] ) {
		cout << "Error: Shell::rawAddFunc: field type mismatch: '" <<
			typeSig << "' != '" << svec[3] << "' on " << mynode << " from " << remotenode << "\n";
		return;
	}
	
	// post->findFinfo( "data" )->add( post, dest, destField );
	// cout << "Shell::rawAddFunc: Successfully added msg on remote node\n";
}

void Shell::rawCopyFunc( const Conn* c, string s )
{
	cout << "Shell::rawCopyFunc( " << s << " )\n";
}

void Shell::rawTestFunc( const Conn* c, string s )
{
	Element* post = c->source().e;
	ASSERT( post->className() == "PostMaster", "rawTestFunc" );
	unsigned int mynode;
	unsigned int remotenode;
	get< unsigned int >( post, "localNode", mynode );
	get< unsigned int >( post, "remoteNode", remotenode );
	char teststr[30];
	sprintf( teststr, "My name is Michael Caine %d,%d", 
		remotenode, mynode );

	// cout << "Shell::rawTestFunc( " << s << " )," << teststr << "\n";
	
	ASSERT( s == teststr, "Shell::rawTestFunc" );
	// cout << "Shell::rawTestFunc( " << s << " )\n";
}

void Shell::pollFunc( const Conn* c )
{
	while( 1 ) {
		// cout << "." << flush;
		send1< int >( c->target(), pollSlot, 1 );
		// Surprisingly, the usleep seems to worsen the responsiveness.
		// usleep( 10 );
	}
}

bool Shell::isSerial( )
{
	return ( Shell::numNodes( ) == 1 );
}

int Shell::getMyNode( Eref e )
{
	return Shell::myNode();
}

unsigned int Shell::myNode()
{
	return myNode_;
}

int Shell::getNumNodes( Eref e )
{
	return Shell::numNodes();
}

unsigned int Shell::numNodes()
{
	return numNodes_;
}

void Shell::setNodes( unsigned int myNode, unsigned int numNodes )
{
	myNode_ = myNode;
	numNodes_ = numNodes;
}

//////////////////////////////////////////////////////////////////////
// Moose fields for Shell
//////////////////////////////////////////////////////////////////////

void Shell::setCwe( const Conn* c, Id id )
{
	// This should only be called on master node.
	if ( !id.bad() ) {
		Shell* s = static_cast< Shell* >( c->data() );
		s->cwe_ = id;
	} else {
		cout << "Error: Attempt to change to nonexistent element.\n";
	}
}

Id Shell::getCwe( Eref e )
{
	assert( e.e != 0 );
	const Shell* s = static_cast< const Shell* >( e.data() );
	return s->cwe_;
}

void Shell::trigCwe( const Conn* c )
						
{
	Shell* s = static_cast< Shell* >( c->data() );
	sendBack1< Id >( c, cweSlot, s->cwe_ );
	// sendTo1< Id >( c->target(), cweSlot, c->targetIndex(), s->cwe_);
}

void Shell::pushe( const Conn* c, Id id )
{
	Shell* s = static_cast< Shell* >( c->data() );
	if ( !id.bad() ) {
		s->workingElementStack_.push_back( s->cwe_ );
		s->cwe_ = id;
	} else {
		cout << "Error: Attempt to pushe to nonexistent element.\n";
	}
	sendBack1< Id >( c, cweSlot, s->cwe_ );
	// sendTo1< Id >( c->targetElement(), 0, cweSlot, c->targetIndex(), s->cwe_);
}

void Shell::pope( const Conn* c )
{
	Shell* s = static_cast< Shell* >( c->data() );
	if ( s->workingElementStack_.size() > 0 ) {
		s->cwe_ = s->workingElementStack_.back();
		if ( s->cwe_.bad() ) { 
			// In case we went back to an element that got deleted in
			// the interim.
			s->cwe_ = Id();
		}
		s->workingElementStack_.pop_back();
	} else {
		cout << "Error: empty element stack.\n";
	}
	sendBack1< Id >( c, cweSlot, s->cwe_ );
	// sendTo1< Id >( c->targetElement(), 0, cweSlot, c->targetIndex(), s->cwe_ );
}


//////////////////////////////////////////////////////////////////////
// Create and destroy are possibly soon to be deleted. These may have
// to go over to the Neutral, but till we've sorted out the SWIG
// interface we'll keep it in case it gets used there.
//////////////////////////////////////////////////////////////////////

// This whole function should be split into parallel and serial versions.
void Shell::trigLe( const Conn* c, Id parent )
{
	// In principle we should use a messaging approach to doing the 'get'
	// command, and it would work independent of node. Anyway, for now
	// we'll manage it explicitly.
	Shell* sh = static_cast< Shell* >( c->data() );
	assert( sh );
	if ( parent.bad() ) {
		cout << "Error: Shell::trigLe: unknown parent Id\n";
		return;
	}

	vector< Id > ret;
	vector< Nid > nret;
	// This will get messier for arrayElements across nodes.
	// Need to do a smarter test for global objects, something like:
	// if ( parent.node() == Id::GlobalNode )
	if ( parent == Id() || parent == Id::shellId() ) {
		bool flag = get< vector< Id > >( 
			parent.eref(), "childList", ret );
		assert( flag );
		
#ifdef USE_MPI
		if ( ! Shell::isSerial( ) )
			getOffNodeValue< vector< Nid >, Nid >( c->target(), 
				requestLeSlot, sh->numNodes(),
				&nret, parent );
#endif
	} else if ( parent.node() == 0 || parent.isGlobal() ) {
		bool flag = get< vector< Id > >( parent.eref(), "childList", ret );
		assert( flag );
	} else { // Off-node to single node.
#ifdef USE_MPI
		getOffNodeValue< vector< Nid >, Nid >( c->target(), 
			requestLeSlot, parent.node(),
			&nret, parent );
#endif
	}
	ret.insert( ret.end(), nret.begin(), nret.end() );
	sendBack1< vector< Id > >( c, elistSlot, ret );
}

/**
 * Creates an object on the specified node, on the specified parent object.
 * This function is only called on node 0.
 * If node < 0 then the new object is created as per system algorithm,
 * which normally just places the child on the same node as the parent.
 * Whenever the parent isGlobal, the child is created on all nodes.
	// Illegal cases: 
	// 		( pa != Id() && !pa.isGlobal() ) && id.isGlobal, 
	// 		pa.isGlobal && !id.isGlobal
	// Local cases:
	// 		pa.node == 0, id.node == 0
	// Remote cases:
	// 		pa.node != 0
	// Global cases:
	// 		pa.isGlobal, id.isGlobal
 */
void Shell::staticCreate( const Conn* c, string type,
					string name, unsigned int node, Id parent )
{
	Shell* s = static_cast< Shell* >( c->data() );

	assert( s->myNode_ == 0 );
	Id id;
	Nid paNid( parent );

	if ( node == Id::UnknownNode ) { // Leave it to the system.
		// This is where the IdManager does clever load balancing etc
		// to assign child node.
		id = Id::childId( paNid );
	} else if ( node == Id::GlobalNode ) {
		id = Id::newId();
		id.setGlobal();
	} else if ( node >= s->numNodes_ ) {
		cout << "Warning: Shell::staticCreate: unallocated target node " <<
			node << ", using node 0\n";
		node = 0;
		id = Id::childId( paNid );
	} else {
		id = Id::makeIdOnNode( node );
	}
	if ( id.isGlobal() && !( parent == Id() || paNid.isGlobal() ) ) {
		cout << "Error: Cannot create global object unless parent is global\n";
		return;
	}
	//~ if ( ( !( parent == Id() ) && paNid.isGlobal( )) && !id.isGlobal() ) {
		//~ cout << "Error: Cannot create local object on global parent\n";
		//~ return;
	//~ }
	
	Element* ret;
	if ( ( paNid.isGlobal() || paNid.node() == 0 || parent == Id() ) &&
		( id.isGlobal() || id.node() == 0 ) ) { // Make it here.
		if ( id.isGlobal() )
			ret = createGlobal( type, name, parent, id );
		else
			ret = s->create( type, name, parent, id );

		if ( ret ) { // Tell the parser it was created happily.
#ifdef DO_UNIT_TESTS
			// Nasty issue of callback to a SetConn here.
			if ( dynamic_cast< const SetConn* >( c ) == 0 )
				sendBack1< Id >( c, createSlot, id );
#else
			sendBack1< Id >( c, createSlot, id );
#endif
		} else {
			cout << "Error: Shell::staticCreate: unable to create '" <<
				name << "' on parent " << parent.path() << endl;
		}
#ifdef USE_MPI
	} else { // Has to go off-node, to a single target node.
		unsigned int targetNode;
		if ( parent.isGlobal() ) {
			targetNode = id.node();
			assert( targetNode != 0 );
		} else {
			targetNode = paNid.node();
			if ( targetNode == 0 ) {
				cerr << "Error: Shell::staticCreate: Unable to create '"
					<< name << "' on node " << id.node()
					<< ". Parent is on node 0.\n";
				return;
			}
		}
		assert( targetNode < numNodes() );

		sendTo4< string, string, Nid, Nid >( 
			c->target(), rCreateSlot, targetNode - 1,
			type, name, 
			paNid, id );
#endif
	}
}

/**
 * Like Neutral::create, except for creating globals.
 * Should be called on master node. Requests worker nodes to create the object.
 * Not necessary for the caller to call id.setGlobal(), since this function
 * does that.
 */
Element* Shell::createGlobal(
	const string& type, const string& name, Id parent, Id id )
{
	assert( myNode() == 0 );
	assert( parent == Id() || parent.isGlobal() );
	
	id.setGlobal();
	Eref shellE = Id::shellId().eref();
	Shell* sh = static_cast< Shell* >( shellE.data() );
	Element* ret = sh->create( type, name, parent, id );
#ifdef USE_MPI
	if ( ret )
		send4< string, string, Nid, Nid >( 
			shellE, rCreateSlot,
			type, name, parent, id );
#endif
	
	return ret;
}

// Static function
// parameter has following clumped in the order mentioned, Nx, Ny, dx, dy, xorigin, yorigin
// Unlike the regular staticCreate function, here we have no option of
// specifying target node. Just sit on whatever node the parent is on.
void Shell::staticCreateArray( const Conn* c, string type,
					string name, Id parent, vector <double> parameter )
{
	Shell* s = static_cast< Shell* >( c->data() );
	assert( s->myNode_ == 0 );
	Nid paNid( parent );

	// This is where the IdManager does clever load balancing etc
	// to assign child node. If parent is not node 0, put child on parent.
	Id id = Id::childId( parent );
	// Id id = Id::scratchId();
	if ( id.node() == 0 || id.isGlobal() ) { // local node
		int n = (int) (parameter[0]*parameter[1]);
		Element* ret = s->createArray( type, name, parent, id, n );
		assert(parameter.size() == 6);
		Element* child = id();
		ArrayElement* f = static_cast <ArrayElement *> (child);
		f->setNoOfElements((int)(parameter[0]), (int)(parameter[1]));
		f->setDistances(parameter[2], parameter[3]);
		f->setOrigin(parameter[4], parameter[5]);
#ifdef USE_MPI
		if ( ret ) { // Tell the parser it was created happily.
			if ( id.isGlobal() ) { // Also make child on all remote nodes.
				pair< Nid, Nid > temp( paNid, id );
				send4< string, string, pair< Nid, Nid >, vector< double > >( 
				c->target(), rCreateArraySlot,
				type, name, temp, parameter );
			}
			//GenesisParserWrapper::recvCreate(conn, id)
			sendBack1< Id >( c, createSlot, id);
		}
	} else { // Make child on single remote node.
		assert( id.node() > 0 );
		// OffNodeInfo* oni = static_cast< OffNodeInfo* >( child->data() );
		// Element* post = oni->post;
		unsigned int target = id.node() - 1;
		//e->connSrcBegin( rCreateSlot.msg() ) - e->lookupConn( 0 ) +
		//	id.node() - 1;
		pair< Nid, Nid > temp( paNid, id );
		sendTo4< string , string, pair< Nid, Nid >, vector< double > >( 
			c->target(), rCreateArraySlot, target,
			type, name, temp, parameter );
		// delete oni;
		// delete child;
#else
		if ( ret ) { // Tell the parser it was created happily.
			sendBack1< Id >( c, createSlot, id);
		}
#endif
	}
}

void Shell::planarconnect( const Conn* c, 
	string source, string dest, double probability)
{
	vector<Id> src_list; 
	vector<Id > dst_list;

	innerGetWildcardList( c, source, 1, src_list );
	innerGetWildcardList( c, dest, 1, dst_list );
	// simpleWildcardFind( source, src_list );
	// simpleWildcardFind( dest, dst_list );
	for (size_t i = 0; i < src_list.size(); i++) {
		/*
		if (src_list[i]()->className() != "SpikeGen" && src_list[i]()->className() != "RandomSpike"){
			cout << "The source element must be SpikeGen or RandomSpike" << endl;
			return;
		}
		*/
		for(size_t j = 0; j < dst_list.size(); j++) {
			//cout << src_list[i]->id().path() << " " << dst_list[i]->id().path() << endl;
			/*
			if (dst_list[j]()->className() != "SynChan"){
				cout <<  "The dest element must be SynChan" << endl;
				return;
			}
			*/
			// RD: random number has to be changed. 
			if (mtrand() <= probability){
// 				cout << i+1 << " " << j+1 << endl;

				// This is a parallelized call
				addSingleMessage( c, src_list[i], "event", 
					dst_list[j], "synapse" );
				// src_list[i].eref().add( "event", dst_list[j].eref(), "synapse" );
				// src_list[i]()->findFinfo("event")->add(src_list[i](), dst_list[j](), dst_list[j]()->findFinfo("synapse"));
			}
		}
	}
}

// void Shell::planardelay(const Conn* c, string source, double delay)
// {
// 	static const Cinfo* sgCinfo = Cinfo::find( "SpikeGen" );
// 	// static const Finfo* eventFinfo = sgCinfo->findFinfo( "event" );
// 	static const Slot eventSlot = sgCinfo->getSlot( "event" );
// 
// 	vector <Element* > srcList;
// 	simpleWildcardFind( source, srcList );
// 	for ( size_t i = 0 ; i < srcList.size(); i++){
// 		if ( srcList[ i ]->className() != "SpikeGen"){
// 			cout << "Shell::planardelay: Error: Source is not SpikeGen" << endl; 	
// 			return;
// 		}
// 
// 		vector< ConnTainer* >::const_iterator j;
// 		const Msg* m = srcList[ i ]->msg( eventSlot.msg() );
// 
// 		for( j = m->begin(); j != m->end(); j++ ) {
// 			// Need to sort out the iteration through all targets, here.
// 			// Many targets will be ArrayElements and should have
// 			// special fast assignment for all indices.
// 			// for ( Conn* k = ( *j )->conn( eIndex, m->isDest() ); j->good(); j++ )
// 				// setDelay( k );
// 		}
// 		
// 		/*
// 		vector <Conn> conn;
// 		srcList[i]->findFinfo("event")->outgoingConns(srcList[i], conn);
// 		for (size_t j = 0; j < conn.size(); j++){
// 			unsigned int numSynapses;
// 			Element *dest = conn[j].targetElement();
// 			if (destination != ""){
// 				bool found = false;
// 				for (vector<Element *>::iterator iter = dst_list.begin(); 
// 					iter != dst_list.end(); iter++)
// 					if (*iter == dest) {found = true; break;}
// 				if (!found) continue;
// 			}
// 			bool ret = get< unsigned int >( dest, "numSynapses", numSynapses );
// 			if (!ret) {cout << "Shell::planardelay: Error2" << endl; return;}
// 			for (size_t k = 0 ; k < numSynapses; k++){
// 				double number = 0;
// 				if (delaychoice){
// 					cout << "planardelay:: radial not implemented."<< endl;
// 					// Not decided what to do
// 				}
// 				else {
// 					number = delay;
// 				}
// 				if (randchoice){
// 					double random = p->getNextSample();
// 					while (random > maxallowed ) random = p->getNextSample();
// 					if (absoluterandom)
// 						{number += random;}
// 					else 
// 						{number += number*random;}
// 				}
// 				if (add){
// 					double delay_old = 0;
// 					ret = lookupGet< double, unsigned int >( dest, "delay", delay_old, k );
// 					if (!ret) {
// 						cout << "planardelay:: Error3" << endl;
// 					}
// 					number += delay_old;
// 				}
// 				lookupSet< double, unsigned int >( dest, "delay", number, k );
// 			}
// 		}
// 		*/
// 	}
// }


void Shell::planardelay(
		const Conn* c,
		string source,
		string destination,
		vector< double > parameter)
{
	assert (parameter.size() == 11);
	double delay = parameter[0];
// 	double conduction_velocity = parameter[1];
	bool add = parameter[2];
	double scale = parameter[3];
	double stdev = parameter[4];
	double maxdev = parameter[5];
	double mid = parameter[6];
	double max = parameter[7];
	bool absoluterandom = parameter[8];
	int delaychoice = int(parameter[9]);
	int randchoice = int(parameter[10]);
	double maxallowed;
	Probability *p;
	switch (randchoice){
		case 0:
			break;
		case 1:
			p = new Uniform(-scale, scale);
			maxallowed = scale;
			break;
		case 2: 
			p = new Normal(0, stdev);
			maxallowed = maxdev;
			break;
		case 3: 
			p = new Exponential(log(2.0)/mid);
			maxallowed = max;
			break;
	}
	
	static const Cinfo* sgCinfo = Cinfo::find( "SpikeGen" );
	static const Slot sgeventSlot = sgCinfo->getSlot( "event" );
	static const Cinfo* rsCinfo = Cinfo::find( "RandomSpike" );
	static const Slot rseventSlot = rsCinfo->getSlot( "event" );
	vector <Id> srcList;
	vector <Id> dst_list;
	simpleWildcardFind( source, srcList );
	if (destination != "")
		simpleWildcardFind( destination, dst_list );
	for (size_t i = 0 ; i < srcList.size(); i++){
		if (srcList[i]()->className() != "SpikeGen" && srcList[i]()->className() != "RandomSpike"){cout << "Shell::planardelay: Source = " << srcList[i]()->className() << " is not SpikeGen" << endl; return;}
		vector< ConnTainer* >::const_iterator j;
		
		const Msg* m;
		if (srcList[i]()->className() == "SpikeGen"){ 
			m= srcList[ i ]()->msg( sgeventSlot.msg() );
		}
		else if (srcList[i]()->className() == "RandomSpike"){ 
			m= srcList[ i ]()->msg( rseventSlot.msg() );
		}
		//srcList[i]->findFinfo("event")->outgoingConns(srcList[i], conn);
		for( j = m->begin(); j != m->end(); j++ ) {
			unsigned int numSynapses;
			//Element *dest = (*j)->e2();
			Eref eref;
			Conn* k = ( *j )->conn( srcList[i].eref(), 0 );
			size_t eIndex = 0;
// 			for ( Conn* k = ( *j )->conn( 0, m->isDest() ); k->good(); k++ ){
			while (k->good()){
				eref = k->target();
// 				eref = ( *j )->conn( /*eIndex*/0, m->isDest())->target();
				if (destination != ""){
					bool found = false;
					for (vector<Id>::iterator iter = dst_list.begin(); 
						iter != dst_list.end(); iter++)
						if (*iter == eref.id()) {found = true; break;}
					if (!found) continue;
				}
				bool ret = get< unsigned int >( eref, "numSynapses", numSynapses );
				if (!ret) {cout << "Shell::planardelay: Could not access number of synapses." << endl; return;}
				for (size_t l = 0 ; l < numSynapses; l++){
					double number = 0;
					if (delaychoice){
						cout << "planardelay:: radial not implemented."<< endl;
						// Not decided what to do
					}
					else {
						number = delay;
					}
					if (randchoice){
						double random = p->getNextSample();
						while (random > maxallowed ) random = p->getNextSample();
						if (absoluterandom)
							{number += random;}
						else 
							{number += number*random;}
					}
					if (add){
						double delay_old = 0;
						ret = lookupGet< double, unsigned int >( eref, "delay", delay_old, l );
						if (!ret) {
							cout << "planardelay:: Error3" << endl;
						}
						number += delay_old;
					}
					lookupSet< double, unsigned int >( eref, "delay", number, l );
				}
				eIndex++;
				Eref e( srcList[i].eref().e, eIndex );
				k = ( *j )->conn( e, 0 );
			}
		}
	}
}

void Shell::planarweight(
		const Conn* c,
		string source,
		string destination,
		vector< double > parameter)
{
	assert (parameter.size() == 12);
	double weight = parameter[0];
// 	double decay_rate = parameter[1];
// 	double max_weight = parameter[2];
// 	double min_weight = parameter[3];
	double scale = parameter[4];
	double stdev = parameter[5];
	double maxdev = parameter[6];
	double mid = parameter[7];
	double max = parameter[8];
	bool absoluterandom = parameter[9];
	int weightchoice = int(parameter[10]);
	int randchoice = int(parameter[11]);
	double maxallowed;
	Probability *p;
	switch (randchoice){
		case 0:
			break;
		case 1:
			p = new Uniform(-scale, scale);
			maxallowed = scale;
			break;
		case 2: 
			p = new Normal(0, stdev);
			maxallowed = maxdev;
			break;
		case 3: 
			p = new Exponential(log(2.0)/mid);
			maxallowed = max;
			break;
	}
	
	static const Cinfo* sgCinfo = Cinfo::find( "SpikeGen" );
	static const Slot sgeventSlot = sgCinfo->getSlot( "event" );
	static const Cinfo* rsCinfo = Cinfo::find( "RandomSpike" );
	static const Slot rseventSlot = rsCinfo->getSlot( "event" );
	vector <Id> srcList;
	vector <Id> dst_list;
	simpleWildcardFind( source, srcList );
	if (destination != "")
		simpleWildcardFind( destination, dst_list );
	for (size_t i = 0 ; i < srcList.size(); i++){
		if (srcList[i]()->className() != "SpikeGen" && srcList[i]()->className() != "RandomSpike"){cout << "Shell::planarweight: Source is not SpikeGen or RandomSpike" << endl; return;}
		vector< ConnTainer* >::const_iterator j;
		const Msg* m;
		if (srcList[i]()->className() == "SpikeGen"){ 
			m= srcList[ i ]()->msg( sgeventSlot.msg() );
		}
		else if (srcList[i]()->className() == "RandomSpike"){ 
			m= srcList[ i ]()->msg( rseventSlot.msg() );
		}
		//srcList[i]->findFinfo("event")->outgoingConns(srcList[i], conn);
		for( j = m->begin(); j != m->end(); j++ ) {
			unsigned int numSynapses;
			//Element *dest = (*j)->e2();
			Eref eref;
			Conn* k = ( *j )->conn( srcList[i].eref(), 0);
			size_t eIndex = 0;
// 			for ( Conn* k = ( *j )->conn( /*eIndex*/0, m->isDest() ); k->good(); k++ ){
			while (k->good()){
				eref = k->target();
				if (destination != ""){
					bool found = false;
					for (vector<Id>::iterator iter = dst_list.begin(); 
						iter != dst_list.end(); iter++)
						if (*iter == eref.id()) {found = true; break;}
					if (!found) continue;
				}
				bool ret = get< unsigned int >( eref, "numSynapses", numSynapses );
				if (!ret) {cout << "Shell::planarweight: Could not access number of synapses." << endl; return;}
				for (size_t l = 0 ; l < numSynapses; l++){
					double number = 0;
					if (weightchoice){
						cout << "planarweight:: decay not implemented."<< endl;
					}
					else {
						number = weight;
					}
					if (randchoice){
						double random = p->getNextSample();
						while (random > maxallowed ) random = p->getNextSample();
						if (absoluterandom)
							{number += random;}
						else 
							{number += number*random;}
					}
					lookupSet< double, unsigned int >( eref, "weight", number, l );
				}
				Eref e( srcList[i].eref().e, ++eIndex );
				k = ( *j )->conn( e, 0 );
			}
		}
	}
}




// does not do - destination is a SynChan test
void Shell::getSynCount2(const Conn* c, Id dest){
	Element* dst = dest();
	unsigned int numSynapses;
	bool b = get< unsigned int >( dst, "numSynapses", numSynapses );
	if (!b) {
		cout << "Shell:: syncount failed at" << dst->name() <<endl; 
		return;
	}
	char e[10];
	sprintf (e, "%d", numSynapses);
	string ret = e;
	sendBack1< string >( c, getFieldSlot, ret );
	/*
	sendTo1< string >( c->targetElement(),
				getFieldSlot, c->targetIndex(), 0, ret );
	*/
}




// Static function
void Shell::staticDestroy( const Conn* c, Id victim )
{
	// Should be more general:
	// if ( victim.node() == Id::globalNode ) { error() };
	
	if ( victim == Id() ) {
		cout << "Error: Shell::staticDestroy: cannot delete /root\n";
		return;
	}
	if ( victim == Id::shellId() ) {
		cout << "Error: Shell::staticDestroy: cannot delete /shell\n";
		return;
	}

	Shell* s = static_cast< Shell* >( c->data() );
	if ( victim.node() == s->myNode() ) {
		s->destroy( victim );
#ifdef USE_MPI
	} else { // Ask another node to do the dirty work.
		unsigned int tgt = victim.node();
		if ( tgt > s->myNode() )
			tgt--;
		sendTo1< Id >( Id::shellId().eref(), parDeleteSlot, tgt, victim );
#endif
	}
}

/**
 * This function invokes localAddField on the target nodes.
 */
void Shell::addField( const Conn* c, Id id, string fieldname )
{
	assert( id.good() );
#ifdef USE_MPI
	if ( id.isGlobal() ) { // do the addfield on all nodes
		send2< Id, string >( c->target(), parAddFieldSlot,
			id, fieldname );
		localAddField( c, id, fieldname );
	} else if ( id.node() == Shell::myNode() ) { // do a local addfield
		localAddField( c, id, fieldname );
	} else {	// do a remote addfield on selected node
		unsigned int tgt = ( id.node() < myNode() ) ? 
			id.node() : id.node() - 1;
		sendTo2< Id, string >( c->target(), parAddFieldSlot, tgt,
			id, fieldname );
	}
#else
	localAddField( c, id, fieldname );
#endif
}

/**
 * This function adds an ExtFieldFinfo locally. Invoked by the parAddFieldSlot.
 */
void Shell::localAddField( const Conn* c, Id id, string fieldname )
{
	assert( id.good() );
	Element* e = id();
	assert( e != 0);
	Finfo* f = new ExtFieldFinfo(fieldname, Ftype1<string>::global());
	e->addExtFinfo( f );
}

// Static function
/**
 * This function handles request to get a field value. It triggers
 * a return function to the calling object, as a string.
 */
void Shell::getField( const Conn* c, Id id, string field )
{
	if ( id.bad() )
		return;

	string ret;
	Eref eref = id.eref();

	unsigned int node = id.node();
	assert( node < numNodes() || node == Id::GlobalNode );
	if ( node == myNode() || node == Id::GlobalNode ) {
		const Finfo* f = eref.e->findFinfo( field );
		if ( f ) {
			if ( f->strGet( eref, ret ) ){
				sendBack1< string >( c, getFieldSlot, ret );
			}
		} else {
			cout << "Shell::getField: Failed to find field " << field << 
				" on object " << id.path() << endl;
		}
#ifdef USE_MPI
	} else {
		Shell* sh = static_cast< Shell* >( c->data() );
		unsigned int requestId = 
			openOffNodeValueRequest< string >( sh, &ret, 1 );
		unsigned int tgt = ( node < myNode() ) ? node : node - 1;
		sendTo3< Id, string, unsigned int >(
			c->target(), parGetFieldSlot, tgt,
			id, field, requestId
		);
		string* temp = closeOffNodeValueRequest< string >( sh, requestId );
		assert( &ret == temp );
		sendBack1< string >( c, getFieldSlot, ret );
#endif
	}
}

#ifndef USE_MPI
// Static function
/**
 * This copies the element tree from src to parent. If name arg is 
 * not empty, it renames the resultant object. It first verifies
 * that the planned new object has a different name from any of
 * the existing children of the prospective parent.
 */
void Shell::copy(
	const Conn* c, 
	Id src,
	Id parent,
	string name )
{
	// Shell* s = static_cast< Shell* >( c.targetElement()->data() );
	Element* e = src()->copy( parent(), name );
	if ( e ) { // Send back the id of the new element base
		sendBack1< Id >( c, createSlot, e->id() );
	}
}

// Static function. Dummy for the single-node version.
void Shell::parCopy(
	const Conn* c,
	Nid src,
	Nid parent,
	string name,
	IdGenerator idGen )
{
	;
}

/**
 * This function copies the prototype element in form of an array.
 * It is similar to copy() only that it creates an array of copied 
 * elements. Used in createmap.
*/
void Shell::copyIntoArray(
	const Conn* c, 
	Id src,
	Id parent,
	string name,
	vector< double > parameter )
{
	Element* e = localCopyIntoArray( c, src, parent, name, parameter );
	if ( e != 0 )
		sendBack1< Id >( c, createSlot, e->id() );
}
#endif // ndef USE_MPI

Element* Shell::localCopyIntoArray(
	const Conn* c, 
	Id src,
	Id parent,
	string name,
	vector< double > parameter,
	IdGenerator idGen )
{
	int n = (int) (parameter[0]*parameter[1]);
	Element* e = src()->copyIntoArray( parent, name, n, idGen );
	//assign the other parameters to the arrayelement
	assert(parameter.size() == 6);
	ArrayElement* f = static_cast <ArrayElement *> (e);
	f->setNoOfElements((int)(parameter[0]), (int)(parameter[1]));
	f->setDistances(parameter[2], parameter[3]);
	f->setOrigin(parameter[4], parameter[5]);
	return e;
}

/**
 * This moves the element tree from src to parent. If name arg is 
 * not empty, it renames the resultant object. It first verifies
 * that the planned new object has a different name from any of
 * the existing children of the prospective parent.
 * Unlike the 'copy', this function is handled by the shell and may
 * involve interesting node relocation issues.
 */
void Shell::move( const Conn* c, Id src, Id parent, string name )
{
	assert( !src.bad() );
	assert( !parent.bad() );
	// Cannot move object onto its own descendant
	Element* e = src();
	Element* pa = parent();
	if ( pa->isDescendant( e ) ) {
		cout << "Error: move '" << e->name() << "' to '" << 
				pa->name() << 
				"': cannot move object onto itself or descendant\n";
		return;
	}
	Id srcPaId = Neutral::getParent( e );
	assert ( !srcPaId.bad() );
	if ( srcPaId == parent ) { // Just rename it.
		assert ( name != "" ); // New name must exist.
		if ( Neutral::getChildByName( pa, name ).bad() ) {
			// Good, we do not have name duplication.
			e->setName( name );
			// OK();
			return;
		} else {
			// Bad, we do have name duplication. This should not happen
			// because this ought to mean that we are moving the 
			// object as a child of the named object. 
			assert( 0 );
		}
	} else { // Move the object onto a new parent.
		string temp = name;
		if ( name == "" )
			temp = e->name();
		if ( Neutral::getChildByName( pa, temp ).bad() ) {
			// Good, we do not have name duplication.
			if ( name != "" )
				e->setName( name );
			/// \todo: Here we don't take into acount multiple parents.
			// Here we drop all parents.
			Eref( e ).dropAll( "child" );

			bool ret = Eref( pa ).add( "childSrc", e, "child" );
			assert ( ret );
			// OK();
			return;
		} else {
			// Bad, we do have name duplication. GENESIS
			// allows this but we will not.
			cout << "Error: move '" << e->name() << "' to '" << 
				pa->name() << "': same name child already exists.\n";
			return;
		}
	}
}

// Static function
/**
 * This function handles request to set a field value.
 * The reason why we take this function to the Shell is because
 * we will eventually need to be able to handle this for off-node
 * object requests.
 */
void Shell::setField( const Conn* c, Id id, string field, string value )
{
	assert( id.good() );
#ifdef USE_MPI
	if ( id.isGlobal() ) { // do the set on all nodes
		send3< Id, string, string >( c->target(), parSetFieldSlot,
			id, field, value );
		localSetField( c, id, field, value );
	} else if ( id.node() == Shell::myNode() ) { // do a local set
		localSetField( c, id, field, value );
	} else {	// do a remote set on selected node
		unsigned int tgt = ( id.node() < myNode() ) ? 
			id.node() : id.node() - 1;
		sendTo3< Id, string, string >( c->target(), parSetFieldSlot, tgt,
			id, field, value );
	}
#else
	localSetField( c, id, field, value );
#endif
}

/**
 * This is the local function to handle the setfield command
 * It is invoked by the parSetFieldSlot.
 */
void Shell::localSetField( const Conn* c, 
	Id id, string field, string value )
{
	assert( id.node() == myNode() || id.isGlobal() );
	Element* e = id();
	if ( !e ) {
		cout << "Shell::setField:Error: Element not found: " 
			<< id << endl;
		return;
	}

	const Finfo* f = e->findFinfo( field );
	if ( f ) {
		if ( !f->strSet( id.eref(), value ) )
			cout << "localSetField@" << myNode() << 
				": Error: cannot set field " << e->name() <<
				"." << field << " to " << value << endl;
	} else {
		cout << "localSetField@" << myNode() << 
			": Error: cannot find field: " << id.path() << ", " 
			<< e->name() << "." << field << endl;
	}
}

/**
 * This function handles request to set identical field value for a 
 * vector of objects. Used for the GENESIS SET function.
 * This could be optimized by organizing separate vectors, one for each
 * target node. For now just do a simple setField for each id.
 *
 * static function.
 */
void Shell::setVecField( const Conn* c, 
				vector< Id > elist, string field, string value )
{
	vector< Id >::iterator i;
	for ( i = elist.begin(); i != elist.end(); i++ ) {
		setField( c, *i, field, value );
/*
		// Cannot use i->good() here because we might set fields on /root.
		assert( !i->bad() ); 
		//Element* e = ( *i )();
		Eref eref = i->eref(); 
		// Appropriate off-node stuff here.
		const Finfo* f = eref.e->findFinfo( field );
		if ( f ) {
			if ( !f->strSet( eref, value ) )
				cout << "setVecField: Error: cannot set field " << i->path() <<
						"." << field << " to " << value << endl;
		} else {
			cout << "setVecField: Error: cannot find field: " << i->path() <<
				"." << field << endl;
		}
*/
	}
}

/**
 * This function handles request to load a file into an Interpol object
 */
void Shell::file2tab(
	const Conn* c, 
	Id id,
	string filename,
	unsigned int skiplines )
{
	assert( id.good() );
	
	if ( id.node() == 0 ) {
		localFile2tab( c, id, filename, skiplines );
	} else if ( id.isGlobal() ) {
		localFile2tab( c, id, filename, skiplines );
#ifdef USE_MPI
		send3< Nid, string, unsigned int >(
			c->target(), file2tabSlot,
			id, filename, skiplines );
	} else {
		unsigned int tgtNode = id.node() - 1;
		sendTo3< Nid, string, unsigned int >(
			c->target(), file2tabSlot, tgtNode,
			id, filename, skiplines );
#endif // USE_MPI
	}
}

void Shell::localFile2tab(
	const Conn* c, 
	Nid nid,
	string filename,
	unsigned int skiplines )
{
	Id id( nid );
	assert( id.good() );
	Element* e = id();
	if ( !e ) {
		cerr <<
			"Error: Shell::file2tab: Element not found: " << id << endl;
		return;
	}
	
	if ( !::set< string, unsigned int >( e, "load", filename, skiplines ) ) {
		cerr <<
			"Error: Shell::file2tab: cannot set field " << e->name() << ".load\n";
	}
}

void Shell::readSbml( const Conn* c, string filename, string location, int childnode )
{
#ifdef USE_SBML
	SbmlReader sr;
	
	Id loc( location );
	if ( loc.bad() ) {
		string::size_type pos = location.find_last_of( "/" );
		Id pa;
		string name;
		if ( pos == string::npos ) {
			pa = getCwe( c->target() );
			name = location;
		} else if ( pos == 0 ) {
			pa = Id();
			name = location.substr( 1 );
		} else {
			pa = Id( location.substr( 0, pos ), "/" );
			if ( pa.bad() ) {
				cout << "Error: readSbml: model path '" << location << "' not found.\n";
				return;
			}
			name = location.substr( pos + 1 );
		}
		
		Element* locE = Neutral::create( "KineticManager", name, pa, Id::scratchId() );
		loc = locE->id();
	}
	
	sr.read( filename, loc );
#else
	cerr << "Error: readSbml: This MOOSE is not built with SBML compatibility.\n";
#endif
}

void Shell::writeSbml( const Conn* c, string filename, string location, int childnode )
{
#ifdef USE_SBML 
	SbmlWriter sw;
	
	Id loc( location );
	if ( loc.bad() ) {
		cerr << "Error: Shell::writeSbml: Path " << location << " does not exist.\n";
		return;
	}
	
	sw.write( filename, loc );
#else
	cerr << "Error: writeSbml: This MOOSE is not built with SBML compatibility.\n";
#endif
}
void Shell::readNeuroml( const Conn* c, string filename, string location, int childnode )
{
#ifdef USE_NEUROML
	NeuromlReader nr;
	
	Id loc( location );
	if ( loc.bad() ) {
		string::size_type pos = location.find_last_of( "/" );
		Id pa;
		string name;
		if ( pos == string::npos ) {
			pa = getCwe( c->target() );
			name = location;
		} else if ( pos == 0 ) {
			pa = Id();
			name = location.substr( 1 );
		} else {
			pa = Id( location.substr( 0, pos ), "/" );
			if ( pa.bad() ) {
				cout << "Error: readNeuroml: model path '" << location << "' not found.\n";
				return;
			}
			name = location.substr( pos + 1 );
		}
		
		Element* locE = Neutral::create( "Cell", name, pa, Id::scratchId() );
		loc = locE->id();
	}
	
	nr.readModel( filename, loc );
#else
	cerr << "Error: readNeuroML: This MOOSE is not built with NeuroML compatibility.\n";
#endif
}

void Shell::writeNeuroml( const Conn* c, string filename, string location, int childnode )
{
#ifdef USE_NEUROML
	NeuromlWriter nw;
	
	Id loc( location );
	if ( loc.bad() ) {
		cerr << "Error: Shell::writeNeuroml: Path " << location << " does not exist.\n";
		return;
	}
	
	nw.writeModel( filename, loc );
#else
	cerr << "Error: writeNeuroml: This MOOSE is not built with NeuroML compatibility.\n";
#endif
}
// Static function
void Shell::createGateMaster( const Conn* c, Id chan, string gateName )
{
	assert( myNode() == 0 );
	
	IdGenerator idGen = Id::generator( chan.node() );
	::set< string >( chan(), "createGate", gateName, idGen );
	
#ifdef USE_MPI
	if ( chan.isGlobal() ) {
		send3< Id, string, IdGenerator >(
			c->target(), createGateSlot,
			chan, gateName, idGen );
	}
#endif
}

// Static function
void Shell::createGateWorker(
	const Conn* c,
	Id chan, string gateName, IdGenerator idGen )
{
	assert( myNode() != 0 );
	assert( chan.good() && chan.isGlobal() );
	
	::set< string, IdGenerator >( chan(), "createGate", gateName, idGen );
}


// Static function
/**
 * Assigns dt and optionally stage to a clock tick. If the Tick does
 * not exist, create it. The ClockJob and Tick0 are created by default.
 * I keep this function in the Shell because we'll want something
 * similar in Python. Otherwise it can readily go into the
 * GenesisParserWrapper.
 */
void Shell::setClock( const Conn* c, int clockNo, double dt,
				int stage )
{
#ifdef USE_MPI
	Shell* sh = static_cast< Shell* >( c->data() );
	if ( ! isSerial( ) && sh->myNode() == 0 ) {
		send3< int, double, int >( c->target(), parSetClockSlot,
			clockNo, dt, stage );
	}
#endif
	char line[20];
	sprintf( line, "t%d", clockNo );
	string TickName = line;
	string clockPath = string( "/sched/cj/" + TickName );
	Id id = Id::localId( clockPath );
	Id cj = Id::localId( "/sched/cj" );
	// Id id = sh->innerPath2eid( clockPath, "/", 1 );
	// Id cj = sh->innerPath2eid( "/sched/cj", "/", 1 );
	assert( cj.good() );
	Element* tick = 0;
	if ( id.zero() || id.bad() ) {
		if ( numNodes() > 1 ) {
			tick = Neutral::create( 
						"ParTick", TickName, cj, Id::scratchId() );
			assert( tick != 0 );
			bool ret = ::set< bool >( tick, "doSync", 1 );
			assert(ret );
			cout << "setClock: Creating parTick on node " << myNode() << endl;
			Eref pe = Id::postId( Id::AnyIndex ).eref();
			ret = Eref( tick ).add( "parTick", pe, "parTick", 
				ConnTainer::One2All );
			assert(ret );
		} else {
			tick = Neutral::create( 
						"Tick", TickName, cj, Id::scratchId() );
		}
	} else {
		tick = id();
	}
	assert ( tick != 0 && tick != Element::root() );
	::set< double >( tick, "dt", dt );
	if ( stage >= 0 ) // Otherwise leave it at earlier value.
		::set< int >( tick, "stage", stage );
	::set( cj(), "resched" );
	// Call the function
}

// static function
/**
 * Sets up the path controlled by a given clock tick. The final 
 * argument sets up the target finfo for the message. Usually this
 * is 'process' but some objects need multi-phase clocking so we
 * add the 'function' argument to specify what the target finfo is.
 * The function does a unique merge of the path
 * with the existing targets of the clock tick by checking if the
 * elements on the path are already tied to this tick. (This avoids
 * the N^2 problem of matching them against the list). If they are
 * on some other tick that message is dropped and this new one added.
 * The function does not reinit the clocks or reschedule them: the
 * simulation can resume right away.
 * It is the job of the parser to provide defaults
 * and to decode the path list from wildcards.
 */
void Shell::localUseClock( const Conn* c, 
	string tickName, string pathStr, string function )
{
	if ( tickName.length() < 3 )
		tickName = "/sched/cj/" + tickName;
	Id tickId = Id::localId( tickName );
	vector< Id > path;
	localGetWildcardList( c, pathStr, 1, path );

	if ( !tickId.good() || path.size() == 0 ) {
		// cout << "Shell::localUseClock@" << myNode() << ": Warning: no tick " << tickName << " or empty target path " << pathStr << endl;
		return;
	}
	innerUseClock( tickId, path, function );
}

void Shell::innerUseClock( Id tickId, vector< Id >& path, string function )
{
	Element* tick = tickId();
	assert ( tick != 0 );
	const Finfo* tickProc = tick->findFinfo( "process" );

	// vector< Conn > list;

	// Scan through path and check for existing process connections.
	// If they are to the same tick, skip the object
	// If they are to a different tick, delete the connection.
	vector< Id >::iterator i;
	for (i = path.begin(); i != path.end(); i++ ) {
		bool ret;
		assert ( !i->zero() );
		Element* e = ( *i )( );
		assert ( e && e != Element::root() );
		const Finfo* func = e->findFinfo( function );
		if ( func ) {
			Conn* c = e->targets( func->msg(), i->index() );
			if ( !c->good() ) {
				ret = Eref( tick ).add( tickProc->msg(), i->eref(), func->msg(),
					ConnTainer::Default );
				// ret = tickProc->add( tick, e, func );
				assert( ret );
			} else {
				if ( c->target().e != tick ) {
					i->eref().dropAll( func->msg() );
					Eref( tick ).add( tickProc->msg(), i->eref(), func->msg(),
						ConnTainer::Default);
					// tick->add( tickProc->msg(), e, func->msg() );
				}
			}
			delete c;

		} else {
			// This cannot be an 'assertion' error because the 
			// user might do a typo.
			cout << "Error: Shell::useClock: unknown function " << function << " in " << i->path() << endl;
		}
	}
}

/**
 * This function converts the path from relative, recent or other forms
 * to a canonical absolute path form that the wildcards can handle.
 */
void Shell::digestPath( string& path )
{
	// Here we deal with all sorts of awful cases involving current and
	// parent element paths.
	if ( path.length() == 0 )
		return;
	if ( path[0] == '/' ) // already canonical form.
		return;
	if ( path.length() == 1 ) {
		if ( path == "." ) {
			path = cwe_.path();
		} else if ( path == "^" ) {
			path = recentElement_.path();
		} else {
			path = cwe_.path() + "/" + path;
		}
	} else if ( path.length() == 2 ) {
		if ( path[0] == '.' ) {
			if ( path[1] == '/' ) { // start from cwe
				path = cwe_.path();
			} else if ( path[1] == '.' ) {
				if ( cwe_ == Id() ) {
					path = "/";
				} else  {
					path = cwe_.path();
					string::size_type pos = path.rfind( '/' );
					path = path.substr( 0, pos );
				}
			}
		} else {
			path = cwe_.path() + "/" + path;
		}
	} else {
		string::size_type pos = path.find_first_of( '/' );
		if ( pos == 1 ) {
			if ( path[0] == '^' ) 
				path = recentElement_.path() + path.substr( 1 );
			else if ( path[0] == '.' )
				path = cwe_.path() + path.substr( 1 );
			else
				path = cwe_.path() + "/" + path;
		} else if ( pos == 2 && path[0] == '.' && path[1] == '.' ) {
			if ( cwe_ == Id() ) {
				path = path.substr( 2 );
			} else { // get parent of cwe and tag path onto it.
				string temp = cwe_.path();
				string::size_type pos = temp.rfind( '/' );
				path = temp.substr( 0, pos ) + path.substr( 2 );
			}
		} else if ( pos != 0 ) {
			path = cwe_.path() + "/" + path;
		}
	}
	// Handle issues with initial double slash.
	if ( path[0] == '/' && path[1] == '/' )
		path = path.substr( 1 );
}
// static function
/** 
 * getWildcardList obtains a wildcard list specified by the path.
 * Normally the list is tested for uniqueness and sorted by pointer -
 * it becomes effectively random.
 * The flag specifies if we want a list in breadth-first order,
 * in which case commas are not permitted.
 *
 * In order to do this on multiple nodes, we simply issue the command
 * on each node and harvest all the responses.
 */
void Shell::getWildcardList( const Conn* c, string path, bool ordered )
{
	vector< Id > list;
	innerGetWildcardList( c, path, ordered, list );
	
	sendBack1< vector< Id > >( c, elistSlot, list );
}

/**
 * Inner utility function for getting wildcard list in parallel.
 * Used by other functions such as planarconnect.
 */
void Shell::innerGetWildcardList( const Conn* c, string path, 
	bool ordered, vector<Id>& list )
{
	//vector< Id > ret;

	localGetWildcardList( c, path, ordered, list );

	// We are done if running in serial
	if ( Shell::isSerial( ) )
		return;
	
#ifdef USE_MPI
	Shell* sh = static_cast< Shell* >( c->data() );
	vector< Nid > ret;
	unsigned int requestId = openOffNodeValueRequest< vector< Nid > >(
		sh, &ret, sh->numNodes() - 1 );
	
	// args are: path, ordered, requestId
	send3< string, bool, unsigned int >( 
		c->target(), requestWildcardSlot,
		path, ordered, requestId );
	
	// Retrieve values
	vector< Nid >* temp = closeOffNodeValueRequest< vector< Nid > >(
		sh, requestId
	);
	assert( &ret == temp );

	// cout << "innerGetWildcardList: on " << myNode() << " list nodes= ";
	for( vector< Nid >::iterator i = ret.begin(); i != ret.end(); i++ ) {
		list.push_back( Id( *i ) );
		// cout << *i << "." << i->node() << ", ";
		// cout << list.back() << "." << list.back().node() << "       ";
	}
	// cout << endl << flush;
#endif
}

/**
 * Does its stuff on local node only
 */
void Shell::localGetWildcardList( const Conn* c, string path, bool ordered,
	vector< Id >& list)
{
	//vector< Id > ret;
	static_cast< Shell* >( c->data() )->digestPath( path );

	// Finally, refer to the wildcard functions in Wildcard.cpp.
	if ( ordered )
		simpleWildcardFind( path, list );
	else
		wildcardFind( path, list );
	//ret.resize( list.size() );
// 	vector< Id >::iterator i;
// 	vector< Element* >::iterator j;

	//for (i = ret.begin(), j = list.begin(); j != list.end(); i++, j++ )
	//		*i = ( *j )->id();
	
	//GenesisParserWrapper::recvElist(conn, elist)
	// sendBack1< vector< Id > >( c, elistSlot, list );
}

/**
 * Utility function to find the ClockJob pointer
 */
Element* findCj()
{
	Id schedId;
	lookupGet< Id, string >( 
		Element::root(), "lookupChild", schedId, "sched" );
	assert( !schedId.bad() );
	Id cjId;
	lookupGet< Id, string >( 
		schedId(), "lookupChild", cjId, "cj" );
	assert( !cjId.bad() );
	return cjId();
}

void Shell::resched( const Conn* c )
{
#ifdef USE_MPI
	if ( myNode() == 0 ) {
		send0( c->target(), parReschedSlot );
		// getOffNodeValue< vector< Nid >, Nid >( c->target(), requestLeSlot, sh->numNodes(), &nret, parent );
	}
#endif
	// Should be a msg
	Element* cj = findCj();
	::set( cj, "resched" );
	Id kinetics = Id::localId( "/kinetics" );
	if ( kinetics.good() && 
		kinetics.eref().e->className() == "KineticManager" )
		::set( kinetics(), "resched" );
}

void Shell::reinit( const Conn* c )
{
#ifdef USE_MPI
	if ( myNode() == 0 ) {
		send0( c->target(), parReinitSlot );
		pollPostmaster(); // Needed so info goes out to other nodes.
	}
#endif
	// Should be a msg
	Element* cj = findCj();
	::set( cj, "reinit" );
}

void Shell::stop( const Conn* c )
{
	// Element* cj = findCj();
	// set( cj, "stop" ); // Not yet implemented
}

void Shell::step( const Conn* c, double time )
{
#ifdef USE_MPI
	if ( myNode() == 0 ) {
		send1< double >( c->target(), parStepSlot, time );
		// The send only goes out after at least one poll cycle.
		pollPostmaster();
	}
#endif
	// Should be a msg
	Element* cj = findCj();
	::set< double >( cj, "start", time );
}

/**
 * requestClocks builds a list of all clock times in order of clock
 * number. Puts these into a vector of doubles to send back to the
 * calling parser.
 * \todo: Need to fix requestClocks as it will give the wrong index
 * if we have non-contiguous clock ticks.
 */
void Shell::requestClocks( const Conn* c )
{
	// Here we fill up the clock timings.
	
	Element* cj = findCj();
	//RD assuming that cj is a simple element
	// assert for this (assert is not fool proof)
	assert(cj->numEntries() == 1);
	
	Conn* ct = cj->targets( "childSrc", 0 );// zero index for SE
	vector< double > times;

	while ( ct->good() ) {
		double dt;
		if ( get< double >( ct->target(), "dt", dt ) )
			times.push_back( dt );
		ct->increment();
	}
	delete ct;
	sendBack1< vector< double > >( c, clockSlot, times );

	/*
	Element* cj = findCj();
	vector< Conn > kids;
	vector< Conn >::iterator i;
	vector< double > times;
	cj->findFinfo( "childSrc" )->outgoingConns( cj, kids );
	double dt;
	for ( i = kids.begin(); i != kids.end(); i++ ) {
		if ( get< double >( i->targetElement(), "dt", dt ) )
			times.push_back( dt );
	}

	send1< vector< double > >( c->targetElement(), clockSlot, times );
	*/
}

void Shell::requestCurrTime( const Conn* c )
{
	Element* cj = findCj();
	string ret;
	const Finfo* f = cj->findFinfo( "currentTime" );
	assert( f != 0 );
	f->strGet( cj, ret );
	sendBack1< string >( c, getFieldSlot, ret );
}

void Shell::addMessage( const Conn* c,
	vector< Id > src, string srcField,
	vector< Id > dest, string destField )
{
	vector< Id >::iterator i;
	vector< Id >::iterator j;
	bool ok = 1;
	for ( i = src.begin(); i != src.end(); ++i ) {
		for ( j = dest.begin(); j != dest.end(); ++j ) {
			if ( !Shell::addSingleMessage( c, *i, srcField, *j, destField ) ) {
				cout << "Error: Shell::addMessage failed from " <<
					i->path() << " to " << j->path () << endl;
				ok = 0;
			}
		}
	}
	// if ( ok ) cout << "msg add OK\n";
}

bool Shell::addLocal( const Conn* c, 
	Id src, string srcField, 
	Id dest, string destField )
{
	return innerAddLocal( src, srcField, dest, destField );
}

bool Shell::innerAddLocal( 
	Id src, string srcField, Id dest, string destField )
{
	// cout << "innerAddLocal " << src << "." << src.node() << " to " << dest << "." << dest.node() << " on " << myNode() << endl << flush;
	return src.eref().add( srcField, dest.eref(), destField,
		ConnTainer::Default );
}

void Shell::parMsgErrorFunc( 
	const Conn* c, string errMsg, Id src, Id dest )
{
	cout << "Error: " << errMsg << " from " << src.path() << " to " <<
		dest.path() << endl;
}

void Shell::parMsgOkFunc( const Conn* c, Id src, Id dest )
{
	// cout << "OK: msg set up from " << src.path() << " to " << dest.path() << endl;
	cout << "internode msg setup OK\n";
}

#ifndef USE_MPI
bool Shell::addSingleMessage( const Conn* c, 
	Id src, string srcField, 
	Id dest, string destField )
{
	return innerAddLocal( src, srcField, dest, destField );
}

void Shell::addParallelSrc( const Conn* c, 
	Nid src, string srcField, 
	Nid dest, string destField )
{
	;
}

void Shell::addParallelDest( const Conn* c,
	Nid src, unsigned int srcSize, string srcTypeStr, 
	Nid dest, string destField )
{
	;
}

/*
Eref Shell::getPost( unsigned int node ) const
{
	return 0;
}
*/

Id Shell::parallelTraversePath( Id start, vector< string >& names )
{
	return Id::badId(); // Fails on single node version.
}

string Shell::eid2path( Id eid )
{
	return Shell::localEid2Path( eid );
}

// static function
void Shell::useClock( const Conn* c, string tickName, string path,
	string function )
{
	localUseClock( c, tickName, path, function );
}

unsigned int Shell::newIdBlock( unsigned int size )
{
	return 0;
}

void Shell::handleRequestNewIdBlock( const Conn* c,
	unsigned int size, unsigned int node, unsigned int requestId )
{ ; }

void Shell::handleReturnNewIdBlock( const Conn* c,
	unsigned int value, unsigned int requestId )
{ ; }

#endif // ndef USE_MPI

void Shell::addEdge( const Conn* c, Fid src, Fid dest, int connType )
{
	src.eref().add( src.fieldNum(), dest.eref(), 
		dest.fieldNum(), connType );
}

void Shell::deleteMessage( const Conn* c, Fid src, int msg )
{
	src.eref().drop( src.fieldNum(), msg );
}

void Shell::deleteMessageByDest( const Conn* c, 
	Id src, string srcField, Id dest, string destField )
{
}

void Shell::deleteEdge( const Conn* c, Fid src, Fid dest )
{
}

/**
 * static func.
 * listMessages builds a list of messages associated with the 
 * specified element on the named field, and sends it back to
 * the calling parser. It extracts the
 * target element from the connections, and puts this into a
 * vector of unsigned ints.
 */
void Shell::listMessages( const Conn* c,
				Id id, string field, bool isIncoming )
{
	// Shell* sh = static_cast< Shell* >( c->data() );
	vector< Id > ret;
	string remoteFields = "";
	if ( id.node() != 0 ) {
		cout << "listMessages: Sorry, cannot list off-node messages yet\n";
	} else {
		innerListMessages( c, id, field, isIncoming, ret, remoteFields );
	}
	sendBack2< vector< Id >, string >(
		c, listMessageSlot, ret, remoteFields );
}

void Shell::innerListMessages( const Conn* c,
				Id id, string field, bool isIncoming,
				vector< Id >& ret, string& remoteFields )
{
	assert( !id.bad() && id.node() == myNode() );
	Element* e = id();
	const Finfo* f = e->findFinfo( field );
	assert( f != 0 );

	// vector< pair< Element*, unsigned int > > list;
	string separator = "";
	Conn* tc = e->targets( f->msg(), id.index() );
	while( tc->good() ) {
		if ( tc->isDest() == isIncoming ) {              
			Eref tgt = tc->target();
			ret.push_back( tgt.id() );
			if ( tgt.id().node() == myNode() ) {
				const Finfo* targetFinfo = tgt.e->findFinfo( tc->targetMsg() );
				assert( targetFinfo != 0 );
				remoteFields = remoteFields + separator + targetFinfo->name();
			} else {
				remoteFields = remoteFields + separator + "proxy";
			}
			separator = ", ";
		}
		tc->increment();
	}
	delete tc;
}

/**
 * Node = Id::UnknownNode tells the system to do the default, based on the
 * node of cellpath parent.
 * So it needs to find cellpath parent and decide there. This includes
 * building on a global such as /library.
 * node = 0 .. numNodes - 1 tells the system to build on specified node.
 * This is legal only if the parent is /root
 */
void Shell::readCell(
	const Conn* c,
	string filename,
	string cellpath,
	vector< double > globalParms,
	unsigned int node )
{
	if ( node >= numNodes() && node != Id::UnknownNode && node != Id::GlobalNode ) {
		cerr
			<< "readcell: warning: requested node " << node
			<< " > numNodes(" << numNodes() << "), using 0\n";
		node = 0;
	}

	string::size_type pos = cellpath.find_last_of( "/" );
	Id pa;
	string cellname;
	if ( pos == string::npos ) {
		pa = getCwe( c->target() );
		cellname = cellpath;
	} else if ( pos == 0 ) {
		pa = Id();
		cellname = cellpath.substr( 1 );
	} else {
		pa = Id( cellpath.substr( 0, pos ), "/" );
		if ( pa.bad() ) {
			cout << "Error: readCell: cell path '" << cellpath << "' not found.\n";
			return;
		}
		cellname = cellpath.substr( pos + 1 );
	}
	
	//~ Id cellId;
	//~ if ( node == Id::UnknownNode ) { // Let load-balancer decide
		//~ cellId = Id::childId( pa );
	//~ } else { // Legal node # request
		//~ if ( pa.node() == node || pa == Id() ) { // make on unode.
			//~ cellId = Id::makeIdOnNode( unode );
		//~ }
	//~ }
	
	if ( node == Id::UnknownNode ) { // Let load-balancer decide
		node = Id::childNode( pa );
	} else { // Legal node # request
		if ( pa.node() != node && pa != Id() ) {
			cerr <<
				"Error: Shell::readCell: Don't know to create " << cellpath <<
				" on " << pa.path() << ". Both must be on the same node.\n";
			return;
		}
	}
	
	// Now we know both where to put the child.
	IdGenerator idGen = Id::generator( node );
	if ( node == 0 ) {
		localReadCell( c, filename, cellname, globalParms, pa, idGen );
	} else if ( idGen.isGlobal() ) {
		localReadCell( c, filename, cellname, globalParms, pa, idGen );
#ifdef USE_MPI
		send5< string, string, vector< double >, Nid, IdGenerator >( 
			c->target(), parReadCellSlot,
			filename, cellname, globalParms, Nid( pa ), idGen );
	} else { // Off on some specific target node.
		sendTo5< string, string, vector< double >, Nid, IdGenerator >( 
			c->target(), parReadCellSlot, node - 1,
			filename, cellname, globalParms, Nid( pa ), idGen );
#endif
	}
}

void Shell::localReadCell(
	const Conn* c, 
	string filename,
	string cellname,
	vector< double > globalParms,
	Nid pa,
	IdGenerator idGen )
{
	ReadCell rc( globalParms, idGen );
	
	rc.read( filename, cellname, Id( pa ) );
}

void Shell::setupAlpha( const Conn* c, Id gateId,
				vector< double > parms )
{
	static const Finfo* setupAlphaFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "setupAlpha" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::setupAlpha: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	::set< vector< double > >( gate, setupAlphaFinfo, parms );
	
#ifdef USE_MPI
	if ( gateId.isGlobal() && myNode() == 0 )
		send2< Id, vector< double > >(
			c->target(), setupAlphaSlot,
			gateId, parms );
#endif
}

void Shell::setupTau( const Conn* c, Id gateId,
				vector< double > parms )
{
	static const Finfo* setupTauFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "setupTau" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::setupTau: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	::set< vector< double > >( gate, setupTauFinfo, parms );
	
#ifdef USE_MPI
	if ( gateId.isGlobal() && myNode() == 0 )
		send2< Id, vector< double > >(
			c->target(), setupTauSlot,
			gateId, parms );
#endif
}

void Shell::tweakAlpha( const Conn* c, Id gateId )
{
	static const Finfo* tweakAlphaFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "tweakAlpha" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::tweakAlpha: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	::set( gate, tweakAlphaFinfo );
	
#ifdef USE_MPI
	if ( gateId.isGlobal() && myNode() == 0 )
		send1< Id >(
			c->target(), tweakAlphaSlot,
			gateId );
#endif
}

void Shell::tweakTau( const Conn* c, Id gateId )
{
	static const Finfo* tweakTauFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "tweakTau" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::tweakTau: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	::set( gate, tweakTauFinfo );
	
#ifdef USE_MPI
	if ( gateId.isGlobal() && myNode() == 0 )
		send1< Id >(
			c->target(), tweakTauSlot,
			gateId );
#endif
}

void Shell::setupGate( const Conn* c, Id gateId,
				vector< double > parms )
{
	static const Finfo* setupGateFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "setupGate" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::setupGate: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	::set< vector< double > >( gate, setupGateFinfo, parms );
	
#ifdef USE_MPI
	if ( gateId.isGlobal() && myNode() == 0 )
		send2< Id, vector< double > >(
			c->target(), setupGateSlot,
			gateId, parms );
#endif
}

//////////////////////////////////////////////////////////////////
// SimDump functions
//////////////////////////////////////////////////////////////////
/**
 * readDumpFile loads in a simulation from a GENESIS simdump file.
 * Works specially for reading in Kinetikit dump files.
 * In many old simulations the dump files are loaded in as regular script 
 * files. Here it is the job of the GENESIS parser to detect that these
 * are simDump files and treat them accordingly.
 * In a few cases the simulation commands and the simdump file have been
 * mixed up in a single file. MOOSE does not handle such cases.
 * This uses a local instance of SimDump, and does not interfere
 * with the private version in the Shell.
 */
void Shell::readDumpFile( const Conn* c, string filename )
{
	SimDump localSid;
	
	localSid.read( filename );
}

/**
 * writeDumpFile takes the specified comma-separated path and generates
 * an old-style GENESIS simdump file. Used mostly for dumping kinetic
 * models to kkit format.
 * This is equivalent to the simdump command from the GENESIS parser.
 * This uses the private SimDump object on the Shell because the
 * simObjDump function may need to set its state.
 */
void Shell::writeDumpFile( const Conn* c, string filename, string path )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	sh->simDump_->write( filename, path );
}

/**
 * This function sets up the sequence of fields used in a dumpfile
 * This uses the private SimDump object on the Shell because the
 * writeDumpFile and simObjDump functions may need to use this state
 * information.
 * First argument is the function call, second is the name of the class.
 */
void Shell::simObjDump( const Conn* c, string fields )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	sh->simDump_->simObjDump( fields );
}
/**
 * This function reads in a single dumpfile line.
 * It is only for the special case where the GenesisParser is reading
 * a dumpfile as if it were a script file.
 * This uses the private SimDump object on the Shell because the
 * simObjDump function may need to set its state.
 */
void Shell::simUndump( const Conn* c, string args )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	sh->simDump_->simUndump( args );
}

void Shell::loadtab( const Conn* c, string data )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	sh->innerLoadTab( data );
}

void Shell::tabop( const Conn* c, Id tab, char op, double min, double max )
{
	::set< char, double, double >( tab(), "tabop", op, min, max );
}

//////////////////////////////////////////////////////////////////
// File handling functions
//////////////////////////////////////////////////////////////////

///\todo These things should NOT be globals.
map <string, FILE*> Shell::filehandler;
vector <string> Shell::filenames;
vector <string> Shell::modes;
vector <FILE*> Shell::filehandles;

void Shell::openFile( const Conn* c, string filename, string mode )
{
	FILE* o = fopen( filename.c_str(), mode.c_str() );
	if (o == NULL){
		cout << "Error: Shell::openFile: Cannot openfile " << filename << endl;
		return;
	}
	map<string, FILE*>::iterator iter = filehandler.find(filename);
	if (iter != filehandler.end() ){
		cout << "File " << filename << " already being used." << endl;
		return;
	}
	//filehandler[filename] = o;
	filenames.push_back(filename);
	modes.push_back(mode);
	filehandles.push_back(o);
}



void Shell::writeFile( const Conn* c, string filename, string text )
{
	size_t i = 0;
	if ( filenames.size() )
		while (filenames[i] != filename && ++i);
	
	if ( i < filenames.size() ){
		if ( !( modes[i] == "w" || modes[i] == "a" ) ) {
			cout << "Error: Shell::writeFile: The file has not been opened in write mode" << endl;
			return;
		}
		fprintf(filehandles[i], "%s", text.c_str());
	}
	else {
		cout << "Error: Shell::writeFile: File "<< filename << " not opened!!" << endl;
		return;
	}
}

void Shell::flushFile( const Conn* c, string filename )
{
	size_t i = 0;
	if ( filenames.size() )
		while (filenames[i] != filename && ++i);
	
	if ( i < filenames.size() ){
		if ( !( modes[i] == "w" || modes[i] == "a" ) ) {
			cout << "Error: Shell::flushFile: The file has not been opened in write mode" << endl;
			return;
		}
		fflush(filehandles[i]);
	}
	else {
		cout << "Error: Shell::flushFile: File "<< filename << " not opened!!" << endl;
		return;
	}
}

void Shell::closeFile( const Conn* c, string filename ){
	size_t i = 0;
	if ( filenames.size() )
		while (filenames[i] != filename && ++i);
	
	if ( i < filenames.size() ){
		if ( fclose(filehandles[i]) != 0 ) {
			cout << "Error: Shell::closeFile: Could not close the file." << endl;
			return;
		}
		filenames.erase( filenames.begin() + i );
		modes.erase( modes.begin() + i );
		filehandles.erase( filehandles.begin() + i );
	}
	else {
		cout << "Error: Shell::closeFile: File "<< filename << " not opened!!" << endl;
		return;
	}
}

void Shell::listFiles( const Conn* c ){
	string ret = "";
	for ( size_t i = 0; i < filenames.size(); i++ ) 
		ret = ret + filenames[i] + "\n";
	sendBack1< string >( c, getFieldSlot, ret );	
}


/*
Limitation: lines should be shorter than 1000 chars
*/
void Shell::readFile( const Conn* c, string filename, bool linemode ){
	size_t i = 0;
	if ( filenames.size() )
		while (filenames[i] != filename && ++i);
	
	if ( i < filenames.size() ){
		char str[1000];
		if (linemode){
			fgets( str, 1000, filehandles[i] );
		}
		else 
			fscanf( filehandles[i], "%s", str);
		string ret = str;
		if (ret[ ret.size() -1 ] == '\n' && !linemode)
			ret.erase( ret.end() - 1 );
		sendBack1< string >( c, getFieldSlot, ret );
	}
	else {
		cout << "Error:: File "<< filename << " not opened!!" << endl;
		return;
	}
}



//////////////////////////////////////////////////////////////////
// Helper functions.
//////////////////////////////////////////////////////////////////

/**
 * Creates a child element with the specified id, and schedules it.
 * Regular function.
 */
Element* Shell::create( const string& type, const string& name, 
		Id parent, Id id )
{
	Element* child = Neutral::create( type, name, parent, id );
	if ( child ) {
		recentElement_ = child->id();
	}

	return child;
}

// Regular function
Element* Shell::createArray( const string& type, const string& name, 
		Id parent, Id id, int n )
{	
	Element* child = Neutral::createArray( type, name, parent, id, n );
	if ( child ) {
		recentElement_ = child->id();
	}
	return child;
	
	/*const Cinfo* c = Cinfo::find( type );
	Element* p = parent();
	if ( !p ) {
		cout << "Error: Shell::create: No parent " << p << endl;
		return 0;
	}

	const Finfo* childSrc = p->findFinfo( "childSrc" );
	if ( !childSrc ) {
		// Sorry, couldn't resist it.
		cout << "Error: Shell::create: parent cannot handle child\n";
		return 0;
	}
	if ( c != 0 && p != 0 ) {
		Element* e = c->createArray( id, name, n, 0 );
		bool ret = childSrc->add( p, e, e->findFinfo( "child" ) );
		assert( ret );
		recentElement_ = id;
		ret = c->schedule( e );
		assert( ret );
		return e;
	} else  {
		cout << "Error: Shell::create: Unable to find type " <<
			type << endl;
	}
	return 0;
	*/
}



// Regular function
void Shell::destroy( Id victim )
{
	// cout << "in Shell::destroy\n";
	Element* e = victim();
	if ( !e ) {
		cout << "Error: Shell::destroy: No element " << victim << endl;
		return;
	}

	::set( e, "destroy" );
}

/**
 * Static function. True till simulator quits.
 */
bool Shell::running()
{
	return running_;
}

/**
 * Tells all nodes to quit, then quits. Should be called only on 
 * master node.
 */
void Shell::quit( const Conn* c )
{
#ifdef USE_MPI
	send0( c->target(), parQuitSlot );
	pollPostmaster(); // needed to send the message out.
#endif
	running_ = 0;
	// exit( 0 );
}

/**
 * static function. Sets 'running' flag to zero, in due course the
 * event loop will terminate.
 */
void Shell::innerQuit( const Conn* c)
{
	running_ = 0;
	// Brute force, till we sort out the Barrier issue for quits.
	// exit( 0 );
}

//////////////////////////////////////////////////////////////////////
// Deleted stuff.
//////////////////////////////////////////////////////////////////////

#ifdef OLD_SHELL_FUNCS
void Shell::pwe() const
{
	cout << cwe_ << endl;
}

void Shell::ce( Id dest )
{
	if ( dest() )
		cwe_ = dest;
}

void Shell::le ( Id eid )
{
	Element* e = eid( );
	if ( e ) {
		vector< Id > elist;
		vector< Id >::iterator i;
		get( e, "childList", elist );
		for ( i = elist.begin(); i != elist.end(); i++ ) {
			if ( ( *i )() != 0 )
				cout << ( *i )()->name() << endl;
		}
	}
}
#endif
