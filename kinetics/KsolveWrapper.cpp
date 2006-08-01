/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "header.h"
#include "RateTerm.h"
#include "SparseMatrix.h"
#include "Ksolve.h"
#include "KsolveWrapper.h"


Finfo* KsolveWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// EvalField definitions
///////////////////////////////////////////////////////
	new ValueFinfo< string >(
		"path", &KsolveWrapper::getPath, 
		&KsolveWrapper::setPath, "string" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processOut", &KsolveWrapper::getProcessSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitOut", &KsolveWrapper::getReinitSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"molOut", &KsolveWrapper::getMolSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"rateOut", &KsolveWrapper::getRateSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< ProcInfo >(
		"processIn", &KsolveWrapper::processFunc,
		&KsolveWrapper::getProcessInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &KsolveWrapper::reinitFunc,
		&KsolveWrapper::getReinitInConn, "" ),
	new Dest3Finfo< double, double, int >(
		"molIn", &KsolveWrapper::molFunc,
		&KsolveWrapper::getMolSolveConn, "", 1 ),
	new Dest1Finfo< double >(
		"rateIn", &KsolveWrapper::rateFunc,
		&KsolveWrapper::getRateSolveConn, "", 1 ),
	new Dest2Finfo< double, double >(
		"reacIn", &KsolveWrapper::reacFunc,
		&KsolveWrapper::getReacSolveConn, "", 1 ),
	new Dest3Finfo< double, double, double >(
		"enzIn", &KsolveWrapper::enzFunc,
		&KsolveWrapper::getEnzSolveConn, "", 1 ),
	new Dest3Finfo< double, double, double >(
		"mmEnzIn", &KsolveWrapper::mmEnzFunc,
		&KsolveWrapper::getMmEnzSolveConn, "", 1 ),
	new Dest1Finfo< double >(
		"tabIn", &KsolveWrapper::tabFunc,
		&KsolveWrapper::getTabSolveConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"molSolve", &KsolveWrapper::getMolSolveConn,
		"processOut, reinitOut, molOut, molIn" ),
	new SharedFinfo(
		"reacSolve", &KsolveWrapper::getReacSolveConn,
		"processOut, reinitOut, reacIn" ),
	new SharedFinfo(
		"enzSolve", &KsolveWrapper::getEnzSolveConn,
		"processOut, reinitOut, enzIn" ),
	new SharedFinfo(
		"mmEnzSolve", &KsolveWrapper::getMmEnzSolveConn,
		"processOut, reinitOut, mmEnzIn" ),
	new SharedFinfo(
		"tabSolve", &KsolveWrapper::getTabSolveConn,
		"processOut, reinitOut, tabIn" ),
	new SharedFinfo(
		"rateSolve", &KsolveWrapper::getRateSolveConn,
		"processOut, reinitOut, rateOut, rateIn" ),
};

const Cinfo KsolveWrapper::cinfo_(
	"Ksolve",
	"Upinder S. Bhalla, June 2006, NCBS",
	"Ksolve: Wrapper object for zombifying reaction systems. Interfaces\nboth with the reaction system (molecules, reactions, enzymes\nand user defined rate terms) and also with the Stoich\nclass which generates the stoichiometry matrix and \nhandles the derivative calculations.",
	"Neutral",
	KsolveWrapper::fieldArray_,
	sizeof(KsolveWrapper::fieldArray_)/sizeof(Finfo *),
	&KsolveWrapper::create
);

///////////////////////////////////////////////////
// EvalField function definitions
///////////////////////////////////////////////////

string KsolveWrapper::localGetPath() const
{
			return path_;
}
void KsolveWrapper::localSetPath( const string& value ) {
			path_ = value;
			vector< Element* > ret;
			vector< Element* >::iterator i;
			Field solveSrc( this, "molSolve" );
			Element::startFind( path_, ret );
			for ( i = ret.begin(); i != ret.end(); i++ ) {
				if ( ( *i )->cinfo()->name() == "Molecule" ) {
					molZombify( *i, solveSrc );
				}
			}
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processInConnKsolveLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( KsolveWrapper, processInConn_ );
	return reinterpret_cast< KsolveWrapper* >( ( unsigned long )c - OFFSET );
}

Element* reinitInConnKsolveLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( KsolveWrapper, reinitInConn_ );
	return reinterpret_cast< KsolveWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
void KsolveWrapper::molZombify( Element* e, Field& solveSrc )
{
	Field f( e, "process" );
	if ( !f.dropAll() ) {
		cerr << "Error: Failed to delete process message into " <<
			e->path() << "\n";
	}
	Field ms( e, "solve" );
	if ( !solveSrc.add( ms ) ) {
		cerr << "Error: Failed to add molSolve message from solver " <<
			path() << " to zombie " << e->path() << "\n";
	}
}
