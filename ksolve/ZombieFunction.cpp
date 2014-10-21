/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"

#include "Variable.h"
#include "Function.h"
#include "ZombieFunction.h"

#include "FuncTerm.h"
#include "RateTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "VoxelPoolsBase.h"
#include "../mesh/VoxelJunction.h"
#include "ZombiePoolInterface.h"
#include "Stoich.h"

#define EPSILON 1e-15

const Cinfo* ZombieFunction::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: mostly inherited from Function
		//////////////////////////////////////////////////////////////
    static ElementValueFinfo< ZombieFunction, string > expr(
        "expr",
        "Mathematical expression defining the function. The underlying parser\n"
        "is muParser. Hence the available functions and operators are (from\n"
        "muParser docs):\n"
        "\nFunctions\n"
        "Name        args    explanation\n"
        "sin         1       sine function\n"
        "cos         1       cosine function\n"
        "tan         1       tangens function\n"
        "asin        1       arcus sine function\n"
        "acos        1       arcus cosine function\n"
        "atan        1       arcus tangens function\n"
        "sinh        1       hyperbolic sine function\n"
        "cosh        1       hyperbolic cosine\n"
        "tanh        1       hyperbolic tangens function\n"
        "asinh       1       hyperbolic arcus sine function\n"
        "acosh       1       hyperbolic arcus tangens function\n"
        "atanh       1       hyperbolic arcur tangens function\n"
        "log2        1       logarithm to the base 2\n"
        "log10       1       logarithm to the base 10\n"
        "log         1       logarithm to the base 10\n"
        "ln  1       logarithm to base e (2.71828...)\n"
        "exp         1       e raised to the power of x\n"
        "sqrt        1       square root of a value\n"
        "sign        1       sign function -1 if x<0; 1 if x>0\n"
        "rint        1       round to nearest integer\n"
        "abs         1       absolute value\n"
        "min         var.    min of all arguments\n"
        "max         var.    max of all arguments\n"
        "sum         var.    sum of all arguments\n"
        "avg         var.    mean value of all arguments\n"
        "\nOperators\n"
        "Op  meaning         prioroty\n"
        "=   assignement     -1\n"
        "&&  logical and     1\n"
        "||  logical or      2\n"
        "<=  less or equal   4\n"
        ">=  greater or equal        4\n"
        "!=  not equal       4\n"
        "==  equal   4\n"
        ">   greater than    4\n"
        "<   less than       4\n"
        "+   addition        5\n"
        "-   subtraction     5\n"
        "*   multiplication  6\n"
        "/   division        6\n"
        "^   raise x to the power of y       7\n"
        "\n"
        "?:  if then else operator   C++ style syntax\n",
        &ZombieFunction::setExpr,
        &Function::getExpr);

	
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions: All inherited from Function
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions: All inherited from Function
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions: Override Function
		//////////////////////////////////////////////////////////////
    static DestFinfo process( "process",
              "Handles process call, updates internal time stamp.",
              new ProcOpFunc< ZombieFunction >( &ZombieFunction::process) );
    static DestFinfo reinit( "reinit",
             "Handles reinit call.",
             new ProcOpFunc< ZombieFunction >( &ZombieFunction::reinit ) );
    static Finfo* processShared[] =
            {
				&process, &reinit
            };
    
    static SharedFinfo proc( "proc",
             "This is a shared message to receive Process messages "
             "from the scheduler objects."
             "The first entry in the shared msg is a MsgDest "
             "for the Process operation. It has a single argument, "
             "ProcInfo, which holds lots of information about current "
             "time, thread, dt and so on. The second entry is a MsgDest "
             "for the Reinit operation. It also uses ProcInfo. ",
             processShared, sizeof( processShared ) / sizeof( Finfo* )
             );

	// Note that here the isOneZombie_ flag on the Dinfo constructor is
	// true. This means that the duplicate and copy operations only make
	// one copy, regardless of how big the array of zombie pools.
	// The assumption is that each Id has a single pool, which can be
	// present in many voxels.
    static Finfo *functionFinfos[] =
            {
                &expr,
                &proc,
            };

    static string doc[] =
            {
                "Name", "ZombieFunction",
                "Author", "Upi Bhalla",
                "Description",
                "ZombieFunction: Takes over Function, which is a general "
				"purpose function calculator using real numbers."
			};

	static Dinfo< ZombieFunction > dinfo;
	static Cinfo zombieFunctionCinfo (
		"ZombieFunction",
		Function::initCinfo(),
		functionFinfos,
		sizeof(functionFinfos) / sizeof(Finfo*),
		&dinfo,
        doc,
       sizeof(doc)/sizeof(string)
	);

	return &zombieFunctionCinfo;
}




//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieFunctionCinfo = ZombieFunction::initCinfo();

ZombieFunction::ZombieFunction()
{;}

ZombieFunction::~ZombieFunction()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////
void ZombieFunction::process(const Eref &e, ProcPtr p)
{;}

void ZombieFunction::reinit(const Eref &e, ProcPtr p)
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieFunction::setExpr( const Eref& e, string v )
{
	Function::setExpr( e, v );
	if ( _stoich ) {
		Stoich* s = reinterpret_cast< Stoich* >( _stoich );
		s->setFunctionExpr( e, v );
	} else {
		cout << "Warning: ZombieFunction::setExpr: specified entry is not a FuncRateTerm.\n";
	}
}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

void ZombieFunction::setSolver( Id ksolve, Id dsolve )
{
	if ( ksolve.element()->cinfo()->isA( "Ksolve" ) ||
					ksolve.element()->cinfo()->isA( "Gsolve" ) ) {
		Id sid = Field< Id >::get( ksolve, "stoich" );
			_stoich = ObjId( sid, 0 ).data();
	} else if ( ksolve == Id() ) {
			_stoich = 0;
	} else {
			cout << "Warning:ZombieFunction::vSetSolver: solver class " << 
					ksolve.element()->cinfo()->name() << 
					" not known.\nShould be Ksolve or Gsolve\n";
			_stoich = 0;
	}
	
	/*
	if ( dsolve.element()->cinfo()->isA( "Dsolve" ) ) {
			dsolve_= ObjId( dsolve, 0 ).data();
	} else if ( dsolve == Id() ) {
			dsolve_ = 0;
	} else {
			cout << "Warning:ZombieFunction::vSetSolver: solver class " << 
					dsolve.element()->cinfo()->name() << 
					" not known.\nShould be Dsolve\n";
			dsolve_ = 0;
	}
	*/
}

void ZombieFunction::zombify( Element* orig, const Cinfo* zClass,
					Id ksolve, Id dsolve )
{
	if ( orig->cinfo() == zClass )
			return;
	// unsigned int start = orig->localDataStart();
	unsigned int num = orig->numLocalData();
	if ( num == 0 )
		return;
	if ( num > 1 )
		cout << "ZombieFunction::zombify: Warning: ZombieFunction doesn't\n"
				"handle volumes yet. Proceeding without this.\n";

	// We can swap the class because the class data is identical, just 
	// the moose expr and process handlers are different.
	if ( orig->cinfo() == ZombieFunction::initCinfo() ) { // unzombify
		orig->replaceCinfo( Function::initCinfo() );
	} else { // zombify
		orig->replaceCinfo( ZombieFunction::initCinfo() );
		ZombieFunction* zf = reinterpret_cast< ZombieFunction *>(
						Eref( orig, 0 ).data() );
		zf->setSolver( ksolve, dsolve );
	}
}
