/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include <fstream>
#include <math.h>
#include "header.h"
#include "moose.h"
#include "../element/Neutral.h"
#include "../builtins/Interpol.h"
#include "../builtins/Table.h"
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"

extern void testMolecule(); // Defined in Molecule.cpp
extern void testEnzyme(); // Defined in Enzyme.cpp
extern void testKinSparseMatrix(); // Defined in KinSparseMatrix.cpp
extern void testMathFunc(); //Defined in MathFunc.cpp
void testStoich();
void testKintegrator();
#ifdef USE_GSL
void testGslIntegrator();
#endif // USE_GSL

void testKinetics()
{
	testMolecule();
	testEnzyme();
	testKinSparseMatrix();
	testStoich();
	testMathFunc();
	testKintegrator();
#ifdef USE_GSL
	testGslIntegrator();
#endif // USE_GSL
}

//////////////////////////////////////////////////////////////////
// Here we set up a small reaction system for testing with the
// Stoich class.
//////////////////////////////////////////////////////////////////

void testStoich()
{
	cout << "\nTesting Stoich" << flush;
	const unsigned int NUM_COMPT = 10;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(),
		Id::scratchId() );
	vector< Element* >m;
	vector< Element* >r;
	char name[10];
	bool ret;
	const Cinfo* molCinfo = Cinfo::find( "Molecule" );
	assert( molCinfo != 0 );
	const Finfo* rFinfo = molCinfo->findFinfo( "reac" );
	assert( rFinfo != 0 );

	const Cinfo* reacCinfo = Cinfo::find( "Reaction" );
	assert( reacCinfo != 0 );
	const Finfo* sFinfo = reacCinfo->findFinfo( "sub" );
	assert( sFinfo != 0 );
	const Finfo* pFinfo = reacCinfo->findFinfo( "prd" );
	assert( pFinfo != 0 );
	//////////////////////////////////////////////////////////////////
	// Create a linear sequence of molecules with reactions between.
	//////////////////////////////////////////////////////////////////
	for ( unsigned int i = 0; i < NUM_COMPT; i++ ) {
		sprintf( name, "m%d", i );
		Element* mtemp = Neutral::create( "Molecule", name, n->id(),
			Id::scratchId() );
		assert( mtemp != 0 );
		set< double >( mtemp, "nInit", 1.0 * i );
		set< double >( mtemp, "n", 1.0 * i );
		set< int >( mtemp, "mode", 0 );
		m.push_back( mtemp );

		if ( i > 0 ) {
			sprintf( name, "r%d", i );
			Element* rtemp = 
				Neutral::create( "Reaction", name, n->id(),
					Id::scratchId() );
			assert( rtemp != 0 );
			set< double >( rtemp, "kf", 0.1 );
			set< double >( rtemp, "kb", 0.1 );
			r.push_back( rtemp );
			ret = Eref( m[ i - 1 ] ).add( "reac", rtemp, "sub" );
			ASSERT( ret, "adding msg 0" );
			ret = Eref( mtemp ).add( "reac", rtemp, "prd" );
			ASSERT( ret, "adding msg 1" );
		}
	}

	///////////////////////////////////////////////////////////
	// Assign reaction system to a Stoich object
	///////////////////////////////////////////////////////////

	Element* stoich = Neutral::create( "Stoich", "s", Element::root()->id(),
		Id::scratchId() );

	ret = set< string >( stoich, "path", "/n/##" );
	ASSERT( ret, "Setting path" );

	Stoich* s = static_cast< Stoich* >( stoich->data() );
	ASSERT( s->N_.nRows() == NUM_COMPT, "num Species" ); 
	ASSERT( s->nMols_ == NUM_COMPT, "num Species" ); 
	ASSERT( s->N_.nColumns() == NUM_COMPT - 1, "num Reacns" ); 
	ASSERT( s->nReacs_ == NUM_COMPT - 1, "num Reacns" ); 
	ASSERT( s->reacMap_.size() == NUM_COMPT - 1, "num Reacns" ); 
	ASSERT( s->molMap_.size() == 10, "numSpecies" );
	// Note that the order of the species in the matrix is 
	// ill-defined because of the properties of the STL sort operation.
	// So here we need to look up the order based on the mol_map
	// Reac order has also been scrambled.
	
	// cout << s->N_;

	///////////////////////////////////////////////////////////
	// Check that stoich matrix is correct.
	///////////////////////////////////////////////////////////
	unsigned int molNum;
	int entry;
	map< Eref, unsigned int >::iterator k;
	for ( unsigned int i = 0; i < NUM_COMPT; i++ ) {
		k = s->molMap_.find( m[i] );
		ASSERT( k != s->molMap_.end(), "look up molecule" );
		molNum = k->second;
		ASSERT( s->Sinit_[molNum] == 1.0 * i, "mol sequence" );
	}
	for ( unsigned int i = 0; i < NUM_COMPT; i++ ) {
		k = s->molMap_.find( m[i] );
		ASSERT( k != s->molMap_.end(), "look up molecule" );
		molNum = k->second;
		for ( unsigned int j = 0; j < NUM_COMPT - 1; j++ ) {
			k = s->reacMap_.find( r[j] );
			ASSERT( k != s->reacMap_.end(),
					"look up reac" );
			unsigned int reacNum = k->second;
			entry = s->N_.get( molNum, reacNum );
			if ( i == j )
				assert( entry == -1 );
			else if ( i == j + 1 )
				assert( entry == 1 );
			else
				assert( entry == 0 );
		}
	}

	///////////////////////////////////////////////////////////
	// Run an update step on this reaction system
	///////////////////////////////////////////////////////////
	cout << "\nTesting Stoich deriv calculation" << flush;
	for ( unsigned int i = 0; i < NUM_COMPT; i++ )
		s->S_[i] = s->Sinit_[i];

	const double EPSILON = 1e-10;

	vector< double > yprime( NUM_COMPT, 0.0 );
	s->updateRates( &yprime, 1.0 );
	for ( unsigned int i = 0; i < NUM_COMPT; i++ ) {
		k = s->molMap_.find( m[i] );
		molNum = k->second;
		if ( i == 0 ) {
			ASSERT( fabs( yprime[molNum] - 0.1 ) < EPSILON, "update");
		} else if ( i == 9 ) {
			ASSERT( fabs( yprime[molNum] + 0.1 ) < EPSILON, "update");
		} else {
			ASSERT( fabs( yprime[molNum] ) < EPSILON, "update" );
		}
	}

	///////////////////////////////////////////////////////////
	// Connect up stoich to KineticHub
	///////////////////////////////////////////////////////////
	
	cout << "\nTesting Stoich zombie data access" << flush;
	// Clean out the old stoich
	set( stoich, "destroy" );

	stoich = Neutral::create( "Stoich", "s", Element::root()->id(),
		Id::scratchId() );
	Element* hub = Neutral::create( "KineticHub", "hub", Element::root()->id(),
		Id::scratchId() );
	// ret = stoich->findFinfo( "hub" )->add( stoich, hub, hub->findFinfo( "hub" ) );
	ret = Eref( stoich ).add( "hub", hub, "hub" );
	ASSERT( ret, "connecting stoich to hub" );

	// Rebuild the path now that the hub is connected.
	ret = set< string >( stoich, "path", "/n/##" );
	ASSERT( ret, "Setting path" );

	ret = set< double >( m[7], "n", 1234.5 );
	ASSERT( ret, "Setting value" );
	s = static_cast< Stoich* >( stoich->data() );
	k = s->molMap_.find( m[7] );
	molNum = k->second;
	ASSERT( s->S_[molNum] == 1234.5, "Setting value" );

	k = s->molMap_.find( m[3] );
	molNum = k->second;
	s->Sinit_[molNum] = 543210.0;
	double dret;
	ret = get< double >( m[3], "nInit", dret );
	ASSERT( ret, "Getting value" );
	ASSERT( dret == 543210.0, "Getting value" );

	k = s->molMap_.find( m[7] );
	molNum = k->second;
	s->S_[molNum] = 343434.0;
	ret = get< double >( m[7], "n", dret );
	ASSERT( ret, "Getting value" );
	ASSERT( dret == 343434.0, "Getting value" );

	// This extra check is in case there was something strange that
	// failed on a second access to the set of zombies.
	k = s->molMap_.find( m[4] );
	molNum = k->second;
	s->S_[molNum] = 10203.5;
	ret = get< double >( m[4], "n", dret );
	ASSERT( ret, "Getting another value" );
	ASSERT( dret == 10203.5, "Getting another value" );

	/////////////////////////////////////////////////////////
	// Here we try reaction acces.
	/////////////////////////////////////////////////////////
	unsigned int index;
	ret = set< double >( r[5], "kf", 111.222 );
	ASSERT( ret, "Setting reac value" );
	k = s->reacMap_.find( r[5] );
	ASSERT( k != s->reacMap_.end(), "reac values" );
	index = k->second;
	ASSERT( s->rates_[index]->getR1() == 111.222, "Setting reac value" );
	ASSERT( s->rates_[index]->getR2() == 0.1, "Setting reac value" );

	k = s->reacMap_.find( r[2] );
	ASSERT( k != s->reacMap_.end(), "reac values" );
	index = k->second;
	s->rates_[index]->setR2( 999.888 );
	ret = get< double >( r[2], "kb", dret );
	ASSERT( ret, "Getting reac value" );
	ASSERT( dret == 999.888, "Getting reac value" );
	ret = get< double >( r[2], "kf", dret );
	ASSERT( ret, "Getting reac value" );
	ASSERT( dret == 0.1, "Getting reac value" );

	/////////////////////////////////////////////////////////
	// Now move on to handling pre-existing messages.
	/////////////////////////////////////////////////////////

	cout << "\nTesting Stoich external message redirection" << flush;
	// Clean out the old stuff
	set( hub, "destroy" );
	set( stoich, "destroy" );

	Element* table = Neutral::create( "Table", "table", Element::root()->id(),
		Id::scratchId() );
	ret = Eref( table ).add( "outputSrc", m[5], "sumTotal" );
	// ret = table->findFinfo( "outputSrc" )->add( table, m[5], m[5]->findFinfo( "sumTotal" ) );
	ASSERT( ret, "Making test message" );

	stoich = Neutral::create( "Stoich", "s", Element::root()->id(),
		Id::scratchId() );
	s = static_cast< Stoich* >( stoich->data() );

	hub = Neutral::create( "KineticHub", "hub", Element::root()->id(),
		Id::scratchId() );
	ret = Eref( stoich ).add( "hub", hub, "hub" );
	// ret = stoich->findFinfo( "hub" )->add( stoich, hub, hub->findFinfo( "hub" ) );
	ASSERT( ret, "connecting stoich to hub" );

	// Rebuild the path now that the hub is connected.
	ret = set< string >( stoich, "path", "/n/##" );
	ASSERT( ret, "Setting path" );

	set< int >( table, "stepmode", 0 );
	set< int >( table, "xdivs", 1 );
	set< double >( table, "xmin", 0.0 );
	set< double >( table, "xmax", 10.0 );
	lookupSet< double, unsigned int >( table, "table", 33.0, 0 );
	lookupSet< double, unsigned int >( table, "table", 33.0, 1 );
	set< int >( table, "stepmode", 0 );
	SetConn c( table, 0 );
	ProcInfoBase p;
	p.dt_ = 0.001;
	p.currTime_ = 0.0;
	Table::process( &c, &p );
	// Check that the value has been added to the correct molecule
	ret = get< double >( m[5], "n", dret );
	ASSERT( ret, "Getting value" );
	ASSERT( dret == 33.0, "Message redirection" );
	k = s->molMap_.find( m[5] );
	molNum = k->second;
	ASSERT( s->S_[molNum] == 33.0, "Message redirection" );


	/////////////////////////////////////////////////////////
	// Now test handling of stuff on DynamicFinfos.
	/////////////////////////////////////////////////////////

	cout << "\nTesting Stoich DynamicFinfo msg redirection" << flush;
	// Clean out the old stuff
	set( hub, "destroy" );
	set( stoich, "destroy" );
	set( table, "destroy" );

	table = Neutral::create( "Table", "table", Element::root()->id(),
		Id::scratchId() );
	set< int >( table, "stepmode", 3 );
	set< int >( table, "xdivs", 1 );
	set< double >( table, "xmin", 0.0 );
	set< double >( table, "xmax", 10.0 );
	set< double >( table, "output", 0.0 );
	SetConn c1( table, 0 );
	p.dt_ = 0.001;
	p.currTime_ = 0.0;

	ret = Eref( table ).add( "inputRequest", m[4], "n" );
	// ret = table->findFinfo( "inputRequest" )->add( table, m[4], m[4]->findFinfo( "n" ) );
	ASSERT( ret, "Making test message" );

	stoich = Neutral::create( "Stoich", "s", Element::root()->id(),
		Id::scratchId() );
	s = static_cast< Stoich* >( stoich->data() );

	hub = Neutral::create( "KineticHub", "hub", Element::root()->id(),
		Id::scratchId() );
	ret = Eref( stoich ).add( "hub", hub, "hub" );
	// ret = stoich->findFinfo( "hub" )->add( stoich, hub, hub->findFinfo( "hub" ) );
	ASSERT( ret, "connecting stoich to hub" );

	// Rebuild the path now that the hub is connected.
	ret = set< string >( stoich, "path", "/n/##" );
	ASSERT( ret, "Setting path" );


	// Confirm that the field access still works
	k = s->molMap_.find( m[6] );
	molNum = k->second;
	s->S_[molNum] = 346434.0;
	ret = get< double >( m[6], "n", dret );
	ASSERT( ret, "Getting value" );
	ASSERT( dret == 346434.0, "Getting value" );

	// This check turns out to be quite important. Confirm that
	// field access works even through a dynamic Finfo.
	k = s->molMap_.find( m[4] );
	molNum = k->second;
	s->S_[molNum] = 102030.5;

	ret = get< double >( m[4], "n", dret );
	ASSERT( ret, "DynamicFinfo message redirect" );
	ASSERT( dret == 102030.5, "DynamicFinfo message redirect" );

	// Here we finally check that the return message to the
	// DynamicFinfo can look up the solver.
	s->S_[molNum] = 192939.5;
	Table::process( &c1, &p );
	ret = get< double >( table, "input", dret );
	ASSERT( ret, "DynamicFinfo message redirect" );
	ASSERT( dret == 192939.5, "DynamicFinfo message redirect" );
	ret = lookupGet< double, unsigned int >( table, "table", dret, 0 );
	ASSERT( ret, "DynamicFinfo message redirect" );
	ASSERT( dret == 192939.5, "DynamicFinfo message redirect" );

	// Here we check if a brand new DynamicFinfo automagically finds the
	// solver.
	Element* table2 = Neutral::create( "Table", "table2", table->id(),
		Id::scratchId() );
	set< int >( table2, "stepmode", 3 );
	set< int >( table2, "xdivs", 1 );
	set< double >( table2, "xmin", 0.0 );
	set< double >( table2, "xmax", 10.0 );
	set< double >( table2, "output", 0.0 );

	ret = Eref( table2 ).add( "inputRequest", m[8], "n" );
	// ret = table2->findFinfo( "inputRequest" )->add( table2, m[8], m[8]->findFinfo( "n" ) );
	ASSERT( ret, "Making test message" );

	SetConn c2( table2, 0 );

	k = s->molMap_.find( m[8] );
	molNum = k->second;
	s->S_[molNum] = 12.5;

	ret = get< double >( table2, "input", dret );
	ASSERT( ret, "new DynamicFinfo message redirect" );
	ASSERT( dret == 0, "New DynamicFinfo message redirect" );
	ret = lookupGet< double, unsigned int >( table2, "table", dret, 0 );
	ASSERT( ret, "New DynamicFinfo message redirect" );
	ASSERT( dret == 0, "New DynamicFinfo message redirect" );

	Table::process( &c2, &p );

	ret = get< double >( table2, "input", dret );
	ASSERT( ret, "new DynamicFinfo message redirect" );
	ASSERT( dret == 12.5, "New DynamicFinfo message redirect" );
	ret = lookupGet< double, unsigned int >( table2, "table", dret, 0 );
	ASSERT( ret, "New DynamicFinfo message redirect" );
	ASSERT( dret == 12.5, "New DynamicFinfo message redirect" );

	/////////////////////////////////////////////////////////
	// Get rid of all the compartments.
	/////////////////////////////////////////////////////////
	set( table, "destroy" );
	set( hub, "destroy" );
	set( stoich, "destroy" );
	set( n, "destroy" );
}


//////////////////////////////////////////////////////////////////
// Here we set up a small reaction system for testing with the
// solver.
//////////////////////////////////////////////////////////////////

static const double totalMols = 1.0;
#include "Kintegrator.h"
void testKintegrator()
{
	static const double EPSILON = 1.0e-4;
	cout << "\nTesting Kintegrator" << flush;
	const unsigned int NUM_COMPT = 21;
	const double RUNTIME = 500.0;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(),
		Id::scratchId() );
	vector< Element* >m;
	vector< Element* >r;
	char name[10];
	bool ret;
	const Cinfo* molCinfo = Cinfo::find( "Molecule" );
	assert( molCinfo != 0 );
	const Finfo* rFinfo = molCinfo->findFinfo( "reac" );
	assert( rFinfo != 0 );

	const Cinfo* reacCinfo = Cinfo::find( "Reaction" );
	assert( reacCinfo != 0 );
	const Finfo* sFinfo = reacCinfo->findFinfo( "sub" );
	assert( sFinfo != 0 );
	const Finfo* pFinfo = reacCinfo->findFinfo( "prd" );
	assert( pFinfo != 0 );
	//////////////////////////////////////////////////////////////////
	// Create a linear sequence of molecules with reactions between.
	//////////////////////////////////////////////////////////////////
	for ( unsigned int i = 0; i < NUM_COMPT; i++ ) {
		sprintf( name, "m%d", i );
		Element* mtemp = Neutral::create( "Molecule", name, n->id(),
			Id::scratchId() );
		assert( mtemp != 0 );
		set< double >( mtemp, "nInit", 0.0 );
		set< double >( mtemp, "n", 0.0 );
		set< int >( mtemp, "mode", 0 );
		m.push_back( mtemp );

		if ( i > 0 ) {
			sprintf( name, "r%d", i );
			Element* rtemp = 
				Neutral::create( "Reaction", name, n->id(),
					Id::scratchId() );
			assert( rtemp != 0 );
			set< double >( rtemp, "kf", 1 );
			set< double >( rtemp, "kb", 1 );
			r.push_back( rtemp );

			// ret = rFinfo->add( m[i - 1], rtemp, sFinfo );
			ret = Eref( m[i-1]).add( "reac", rtemp, "sub" );
			ASSERT( ret, "adding msg 0" );
			ret = Eref( mtemp ).add( "reac", rtemp, "prd" );
			// ret = rFinfo->add( mtemp, rtemp, pFinfo );
			ASSERT( ret, "adding msg 1" );
		}
	}
	// Buffer the end compartments to fixed values.
	set < int >( m[0], "mode", 4 );
	set < int >( m[ NUM_COMPT - 1 ], "mode", 4 );
	set< double >( m[0], "nInit", totalMols );
	set< double >( m[NUM_COMPT - 1], "nInit", 0.0 );

	///////////////////////////////////////////////////////////
	// Assign reaction system to a Stoich object
	///////////////////////////////////////////////////////////

	Element* stoich = Neutral::create( "Stoich", "s", Element::root()->id(),
		Id::scratchId() );
	Element* hub = Neutral::create( "KineticHub", "hub", Element::root()->id(),
		Id::scratchId() );
	Element* integ = Neutral::create( "Kintegrator", "integ", Element::root()->id(),
		Id::scratchId() );
	SetConn ci( integ, 0 );

	ret = Eref( stoich ).add( "hub", hub, "hub" );
	// ret = stoich->findFinfo( "hub" )->add( stoich, hub, hub->findFinfo( "hub" ) );
	ASSERT( ret, "connecting stoich to hub" );
	ret = Eref( stoich ).add( "integrate", integ, "integrate" );
	// ret = stoich->findFinfo( "integrate" )->add( stoich, integ, integ->findFinfo( "integrate" ) );
	ASSERT( ret, "connecting stoich to integ" );

	ret = set< string >( stoich, "path", "/n/##" );
	ASSERT( ret, "Setting path" );


	Element* table = Neutral::create( "Table", "table", Element::root()->id(),
		Id::scratchId() );

	ret = Eref( table ).add( "inputRequest", m[4], "n" );
	// ret = table->findFinfo( "inputRequest" )->add( table, m[4], m[4]->findFinfo( "n" ) );
	ASSERT( ret, "Making test message" );

	set< int >( table, "stepmode", 3 );
	set< int >( table, "xdivs", 1 );
	set< double >( table, "xmin", 0.0 );
	set< double >( table, "xmax", 10.0 );
	set< double >( table, "output", 0.0 );
	SetConn ct( table, 0 );
	ProcInfoBase p;
	p.dt_ = 0.05;
	p.currTime_ = 0.0;

	Kintegrator::reinitFunc( &ci, &p );

	for ( p.currTime_ = 0.0; p.currTime_ < RUNTIME; p.currTime_ += p.dt_ ) {
		Kintegrator::processFunc( &ci, &p );
		Table::process( &ct, &p );
	}
	double tot = 0.0;
	double dx = totalMols / (NUM_COMPT - 1);
	for ( unsigned int i = 0; i < NUM_COMPT; i++ ) {
		double val;
		get< double >( m[i], "n", val );
		tot += fabs( totalMols - ( val + dx * i ) );
//		cout << val << "\t" << tot << "\n";
	}
	// set< string >( table, "print", "kinteg.plot" );
	ASSERT ( tot < EPSILON, "Diffusion between source and sink by Kintegrator");

	// Note that the order of the species in the matrix is 
	// ill-defined because of the properties of the STL sort operation.
	// So here we need to look up the order based on the mol_map
	// Reac order has also been scrambled.
	
	// cout << s->N_;
	
	set( table, "destroy" );
	set( integ, "destroy" );
	set( hub, "destroy" );
	set( stoich, "destroy" );
	set( n, "destroy" );
}

#ifdef USE_GSL

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "GslIntegrator.h"
#ifndef WIN32
	#include <sys/time.h>
#endif

#include <time.h>

static const unsigned int NUM_COMPT = 21;
void doGslRun( const string& method, Element* integ, Element* stoich,
	const Conn* ct, vector< Element* >& m, double accuracy );

void testGslIntegrator()
{
#ifndef WIN32
	cout << "\nTesting GslIintegrator" << flush;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(),
		Id::scratchId() );
	vector< Element* >m;
	vector< Element* >r;
	char name[10];
	bool ret;
	const Cinfo* molCinfo = Cinfo::find( "Molecule" );
	assert( molCinfo != 0 );
	const Finfo* rFinfo = molCinfo->findFinfo( "reac" );
	assert( rFinfo != 0 );

	const Cinfo* reacCinfo = Cinfo::find( "Reaction" );
	assert( reacCinfo != 0 );
	const Finfo* sFinfo = reacCinfo->findFinfo( "sub" );
	assert( sFinfo != 0 );
	const Finfo* pFinfo = reacCinfo->findFinfo( "prd" );
	assert( pFinfo != 0 );
	//////////////////////////////////////////////////////////////////
	// Create a linear sequence of molecules with reactions between.
	//////////////////////////////////////////////////////////////////
	for ( unsigned int i = 0; i < NUM_COMPT; i++ ) {
		sprintf( name, "m%d", i );
		Element* mtemp = Neutral::create( "Molecule", name, n->id(),
			Id::scratchId() );
		assert( mtemp != 0 );
		set< double >( mtemp, "nInit", 0.0 );
		set< double >( mtemp, "n", 0.0 );
		set< int >( mtemp, "mode", 0 );
		m.push_back( mtemp );

		if ( i > 0 ) {
			sprintf( name, "r%d", i );
			Element* rtemp = 
				Neutral::create( "Reaction", name, n->id(),
					Id::scratchId() );
			assert( rtemp != 0 );
			set< double >( rtemp, "kf", 1 );
			set< double >( rtemp, "kb", 1 );
			r.push_back( rtemp );
			ret = Eref( m[i-1] ).add( "reac", rtemp, "sub" );
			// ret = rFinfo->add( m[i - 1], rtemp, sFinfo );
			ASSERT( ret, "adding msg 0" );
			ret = Eref( mtemp ).add( "reac", rtemp, "prd" );
			// ret = rFinfo->add( mtemp, rtemp, pFinfo );
			ASSERT( ret, "adding msg 1" );
		}
	}
	// Buffer the end compartments to fixed values.
	set < int >( m[0], "mode", 4 );
	set < int >( m[ NUM_COMPT - 1 ], "mode", 4 );
	set< double >( m[0], "nInit", totalMols );
	set< double >( m[NUM_COMPT - 1], "nInit", 0.0 );

	///////////////////////////////////////////////////////////
	// Assign reaction system to a Stoich object
	///////////////////////////////////////////////////////////

	Element* stoich = Neutral::create( "Stoich", "s", Element::root()->id(),
		Id::scratchId() );
	Element* hub = Neutral::create( "KineticHub", "hub", Element::root()->id(),
		Id::scratchId() );
	Element* integ = Neutral::create( "GslIntegrator", "integ", Element::root()->id(),
		Id::scratchId() );

	ret = Eref( stoich ).add( "hub", hub, "hub" );
	// ret = stoich->findFinfo( "hub" )->add( stoich, hub, hub->findFinfo( "hub" ) );
	ASSERT( ret, "connecting stoich to hub" );

	ret = Eref( stoich ).add( "gsl", integ, "gsl" );
	// ret = stoich->findFinfo( "gsl" )->add( stoich, integ, integ->findFinfo( "gsl" ) );
	ASSERT( ret, "connecting stoich to gsl integ" );

	ret = set< string >( stoich, "path", "/n/##" );
	ASSERT( ret, "Setting path" );


	Element* table = Neutral::create( "Table", "table", Element::root()->id(),
		Id::scratchId() );

	ret = Eref( table ).add( "inputRequest", m[4], "n" );
	// ret = table->findFinfo( "inputRequest" )->add( table, m[4], m[4]->findFinfo( "n" ) );
	ASSERT( ret, "Making test message" );

	set< int >( table, "stepmode", 3 );
	set< int >( table, "xdivs", 1 );
	set< double >( table, "xmin", 0.0 );
	set< double >( table, "xmax", 10.0 );
	set< double >( table, "output", 0.0 );
	SetConn ct( table, 0 );

	cout << "\n";
	struct timeval tv1;
	struct timeval tv2;
	gettimeofday( &tv1, 0 );
	    
	doGslRun( "rk2", integ, stoich, &ct, m, 1.0e-6 );
	doGslRun( "rk4", integ, stoich, &ct, m, 1.0e-4 );
	doGslRun( "rk5", integ, stoich, &ct, m, 1.0e-6 );
	doGslRun( "rkck", integ, stoich, &ct, m, 1.0e-8 );
	doGslRun( "rk8pd", integ, stoich, &ct, m, 1.0e-7 );
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!! BUG : in 64 bit machine if rk2imp is run before rk4imp  !!
        //!! the unit test fails with total error exceeding EPSILON  !!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	doGslRun( "rk4imp", integ, stoich, &ct, m, 1.0e-4 );                                                             
	doGslRun( "rk2imp", integ, stoich, &ct, m, 1.0e-6 );
	doGslRun( "gear1", integ, stoich, &ct, m, 1.0e-6 );
	doGslRun( "gear2", integ, stoich, &ct, m, 2.0e-4 );
	
	gettimeofday( &tv2, 0 );
	unsigned long time = tv2.tv_sec - tv1.tv_sec;
	time *= 1000000;
	time += tv2.tv_usec;
	time -= tv1.tv_usec;
	cout << "runtime (usec)= " << time << endl;
	
	set( table, "destroy" );
	set( integ, "destroy" );
	set( hub, "destroy" );
	set( stoich, "destroy" );
	set( n, "destroy" );
#endif
}

void doGslRun( const string& method, Element* integ, Element* stoich,
	const Conn* ct, vector< Element* >& m, double accuracy )
{
#ifndef WIN32
	double EPSILON = accuracy * 50.0;
	const double RUNTIME = 500.0;
	SetConn ci( integ, 0 );
	ProcInfoBase p;
	p.dt_ = 100.0; // Oddly, it isn't accurate given 500 sec to work with.
	p.currTime_ = 0.0;

	if ( method == "gear2") // This is for the dreadful Gear2.
		EPSILON = 0.01;
	if ( method == "rkck") // This is for the strange rkck, which does not
		EPSILON = 1.0e-4;	// take long but doesn't meet its accuracy
							// specs.
	if ( method == "rk8pd") // Another case of accuracy not up to spec.
		EPSILON = 1.0e-4;
	

	set< string >( integ, "method", method );
	set< double >( integ, "relativeAccuracy", accuracy );
	set< double >( integ, "absoluteAccuracy", accuracy );
	cout << method << "." << flush;
	GslIntegrator::reinitFunc( &ci, &p );

	for ( p.currTime_ = 0.0; p.currTime_ < RUNTIME; p.currTime_ += p.dt_ ) {
		GslIntegrator::processFunc( &ci, &p );
		Table::process( ct, &p );
	}
	double tot = 0.0;
	double dx = totalMols / (NUM_COMPT - 1);
	for ( unsigned int i = 0; i < NUM_COMPT; i++ ) {
		double val;
		get< double >( m[i], "n", val );
		tot += fabs( totalMols - ( val + dx * i ) );
                // cout << val << "\t" << tot << "\n";
	}
	// cout << "Err= " << tot << ",	accRequest= " << accuracy  << ",     ";
	// static_cast< Stoich* >( stoich->data() )->runStats();
	// set< string >( table, "print", "kinteg.plot" );
	ASSERT ( tot < EPSILON, "Diffusion between source and sink by GslIntegrator");
#endif
}

#endif // USE_GSL

#endif
