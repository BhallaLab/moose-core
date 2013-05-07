/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include "header.h"
#include "Adaptor.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"

void testAdaptor()
{
	Adaptor foo;
	foo.setInputOffset( 1 );
	foo.setOutputOffset( 2 );
	foo.setScale( 10 );

	for ( unsigned int i = 0; i < 10; ++i )
		foo.input( i );


	assert( doubleEq( foo.getOutput(), 0.0 ) );
	foo.innerProcess();

	assert( doubleEq( foo.getOutput(), ( -1.0 + 4.5) * 10.0 + 2.0 ) );

	// shell->doDelete( nid );
	cout << "." << flush;
}


//////////////////////////////////////////////////////////////////////
// Test of multiscale model setup. Model structure is as follows:
//
// Elec:
// Single elec compt soma with HH channels. 5 spines on it. GluR on spines.
// 		Ca pool in spine head fed by GluR.
// Chem:
// 		PSD: GluR pool.
//    	Head: Ca pool. GluR pool. 'Enz' to take to PSD. 
//    		Reac balances with PSD.
// 		Dend: Ca pool binds to 'kinase'. Kinase phosph K chan. 
// 			Dephosph by reac.
//			Ca is pumped out into a buffered non-reactive pool
//
// 		Diffusion:
// 			Ca from spine head goes to dend
//
// Adaptors:
// 		Electrical Ca in spine head -> Ca pool in spine head
// 		Chem GluR in PSD -> Electrical GluR Gbar
// 		Chem K in soma -> Electrical K Gbar.
//
//////////////////////////////////////////////////////////////////////

// Defined in testBiophysics.cpp
Id makeSquid();
// Defined in testMesh.cpp
Id makeSpine( Id compt, Id cell, unsigned int index, double frac, 
			double len, double dia, double theta );

Id makeSpineWithReceptor( Id compt, Id cell, unsigned int index, 
				double frac )
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );
	double spineLength = 5.0e-6;
	double spineDia = 4.0e-6;
	Id spineCompt = makeSpine( compt, cell, index, frac, 
					spineLength, spineDia, 0.0 );

	Id gluR = shell->doCreate( "SynChan", spineCompt, "gluR", dims );
	Field< double >::set( gluR, "tau1", 1e-3 );
	Field< double >::set( gluR, "tau2", 1e-3 );
	Field< double >::set( gluR, "Gbar", 1e-5 );
	Field< double >::set( gluR, "Ek", 10.0e-3 );
	MsgId mid = shell->doAddMsg( "Single", ObjId( spineCompt ), "channel",
					ObjId( gluR ), "channel" );
	assert( mid != Msg::bad );


	Id caPool = shell->doCreate( "CaConc", spineCompt, "ca", dims );
	Field< double >::set( caPool, "CaBasal", 1e-4 ); // millimolar
	Field< double >::set( caPool, "tau", 0.01 ); // seconds
	double B = 1.0 / ( 
		FaradayConst *  spineLength * spineDia * spineDia * 0.25 * PI );
	B = B / 20.0;
	Field< double >::set( caPool, "B", B ); // Convert from current to conc
	mid = shell->doAddMsg( "Single", ObjId( gluR ), "IkOut",
					ObjId( caPool ), "current" );
	assert( mid != Msg::bad );

	return spineCompt;
}

Id buildSigNeurElec( vector< Id >& spines )
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );
	double comptLength = 100e-6;
	double comptDia = 4e-6;
	unsigned int numSpines = 5;

	Id nid = makeSquid();
	Id compt( "/n/compt" );
	Field< double >::set( compt, "inject", 0.0 );
	Field< double >::set( compt, "x0", 0 );
	Field< double >::set( compt, "y0", 0 );
	Field< double >::set( compt, "z0", 0 );
	Field< double >::set( compt, "x", comptLength );
	Field< double >::set( compt, "y", 0 );
	Field< double >::set( compt, "z", 0 );
	Field< double >::set( compt, "length", comptLength );
	Field< double >::set( compt, "diameter", comptDia );

	// Make a SpikeGen as a synaptic input to the spines.
	Id synInput = shell->doCreate( "SpikeGen", nid, "synInput", dims );
	Field< double >::set( synInput, "refractT", 17e-3 );
	Field< double >::set( synInput, "threshold", -1.0 );
	Field< bool >::set( synInput, "edgeTriggered", false );
	SetGet1< double >::set( synInput, "Vm", 0.0 );

	spines.resize( numSpines );

	for ( unsigned int i = 0; i < numSpines; ++i ) {
		double frac = ( 0.5 + i ) / numSpines;
		spines[i] = makeSpineWithReceptor( compt, nid, i, frac );
		stringstream ss;
		ss << "/n/head" << i << "/gluR";
		string name = ss.str();
		Id gluR(  name );
		assert( gluR != Id() );
		Field< unsigned int >::set( gluR, "num_synapse", 1 );
		Id syn( name + "/synapse" );
		assert( syn != Id() );
		MsgId mid = shell->doAddMsg( "Single", ObjId( synInput ), "event",
					ObjId( syn ), "addSpike" );
		assert( mid != Msg::bad );
		Field< double >::set( syn, "weight", 1 );
		Field< double >::set( syn, "delay", i * 1.0e-3 );
	}

	return nid;
}


void buildSigNeurChem( Id nid, Id neuroMesh, Id spineMesh, Id psdMesh )
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );

	Id spineMeshEntries( "/n/spineMesh/mesh" );
	///////////////////////////////////////////////////////////////////
	// Stuff in PSD
	///////////////////////////////////////////////////////////////////
	Id psdMeshEntries( "/n/psdMesh/mesh" );
	assert( psdMeshEntries != Id() );
	Id psdGluR = shell->doCreate( "Pool", psdMesh, "psdGluR" );
	Field< double >::set( psdGluR, "nInit", 100 );
	MsgId mid = shell->doAddMsg( "OneToOne", psdGluR, "mesh", psdMeshEntries, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", psdMeshEntries, "mesh", psdGluR, "mesh" );
	assert( mid != Msg::bad );
	///////////////////////////////////////////////////////////////////
	// Stuff in spine head
	///////////////////////////////////////////////////////////////////

	Id headGluR = shell->doCreate( "Pool", spineMesh, "headGluR" );
	Field< double >::set( headGluR, "nInit", 100 ); // Add to 200
	Id toPsd = shell->doCreate( "Pool", spineMesh, "toPsd" );
	Field< double >::set( toPsd, "concInit", 1e-3 );
	Id toPsdEnz = shell->doCreate( "Enz", toPsd, "enz" );
	Id toPsdEnzCplx = shell->doCreate( "Pool", toPsdEnz, "cplx" );
	mid = shell->doAddMsg( "OneToOne", toPsdEnz, "enz", toPsd, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "OneToOne", toPsdEnz, "sub", headGluR, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "OneToOne", toPsdEnz, "prd", psdGluR, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "OneToOne", toPsdEnz, "cplx", toPsdEnzCplx, "reac" );
	assert( mid != Msg::bad );
	Field< double >::set( toPsdEnz, "Km", 1e-3 ); 	// 1 uM
	Field< double >::set( toPsdEnz, "kcat", 1 );	// 1/sec.

	Id fromPsd = shell->doCreate( "Reac", spineMesh, "fromPsd" );
	Id headCa = shell->doCreate( "Pool", spineMesh, "Ca" );
	Field< double >::set( headCa, "concInit", 1e-4 );

	mid = shell->doAddMsg( "OneToOne", fromPsd, "sub", psdGluR, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "OneToOne", fromPsd, "prd", headGluR, "reac" );
	assert( mid != Msg::bad );
	Field< double >::set( fromPsd, "Kf", 0.02 );
	Field< double >::set( fromPsd, "Kb", 0.0 );

	mid = shell->doAddMsg( "Single", spineMeshEntries, "mesh", headGluR, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", spineMeshEntries, "mesh", toPsd, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", spineMeshEntries, "mesh", headCa, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", spineMeshEntries, "mesh", toPsd, "mesh" );
	assert( mid != Msg::bad );
	// Id toPsdEnzCplx( toPsdEnz.value() + 1 );
	Id foo( "/n/spineMesh/toPsd/enz/cplx" );
	assert( toPsdEnzCplx != Id() );
	assert( toPsdEnzCplx == foo );
	shell->doAddMsg( "Single", spineMeshEntries, "mesh", toPsdEnzCplx, "mesh" );

	///////////////////////////////////////////////////////////////////
	// Stuff in dendrite
	///////////////////////////////////////////////////////////////////

	Id dendCa = shell->doCreate( "Pool", neuroMesh, "Ca" );
	Field< double >::set( dendCa, "concInit", 1e-4 ); // 0.1 uM.
	Id bufCa = shell->doCreate( "BufPool", neuroMesh, "bufCa" );
	Field< double >::set( bufCa, "concInit", 1e-4 ); // 0.1 uM.
	Id pumpCa = shell->doCreate( "Reac", neuroMesh, "pumpCa" );
	mid = shell->doAddMsg( "OneToOne", pumpCa, "sub", dendCa, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "OneToOne", pumpCa, "prd", bufCa, "reac" );
	assert( mid != Msg::bad );
	Field< double >::set( pumpCa, "Kf", 0.1 );
	Field< double >::set( pumpCa, "Kb", 0.1 );

	Id dendKinase = shell->doCreate( "Pool", neuroMesh, "kinase" );
	Field< double >::set( dendKinase, "concInit", 1e-3 ); // 1 uM.
	Id dendKinaseEnz = shell->doCreate( "Enz", dendKinase, "enz" );
	Id dendKinaseEnzCplx = shell->doCreate( "Pool", dendKinaseEnz, "cplx" );
	Id kChan = shell->doCreate( "Pool", neuroMesh, "kChan");
	Field< double >::set( kChan, "concInit", 1e-3 ); // 1 uM.
	Id kChan_p = shell->doCreate( "Pool", neuroMesh, "kChan_p");
	Field< double >::set( kChan_p, "concInit", 0 ); // 0 uM.
	mid = shell->doAddMsg( 
					"OneToOne", dendKinaseEnz, "enz", dendKinase, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( 
					"OneToOne", dendKinaseEnz, "sub", kChan, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( 
					"OneToOne", dendKinaseEnz, "prd", kChan_p, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( 
			"OneToOne", dendKinaseEnz, "cplx", dendKinaseEnzCplx, "reac" );
	assert( mid != Msg::bad );
	Field< double >::set( dendKinaseEnz, "Km", 1e-3 ); 	// 1 uM
	Field< double >::set( dendKinaseEnz, "kcat", 1 );	// 1/sec.

	Id dendPhosphatase = shell->doCreate( "Reac", neuroMesh, "phosphatase");
	mid = shell->doAddMsg( 
					"OneToOne", dendPhosphatase, "sub", kChan_p, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( 
					"OneToOne", dendPhosphatase, "prd", kChan, "reac" );
	assert( mid != Msg::bad );
	Field< double >::set( dendPhosphatase, "Kf", 0.02 );
	Field< double >::set( dendPhosphatase, "Kb", 0.0 );

	Id dendMeshEntries( "/n/neuroMesh/mesh" );

	mid = shell->doAddMsg( "Single", dendMeshEntries, "mesh", dendCa, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", dendMeshEntries, "mesh", bufCa, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", dendMeshEntries, "mesh", dendKinase, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", dendMeshEntries, "mesh", dendKinaseEnzCplx, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", dendMeshEntries, "mesh", kChan, "mesh" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "Single", dendMeshEntries, "mesh", kChan_p, "mesh" );
	assert( mid != Msg::bad );

	Id bar( "/n/neuroMesh/kinase/enz/cplx" );
	assert( dendKinaseEnzCplx != Id() );
	assert( dendKinaseEnzCplx == bar );
	shell->doAddMsg( "Single", spineMeshEntries, "mesh", dendKinaseEnzCplx, "mesh" );
}

void buildSigNeurNeuroMesh( Id nid, Id& neuroMesh, Id& spineMesh, Id& psdMesh )
{
	const double diffLength = 1e-6;
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );

	neuroMesh = shell->doCreate( "NeuroMesh", nid, "neuroMesh", dims );
	Field< bool >::set( neuroMesh, "separateSpines", true );
	Field< double >::set( neuroMesh, "diffLength", diffLength );
	Field< string >::set( neuroMesh, "geometryPolicy", "cylinder" );
	spineMesh = shell->doCreate( "SpineMesh", nid, "spineMesh", dims );
	MsgId mid;
	mid = shell->doAddMsg( "OneToOne", neuroMesh, "spineListOut", spineMesh, "spineList" );
	assert( mid != Msg::bad );
	psdMesh = shell->doCreate( "PsdMesh", nid, "psdMesh", dims );
	mid = shell->doAddMsg( "OneToOne", neuroMesh, "psdListOut", psdMesh, "psdList" );
	assert( mid != Msg::bad );
}

void makeChemInNeuroMesh()
{
	double spineLength = 5.0e-6;
	double spineDia = 4.0e-6;
	double comptLength = 100.0e-6;
	double comptDia = 4.0e-6;
	vector< Id > spines;
	Id nid = buildSigNeurElec( spines );
	Id neuroMesh, spineMesh, psdMesh;
	buildSigNeurNeuroMesh( nid, neuroMesh, spineMesh, psdMesh );
	buildSigNeurChem( nid, neuroMesh, spineMesh, psdMesh );
	///////////////////////////////////////////////////////////////////
	// Make NeuroMesh
	///////////////////////////////////////////////////////////////////
	Field< Id >::set( neuroMesh, "cell", nid );
	// shell->doReinit();
	// Qinfo::clearQ( 0 );
	// Qinfo::clearQ( 0 );
	// Deprecated. Qinfo::waitProcCycles( 4 );

	///////////////////////////////////////////////////////////////////
	// Check that stuff has been built
	///////////////////////////////////////////////////////////////////
	const unsigned int numComptsInDend = 100;
	const unsigned int numSpines = 5;
	unsigned int size;
	
	// 25 Apr 2013: this doesn't work, though it should. Need to clean up.
	//size = psdMeshEntries.element()->dataHandler()->totalEntries();
	
	Id psdMeshEntries( "/n/psdMesh/mesh" );
	size = Id( "/n/spineMesh/mesh" ).element()->dataHandler()->localEntries();
	assert( size == numSpines );
	size = Id( "/n/psdMesh/mesh" ).element()->dataHandler()->localEntries();
	assert( size == numSpines );

	size = Id( "/n/spineMesh/headGluR" ).element()->dataHandler()->localEntries();
	// size = Field< unsigned int >::get( headGluR, "linearSize");
	assert( size == numSpines );
	double vol;
	vol = Field< double >::get( Id( "/n/spineMesh/headGluR" ), "size" );
	assert( doubleEq( vol, spineLength * spineDia * spineDia * PI / 4.0 ) );

	size = Id( "/n/spineMesh/toPsd" ).element()->dataHandler()->localEntries();
	// size = Field< unsigned int >::get( toPsdEnz, "linearSize");
	assert( size == numSpines );
	size = Id( "/n/spineMesh/fromPsd" ).element()->dataHandler()->localEntries();
	// size = Field< unsigned int >::get( fromPsd, "linearSize");
	assert( size == 1 ); // It is a reac.

	size = Id( "/n/psdMesh/psdGluR" ).element()->dataHandler()->localEntries();
	// size = Field< unsigned int >::get( psdGluR, "linearSize");
	assert( size == numSpines );
	vol = Field< double >::get( Id( "/n/psdMesh/psdGluR" ), "size" );
	assert( doubleEq( vol, spineDia * spineDia * PI / 4.0 ) );

	size = Id( "/n/neuroMesh/Ca" ).element()->dataHandler()->localEntries();
	// size = Field< unsigned int >::get( dendCa, "linearSize");
	assert( size == numComptsInDend );
	vol = Field< double >::get( Id( "/n/neuroMesh/Ca" ), "size" );
	assert( doubleEq( vol, comptLength * comptDia * comptDia * PI / (4.0 * numComptsInDend ) ) );

	size = Id( "/n/neuroMesh/pumpCa" ).element()->dataHandler()->localEntries();
	// size = Field< unsigned int >::get( pumpCa, "linearSize");
	assert( size == 1 );
	size = Id( "/n/neuroMesh/dendKinaseEnz" ).element()->dataHandler()->localEntries();
	// size = Field< unsigned int >::get( dendKinaseEnz, "linearSize");
	assert( size == 1 );


	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	shell->doDelete( nid );
	cout << "." << flush;
} 

void makeChemInCubeMesh()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );
	double dendSide = 10.8e-6;	// Matches vol cylinder of 100x4 microns.
	double spineSide = 6.8e-6;	// Matches vol of 5 spines of 5x4 microns.
	double psdSide = 4.0e-6;	// Matches 'vol' of 5 psds of 4 microns 
			// diameter. Assume thickness of 1 micron, since are not 
			// comparable otherwise.

	Id nid = shell->doCreate( "Neutral", Id(), "n" );
	Id neuroMesh = shell->doCreate( "CubeMesh", nid, "neuroMesh" );
	Id spineMesh = shell->doCreate( "CubeMesh", nid, "spineMesh" );
	Id psdMesh = shell->doCreate( "CubeMesh", nid, "psdMesh" );

	vector< double > coords( 9, dendSide );
	coords[0] = 0.0;
	coords[1] = 0.0;
	coords[2] = 0.0;
	Field< vector< double > >::set( neuroMesh, "coords", coords );
	Field< bool >::set( neuroMesh, "preserveNumEntries", true );

	coords.clear();
	coords.resize( 9, spineSide );
	coords[0] = dendSide;
	coords[1] = 0.0;
	coords[2] = 0.0;
	coords[3] = spineSide + dendSide;
	Field< vector< double > >::set( spineMesh, "coords", coords );
	Field< bool >::set( spineMesh, "preserveNumEntries", true );

	coords.clear();
	coords.resize( 9, psdSide );
	coords[0] = dendSide + spineSide;
	coords[1] = 0.0;
	coords[2] = 0.0;
	coords[3] = psdSide + spineSide + dendSide;
	Field< vector< double > >::set( psdMesh, "coords", coords );
	Field< bool >::set( psdMesh, "preserveNumEntries", true );

	buildSigNeurChem( nid, neuroMesh, spineMesh, psdMesh );

	///////////////////////////////////////////////////////////////////
	// Check that stuff has been built
	///////////////////////////////////////////////////////////////////
	unsigned int size;
	
	Id psdMeshEntries( "/n/psdMesh/mesh" );
	size = Id( "/n/spineMesh/mesh" ).element()->dataHandler()->localEntries();
	assert( size == 1 );
	size = Id( "/n/psdMesh/mesh" ).element()->dataHandler()->localEntries();
	assert( size == 1 );

	size = Id( "/n/spineMesh/headGluR" ).element()->dataHandler()->localEntries();
	assert( size == 1 );
	double vol;
	vol = Field< double >::get( Id( "/n/spineMesh/headGluR" ), "size" );
	assert( doubleEq( vol, pow( spineSide, 3.0 ) ) );

	size = Id( "/n/spineMesh/toPsd" ).element()->dataHandler()->localEntries();
	assert( size == 1 );
	size = Id( "/n/spineMesh/fromPsd" ).element()->dataHandler()->localEntries();
	assert( size == 1 ); // It is a reac.

	size = Id( "/n/psdMesh/psdGluR" ).element()->dataHandler()->localEntries();
	assert( size == 1 );
	vol = Field< double >::get( Id( "/n/psdMesh/psdGluR" ), "size" );
	assert( doubleEq( vol, pow( psdSide, 3.0 ) ) );

	size = Id( "/n/neuroMesh/Ca" ).element()->dataHandler()->localEntries();
	assert( size == 1 );
	vol = Field< double >::get( Id( "/n/neuroMesh/Ca" ), "size" );
	assert( doubleEq( vol, pow( dendSide, 3.0 ) ) );

	size = Id( "/n/neuroMesh/pumpCa" ).element()->dataHandler()->localEntries();
	assert( size == 1 );
	size = Id( "/n/neuroMesh/dendKinaseEnz" ).element()->dataHandler()->localEntries();
	assert( size == 1 );

	shell->doDelete( nid );
	cout << "." << flush;
}

void testSigNeurElec()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );

	vector< Id > spines;
	Id nid = buildSigNeurElec( spines );
	Id compt( "/n/compt" );
	//////////////////////////////////////////////////////////////////////
	// Graph
	//////////////////////////////////////////////////////////////////////
	/*
	vector< Id > ret;
	simpleWildcardFind( "/n/##[ISA=Compartment]", ret );
	for ( vector< Id >::iterator i = ret.begin(); i != ret.end(); ++i ) {
		double Ra = Field< double >::get( *i, "Ra" );
		double Rm = Field< double >::get( *i, "Rm" );
		double Em = Field< double >::get( *i, "Em" );
		double Cm = Field< double >::get( *i, "Cm" );
		string name = i->element()->getName();
		cout << name << ": Ra = " << Ra << ", Rm = " << Rm << 
				", Cm = " << Cm << ", Em = " << Em << endl;
	}
	*/
	Id tab = shell->doCreate( "Table", nid, "tab", dims );
	MsgId mid = shell->doAddMsg( "Single", ObjId( tab, 0 ),
		"requestData", ObjId( spines[2], 0 ), "get_Vm" );
	assert( mid != Msg::bad );
	Id tab2 = shell->doCreate( "Table", nid, "tab2", dims );
	mid = shell->doAddMsg( "Single", ObjId( tab2, 0 ),
		"requestData", ObjId( compt, 0 ), "get_Vm" );
	assert( mid != Msg::bad );

	Id ca2( "/n/head2/ca" );
	Id tabCa = shell->doCreate( "Table", nid, "tab3", dims );
	mid = shell->doAddMsg( "Single", ObjId( tabCa, 0 ),
		"requestData", ObjId( ca2, 0 ), "get_Ca" );
	assert( mid != Msg::bad );

	//////////////////////////////////////////////////////////////////////
	// Schedule, Reset, and run.
	//////////////////////////////////////////////////////////////////////

	shell->doSetClock( 0, 1.0e-5 );
	shell->doSetClock( 1, 1.0e-5 );
	shell->doSetClock( 2, 1.0e-5 );
	shell->doSetClock( 3, 1.0e-4 );

	shell->doUseClock( "/n/compt,/n/shaft#,/n/head#", "init", 0 );
	shell->doUseClock( "/n/compt,/n/shaft#,/n/head#", "process", 1 );
	shell->doUseClock( "/n/synInput", "process", 1 );
	shell->doUseClock( "/n/compt/Na,/n/compt/K", "process", 2 );
	shell->doUseClock( "/n/head#/#", "process", 2 );
	shell->doUseClock( "/n/tab#", "process", 3 );

	shell->doReinit();
	shell->doReinit();
	shell->doStart( 0.1 );

	SetGet2< string, string >::set( tab, "xplot", "SigNeur.plot", "spineVm" );
	SetGet2< string, string >::set( tab2, "xplot", "SigNeur.plot", "comptVm" );
	SetGet2< string, string >::set( tabCa, "xplot", "SigNeur.plot", "headCa" );

	shell->doDelete( nid );

	cout << "." << flush;
}

// This tests stuff without using the messaging.
void testSigNeur()
{
	testAdaptor();
}

// This is applicable to tests that use the messaging and scheduling.
void testSigNeurProcess()
{
	testSigNeurElec();
	makeChemInNeuroMesh();
	makeChemInCubeMesh();
}



#endif
