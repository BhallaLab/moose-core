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
	double spineDia = 10.0e-6; // Bigger for more effect.
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
	double comptDia = 2e-6;
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

void buildSigNeurChem( Id nid )
{
	const double diffLength = 1e-6;
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );

	Id neuroMesh = shell->doCreate( "NeuroMesh", nid, "neuroMesh", dims );
	Field< bool >::set( neuroMesh, "separateSpines", true );
	Field< double >::set( neuroMesh, "diffLength", diffLength );
	Field< string >::set( neuroMesh, "geometryPolicy", "cylinder" );
	Id spineMesh = shell->doCreate( "SpineMesh", nid, "spineMesh", dims );
	MsgId mid = shell->doAddMsg(
		"OneToOne", neuroMesh, "spineListOut", spineMesh, "spineList" );
	assert( mid != Msg::bad );
	Id psdMesh = shell->doCreate( "PsdMesh", Id(), "psdMesh", dims );
	mid = shell->doAddMsg(
		"OneToOne", spineMesh, "psdListOut", psdMesh, "psdList" );
	assert( mid != Msg::bad );

	///////////////////////////////////////////////////////////////////
	// Stuff in PSD
	///////////////////////////////////////////////////////////////////
	Id psdGluR = shell->doCreate( "Pool", psdMesh, "psdGluR" );
	Field< double >::set( psdGluR, "nInit", 100 );
	///////////////////////////////////////////////////////////////////
	// Stuff in spine head
	///////////////////////////////////////////////////////////////////

	Id headGluR = shell->doCreate( "Pool", spineMesh, "headGluR" );
	Field< double >::set( headGluR, "nInit", 100 ); // Add to 200
	Id toPsd = shell->doCreate( "Pool", spineMesh, "toPsd" );
	Field< double >::set( toPsd, "concInit", 1e-3 );
	Id toPsdEnz = shell->doCreate( "Enz", toPsd, "enz" );
	mid = shell->doAddMsg( "OneToOne", toPsdEnz, "enz", toPsd, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "OneToOne", toPsdEnz, "sub", headGluR, "reac" );
	assert( mid != Msg::bad );
	mid = shell->doAddMsg( "OneToOne", toPsdEnz, "prd", psdGluR, "reac" );
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

	///////////////////////////////////////////////////////////////////
	// Make NeuroMesh
	///////////////////////////////////////////////////////////////////
	Field< Id >::set( neuroMesh, "cell", nid );
	Qinfo::clearQ( 0 );
	Qinfo::clearQ( 0 );
}

void testSigNeurElec()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );

	vector< Id > spines;
	Id nid = buildSigNeurElec( spines );
	buildSigNeurChem( nid );
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
}



#endif
