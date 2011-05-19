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
#include "Compartment.h"
#include "HHGate.h"
#include "HHChannel.h"
#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"

extern void testCompartment(); // Defined in Compartment.cpp
extern void testCompartmentProcess(); // Defined in Compartment.cpp
/*
extern void testHHChannel(); // Defined in HHChannel.cpp
extern void testCaConc(); // Defined in CaConc.cpp
extern void testNernst(); // Defined in Nernst.cpp
extern void testSpikeGen(); // Defined in SpikeGen.cpp
extern void testSynChan(); // Defined in SynChan.cpp
extern void testBioScan(); // Defined in BioScan.cpp
*/

void testHHGateCreation()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	// cout << "\nTesting HHChannel";
	vector< unsigned int > dims( 1, 1 );
	Id nid = shell->doCreate( "Neutral", Id(), "n", dims );
	Id comptId = shell->doCreate( "Compartment", nid, "compt", dims );
	Id chanId = shell->doCreate( "HHChannel", nid, "Na", dims );
	MsgId mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
		ObjId( chanId ), "channel" );
	assert( mid != Msg::badMsg );
	
	// Check gate construction and removal when powers are assigned
	vector< Id > kids;
	HHChannel* chan = reinterpret_cast< HHChannel* >(chanId.eref().data());
	assert( chan->xGate_ == 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	assert( kids[0] == Id( chanId.value() + 1 ) );
	assert( kids[1] == Id( chanId.value() + 2 ) );
	assert( kids[2] == Id( chanId.value() + 3 ) );
	assert( kids[0]()->dataHandler()->localEntries() == 0 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Xpower", 1 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );
	// Read the size of the gateIds.

	Field< double >::set( chanId, "Xpower", 2 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Xpower", 0 );
	assert( chan->xGate_ == 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	// Even though gate was deleted, its Id is intact.
	assert( kids[0] == Id( chanId.value() + 1 ) );
	assert( kids[0]()->dataHandler()->localEntries() == 0 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Xpower", 2 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	// New gate was created but has orig Id
	assert( kids[0] == Id( chanId.value() + 1 ) );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Ypower", 3 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ != 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 1 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Zpower", 1 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ != 0 );
	assert( chan->zGate_ != 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	assert( kids[0] == Id( chanId.value() + 1 ) );
	assert( kids[1] == Id( chanId.value() + 2 ) );
	assert( kids[2] == Id( chanId.value() + 3 ) );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 1 );
	assert( kids[2]()->dataHandler()->localEntries() == 1 );

	shell->doDelete( nid );
	cout << "." << flush;
}

// This tests stuff without using the messaging.
void testBiophysics()
{
	testCompartment();
	testHHGateCreation();
	/*
	testHHChannel();
	testCaConc();
	testNernst();
	testSpikeGen();
	testSynChan();
	testBioScan();
	*/
}

// This is applicable to tests that use the messaging and scheduling.
void testBiophysicsProcess()
{
	testCompartmentProcess();
}

#endif
