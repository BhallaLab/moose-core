/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "moose.h"
#include "stdlib.h"
#include "DynamicFinfo.h"
#include "LookupFinfo.h"

/**
 * For now we just require that the name be followed immediately
 * by square braces, and that there is a single index in there.
 * Later may consider wildcards.
 */
const Finfo* LookupFinfo::match( Element* e, const string& s ) const
{
	if ( s == name() )
		return this;
	// Check that the base name matches.
	std::string::size_type openpos = name().length();
	if ( s.substr( 0, openpos ) != name() )
		return 0;

	if ( s.length() < name().length() + 3 ) {
		cout << "Error: LookupFinfo::match: bad indexing: " <<
				s << endl;
		return 0;
	}

	std::string::size_type closepos = s.length() - 1;

	if ( s[openpos] == '[' && s[closepos] == ']' ) {
		string indexStr = 
			s.substr( openpos + 1, closepos - openpos - 1 );

		void* index = ftype()->strToIndexPtr( indexStr );
		string n = name() + "[" + indexStr + "]";
		DynamicFinfo* ret = 
				DynamicFinfo::setupDynamicFinfo(
					e, n, this, 
					get_,
					index
				);
		return ret;
	}
	return 0;
}

/**
* Dynamic Finfo should handle it.
*/
bool LookupFinfo::add( 
	Eref e, Eref destElm, const Finfo* destFinfo,
	unsigned int connTainerOption 
	) const 
{
		assert( 0 );
		return 0;
}
			
/**
* This operation should never be called: The Dynamic Finfo should
* handle it.
*/
bool LookupFinfo::respondToAdd(
		Eref e, Eref src, const Ftype *srcType,
		unsigned int& srcFuncId, unsigned int& returnFuncId,
		int& destMsgId, unsigned int& destIndex
) const
{
		assert( 0 );
		return 0;
}

void LookupFinfo::addFuncVec( const string& cname )
{
	fv_ = new FuncVec( cname, name() );
	fv_->addFunc( set_, ftype() );
	fv_->setDest();
	fv_->makeTrig(); // Make a trigger funcVec
	fv_->makeLookup(); // Make a Lookup funcVec to deal with indexing
}

#ifdef DO_UNIT_TESTS

// Hack to put in Slot globals, which are assigned below in lookupFinfoTest
// but used elsewhere in the class definitions for LookupTestClass
static Slot procSlot;
static Slot sumSlot;
static Slot requestSlot;

/**
 * This test class contains a vector of doubles, a regular double,
 * and an evaluated int with the size of the vector.
 */
class LookupTestClass
{
		public:
			LookupTestClass()
					: dval( 2.2202 )
			{
				dmap["0"] = 0.1;
				dmap["1"] = 0.2;
				dmap["2"] = 0.3;
				dmap["3"] = 0.4;
			}

			static double getDmap( Eref e, const string& s ) {
				map< string, double >::iterator i;
				LookupTestClass* atc = 
						static_cast< LookupTestClass* >( e.data() );
				i = atc->dmap.find( s );
				if ( i != atc->dmap.end() )
					return i->second;

				cout << "Error: LookupTestClass::getDvec: index not found \n";
				return 0.0;
			}

			static void setDmap( 
						const Conn* c, double val, const string& s ) {
				LookupTestClass* atc = 
					static_cast< LookupTestClass* >( c->data( ) );
				map< string, double >::iterator i = atc->dmap.find( s );
				if ( i != atc->dmap.end() )
					i->second = val;
				else
					cout << "Error: LookupTestClass::setDvec: index not found\n";
			}

			static double getDval( Eref e ) {
				return static_cast< LookupTestClass* >( e.data() )->dval;
			}
			static void setDval( const Conn* c, double val ) {
				static_cast< LookupTestClass* >( c->data( ) )->dval = val;
			}

			// A proper message, adds incoming val to dval.
			static void dsum( const Conn* c, double val ) {
				static_cast< LookupTestClass* >( c->data( ) )->dval += val;
			}

			// another proper message. Triggers a local operation,
			// triggers sending of dval, and triggers a trigger out.
			static void proc( const Conn* c ) {
				void* data = c->data();
				LookupTestClass* tc =
						static_cast< LookupTestClass* >( data );
				tc->dval = 0.0;
				map< string, double >::iterator i;
				for ( i = tc->dmap.begin(); i != tc->dmap.end(); i++ )
					tc->dval += i->second;

				// This sends the double value out to a target
				send1< double >( c->target(), sumSlot, tc->dval );

				// This just sends a trigger to the remote object.
				send0( c->target(), procSlot );
			}

		private:
			map< string, double > dmap;
			double dval;
};

void lookupFinfoTest()
{

	cout << "\nTesting lookupFinfo set and get";

	const Ftype* f1a = LookupFtype< double, string >::global();
	const Ftype* f1d = ValueFtype1< double >::global();
	const Ftype* f0 = Ftype0::global();
	static Finfo* requestValShared[] =
	{
		new SrcFinfo( "procout", f0 ),
		new DestFinfo( "dsum", f1d, RFCAST( &LookupTestClass::dsum ) ),
	};
	static Finfo* testFinfos[] = 
	{
		new LookupFinfo( "dmap", f1a,
			reinterpret_cast< GetFunc >( &LookupTestClass::getDmap ),
			reinterpret_cast< RecvFunc >( &LookupTestClass::setDmap ) ),
		new ValueFinfo( "dval", f1d,
				LookupTestClass::getDval,
				reinterpret_cast< RecvFunc >( &LookupTestClass::setDval ) ),
		new SharedFinfo( "requestVal", requestValShared,
			sizeof( requestValShared ) / sizeof( Finfo* ) ),
		new SrcFinfo( "dsumout", f1d ),
		new SrcFinfo( "procout", f0 ),
		new DestFinfo( "dsum", f1d, RFCAST( &LookupTestClass::dsum ) ),
		new DestFinfo( "proc", f0, &LookupTestClass::proc ),
	};

	Cinfo lookuptestclass( "lookuptestclass", "Upi",
					"Lookup Test class",
					initNeutralCinfo(),
					testFinfos, 
					sizeof( testFinfos ) / sizeof( Finfo*),
					ValueFtype1< LookupTestClass >::global() );

	FuncVec::sortFuncVec();

	sumSlot = lookuptestclass.getSlot( "dsumout" );
	procSlot = lookuptestclass.getSlot( "procout" );
	requestSlot = lookuptestclass.getSlot( "requestVal.procout" );


	Element* a1 = lookuptestclass.create( Id::scratchId(), "a1" );
	double dret = 0;

	get< double >( a1, "dval", dret );
	ASSERT( dret == 2.2202, "test get1");
	set< double >( a1, "dval", 555.5 );
	dret = 0;
	get< double >( a1, "dval", dret );
	ASSERT( dret == 555.5, "test set1");

	vector< Finfo* > flist;
	unsigned int ndyn = a1->listLocalFinfos( flist );
	ASSERT( ndyn == 0, "Counting DynFinfos" );

	// Here we have the crucial functions for the LookupFinfo:
	// the ability to lookup the data using the lookupGet/Set functions
	// where the index is provided in the function itself, rather
	// than in the name of the field.
	
	lookupGet< double, string >( a1, "dmap", dret, "0" );
	ASSERT( dret == 0.1, "test lookupGet0");
	lookupSet< double, string >( a1, "dmap", 1111.1, "0" );
	dret = 0;
	lookupGet< double, string >( a1, "dmap", dret, "0" );
	ASSERT( dret == 1111.1, "test lookupSet0");


	// Note that the match function will treat the indices as
	// strings.
	// First we confirm that the zeroth entry is the same as touched
	// by the lookupSet previously.
	get< double >( a1, "dmap[0]", dret );
	ASSERT( dret == 1111.1, "test get0");
	set< double >( a1, "dmap[0]", 1.1 );
	dret = 0;
	get< double >( a1, "dmap[0]", dret );
	ASSERT( dret == 1.1, "test set0");

	// Check that there is only one DynamicFinfo set up when looking at
	// the same lookup index.
	ASSERT( a1->listLocalFinfos( flist ) == 1, "Counting DynFinfos" );

	get< double >( a1, a1->findFinfo( "dmap[1]" ), dret );
	ASSERT( dret == 0.2, "test get1");
	set< double >( a1, a1->findFinfo( "dmap[1]" ), 2.2 );
	dret = 0;
	get< double >( a1, a1->findFinfo( "dmap[1]" ), dret );
	ASSERT( dret == 2.2, "test set1");

	// Check that there is only one DynamicFinfo set up when looking at
	// the same lookup index.
	ASSERT( a1->listLocalFinfos( flist ) == 1, "Counting DynFinfos" );

	get< double >( a1, a1->findFinfo( "dmap[3]" ), dret );
	ASSERT( dret == 0.4, "test get3");
	set< double >( a1, a1->findFinfo( "dmap[3]" ), 3.3 );
	dret = 1234.567;
	get< double >( a1, a1->findFinfo( "dmap[3]" ), dret );
	ASSERT( dret == 3.3, "test set3");

	// Check that the system reuses the DynamicFinfo.
	ASSERT( a1->listLocalFinfos( flist ) == 1, "Counting DynFinfos" );

	////////////////////////////////////////////////////////////////
	// Check string set and get.
	////////////////////////////////////////////////////////////////
	
	string sret = "";
	bool bret = 0;
	bret = a1->findFinfo( "dmap[3]" )->strSet( a1, "-2.3" );
	ASSERT( bret, "strSet" );
	get< double >( a1, a1->findFinfo( "dmap[3]" ), dret );
	ASSERT( dret == -2.3, "test strSet");
	bret = a1->findFinfo( "dmap[3]" )->strGet( a1, sret );
	ASSERT( bret, "strSet" );
	ASSERT( sret == "-2.3", "test strSet");

	bret = a1->findFinfo( "dmap[2]" )->strSet( a1, "-0.03" );
	ASSERT( bret, "strSet" );
	bret = get< double >( a1, a1->findFinfo( "dmap[2]" ), dret );
	ASSERT( bret, "strGet" );
	ASSERT( dret == -0.03, "test strGet");

	////////////////////////////////////////////////////////////////
	// Now we start testing messages between LookupFinfo fields.
	////////////////////////////////////////////////////////////////
	
	cout << "\nTesting lookupFinfo messaging";
	Element* a2 = lookuptestclass.create( Id::scratchId(), "a2" );

	// We will follow a1 messages to call proc on a2. Check a2->dval.
	//
	// proc on a2 will send this value of dval to a1->dmap[0]. Check it.
	//
	// a1 trigger message will call send on a2->dmap[1]. This goes
	//   to a1->dval. Check it.
	//
	// a1 trigger message will call send on a2->dmap[2]. This goes
	//   to a1->dmap[2]. The trigger is created first. Check it.
	//
	// a1 trigger message will call send on a2->dmap[3]. This goes
	//   to a1->dmap[3]. The send is created first. Check it.

	// 1. We will follow a1 messages to call proc on a2. Check a2->dval.
	const Finfo *f1 = a1->findFinfo( "procout" );
	const Finfo *f2 = a2->findFinfo( "proc" );
	bret = f1->add( a1, a2, f2, ConnTainer::Default );
	ASSERT( bret, "adding procout to proc");

	// 2. proc on a2 will send this value of dval to a1->dmap[0].
	f1 = a2->findFinfo( "dsumout" );
	f2 = a1->findFinfo( "dmap[0]" );
	bret = f1->add( a2, a1, f2, ConnTainer::Default );
	ASSERT( bret, "Adding dsumout to dval");
	// We have already made a finfo for a1->dmap[0]. Check that this
	// is the one that is used for the messaging.
	ASSERT( a1->listLocalFinfos( flist ) == 1, "Counting DynFinfos" );

	// 3. a1 trigger message will call request on a2->dmap[1]. The
	// value comes back to a1, and is added to dval.
	f1 = a1->findFinfo( "requestVal" );
	f2 = a2->findFinfo( "dmap[1]" );
	bret = f1->add( a1, a2, f2, ConnTainer::Default );
	ASSERT( bret, "Adding requestVal to dmap[1]");
	// Here we made a new DynamicFinfo for the regular ValueFinfo.
	ASSERT( a2->listLocalFinfos( flist ) == 1, "Counting DynFinfos" );

	// 4. a2 trigger message will call request on a1->dmap[2]. This
	//   is set up using reverse Shared messaging calls. The
	//   value comes back to a2 and is added to dval.
	f1 = a1->findFinfo( "dmap[2]" );
	f2 = a2->findFinfo( "requestVal" );
	bret = f1->add( a1, a2, f2, ConnTainer::Default );
	ASSERT( bret, "Adding a1.dmap[2] to a2.requestVal");

	// We have not made a finfo for a1->dmap[2]. Check that this
	// new one is used for the messaging.
	ASSERT( a1->listLocalFinfos( flist ) == 2, "Counting DynFinfos" );

	///////////////////////////////////////////////////////////////////
	// Now setup is done, let's start sending info around.
	///////////////////////////////////////////////////////////////////
	bret = set< double >( a1, "dval", 4321.0 );
	bret &= set< double >( a2, "dval", 1234.0 );
	bret &= set< double >( a1, "dmap[0]", 10.0 );
	bret &= set< double >( a1, "dmap[1]", 20.0 );
	bret &= set< double >( a1, "dmap[2]", 30.0 );
	bret &= set< double >( a1, "dmap[3]", 40.0 );
	bret &= set< double >( a2, "dmap[0]", 1.0 );
	bret &= set< double >( a2, "dmap[1]", 2.0 );
	bret &= set< double >( a2, "dmap[2]", 3.0 );
	bret &= set< double >( a2, "dmap[3]", 4.0 );
	ASSERT( bret, "assignment");

	bret = get< double >( a1, a1->findFinfo( "dmap[2]" ), dret );
	ASSERT( bret, "test assignment");
	ASSERT( dret == 30.0, "test assignment");

	send0( a1, procSlot ); // procout
	// Here a2->dval should simply become the sum of its lookup entries.
	// As this has just been initialized, the sum should be 10.0.
	// Bad Upi: should never test for equality of doubles.
	get< double >( a2, a2->findFinfo( "dval" ), dret );
	ASSERT( dret == 10.0, "test msg1");

	// proc on a2 will send this value of dval to a1->dmap[0]. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dmap[0]" ), dret );
	ASSERT( dret == 10.0, "test msg2");

	//////////////////////////////////////////////////////////////////////
	// a1 trigger message will call send on a2->dmap[1], which currently
	// holds the value 2. The value is added to a1->dval, which is 4321
	send0( a1, requestSlot ); // procout
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dval" ), dret );
	ASSERT( dret == 4323.0, "test msg3");

	// 4. a2 trigger message will call request on a1->dmap[2]. This
	//   is set up using reverse Shared messaging calls. The
	//   value ( 30 ) comes back to a2 and is added to dval (10).
	bret = get< double >( a2, a2->findFinfo( "dval" ), dret );
	ASSERT( bret, "test msg4");
	ASSERT( dret == 10.0, "test msg4");
	dret = 0;
	bret = get< double >( a1, a1->findFinfo( "dmap[2]" ), dret );
	ASSERT( bret, "test msg4");
	ASSERT( dret == 30.0, "test msg4");

	send0( a2, requestSlot ); // procout
	dret = 0.0;
	get< double >( a2, a2->findFinfo( "dval" ), dret );
	ASSERT( dret == 40, "test msg4");

	// Check that there are no strange things happening with the
	// Finfos when the messaging is actually used.
	// Note that we set all of a1.dmap[], only 2 of which were already used
	// Note that we set all of a2.dmap[], only 1 of which was already used
	// Since the system reuses when it can, the net increment in each is 1.
	ASSERT( a1->listLocalFinfos( flist ) == 3, "Counting DynFinfos" );
	ASSERT( a2->listLocalFinfos( flist ) == 2, "Counting DynFinfos" );
}

#endif // DO_UNIT_TESTS
