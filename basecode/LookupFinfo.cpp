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

		///\todo: need to define the strToIndex function
		// void* index = ftype()->strToIndex( indexStr );
		string* index = new string( indexStr );
		string n = name() + "[" + indexStr + "]";
		DynamicFinfo* ret = 
				new DynamicFinfo(
					n,
					this, 
					set_, get_,
					ftype()->recvFunc(), ftype()->trigFunc()
				);
		ret->setGeneralIndex( static_cast< void* >( index ) );
		e->addFinfo( ret );
		return ret;
	}
	return 0;
}

/**
* Dynamic Finfo should handle it.
*/
bool LookupFinfo::add( 
	Element* e, Element* destElm, const Finfo* destFinfo
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
		Element* e, Element* src, const Ftype *srcType,
		FuncList& srcFl, FuncList& returnFl,
		unsigned int& destIndex, unsigned int& numDest
) const
{
		assert( 0 );
		return 0;
}


/// Dummy function: DynamicFinfo should handle
void LookupFinfo::dropAll( Element* e ) const
{
		assert( 0 );
}

/// Dummy function: DynamicFinfo should handle
bool LookupFinfo::drop( Element* e, unsigned int i ) const
{
		assert( 0 );
		return 0;
}


#ifdef DO_UNIT_TESTS
#include "moose.h"

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

			static double getDmap( const Element* e, const string& s ) {
				map< string, double >::iterator i;
				LookupTestClass* atc = 
						static_cast< LookupTestClass* >( e->data() );
				i = atc->dmap.find( s );
				if ( i != atc->dmap.end() )
					return i->second;

				cout << "Error: LookupTestClass::getDvec: index not found \n";
				return 0.0;
			}

			static void setDmap( 
						const Conn& c, double val, const string& s ) {
				LookupTestClass* atc = 
					static_cast< LookupTestClass* >(
									c.targetElement()->data() );
				map< string, double >::iterator i = atc->dmap.find( s );
				if ( i != atc->dmap.end() )
					i->second = val;
				else
					cout << "Error: LookupTestClass::setDvec: index not found\n";
			}

			static double getDval( const Element* e ) {
				return static_cast< LookupTestClass* >( e->data() )->dval;
			}
			static void setDval( const Conn& c, double val ) {
				static_cast< LookupTestClass* >( 
					c.targetElement()->data() )->dval = val;
			}

			// A proper message, adds incoming val to dval.
			static void dsum( const Conn& c, double val ) {
				static_cast< LookupTestClass* >( 
					c.targetElement()->data() )->dval += val;
			}

			// another proper message. Triggers a local operation,
			// triggers sending of dval, and triggers a trigger out.
			static void proc( const Conn& c ) {
				Element* e = c.targetElement();
				LookupTestClass* tc =
						static_cast< LookupTestClass* >( e->data() );
				tc->dval = 0.0;
				map< string, double >::iterator i;
				for ( i = tc->dmap.begin(); i != tc->dmap.end(); i++ )
					tc->dval += i->second;

				// This sends the double value out to a target
				// dsumout == 0, but we make it one because of
				// base neutral class adding fields.
				send1< double >( e, 1, tc->dval );

				// This just sends a trigger to the remote object.
				// procout == 1, but set to 2 because of base class
				// Either it will trigger dproc itself, or it
				// could trigger a getfunc.
				send0( e, 2 );
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
	static Finfo* testFinfos[] = 
	{
		new LookupFinfo( "dmap", f1a,
			reinterpret_cast< GetFunc >( &LookupTestClass::getDmap ),
			reinterpret_cast< RecvFunc >( &LookupTestClass::setDmap ) ),
		new ValueFinfo( "dval", f1d,
				LookupTestClass::getDval,
				reinterpret_cast< RecvFunc >( &LookupTestClass::setDval ) ),
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

	Element* a1 = lookuptestclass.create( "a1" );
	double dret = 0;

	get< double >( a1, "dval", dret );
	ASSERT( dret == 2.2202, "test get1");
	set< double >( a1, "dval", 555.5 );
	dret = 0;
	get< double >( a1, "dval", dret );
	ASSERT( dret == 555.5, "test set1");

	vector< const Finfo* > flist;
	a1->listFinfos( flist );
	size_t s = flist.size();

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
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 1, "Checking formation of DynamicFinfos" );


	get< double >( a1, a1->findFinfo( "dmap[1]" ), dret );
	ASSERT( dret == 0.2, "test get1");
	set< double >( a1, a1->findFinfo( "dmap[1]" ), 2.2 );
	dret = 0;
	get< double >( a1, a1->findFinfo( "dmap[1]" ), dret );
	ASSERT( dret == 2.2, "test set1");

	// Check that there is only one DynamicFinfo set up when looking at
	// the same lookup index.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 2, "Checking formation of DynamicFinfos" );

	get< double >( a1, a1->findFinfo( "dmap[3]" ), dret );
	ASSERT( dret == 0.4, "test get3");
	set< double >( a1, a1->findFinfo( "dmap[3]" ), 3.3 );
	dret = 1234.567;
	get< double >( a1, a1->findFinfo( "dmap[3]" ), dret );
	ASSERT( dret == 3.3, "test set3");

	// Check that there is only one DynamicFinfo set up when looking at
	// the same lookup index.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 3, "Checking formation of DynamicFinfos" );

	////////////////////////////////////////////////////////////////
	// Now we start testing messages between LookupFinfo fields.
	////////////////////////////////////////////////////////////////
	
	cout << "\nTesting lookupFinfo messaging";
	Element* a2 = lookuptestclass.create( "a2" );

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
	ASSERT( a1->findFinfo( "procout" )->
			add( a1, a2, a2->findFinfo( "proc" ) ),
			"adding procout to proc"
			);

	// 2. proc on a2 will send this value of dval to a1->dmap[0].
	ASSERT( 
		a2->findFinfo( "dsumout" )->
			add( a2, a1, a1->findFinfo( "dmap[0]" ) ),
			"Adding dsumout to dval"
		);
	// We have already made a finfo for a1->dmap[0]. Check that this
	// is the one that is used for the messaging.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 3, "reuse of DynamicFinfos" );

	// 3. a1 trigger message will call send on a2->dmap[1]. This goes
	//   to a1->dval.
	ASSERT( 
		a1->findFinfo( "procout" )->
			add( a1, a2, a2->findFinfo( "dmap[1]" ) ),
			"Adding procout to dmap[1]"
		);
	ASSERT( 
		a2->findFinfo( "dmap[1]" )->
			add( a2, a1, a1->findFinfo( "dval" ) ),
			"Adding dmap[1] to dval"
		);
	// Here we made a new DynamicFinfo for the regular ValueFinfo.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 4, "No new DynamicFinfos." );

	// 4. a1 trigger message will call send on a2->dmap[2]. This goes
	//   to a1->dmap[2]. The trigger is created first. Check it.
	ASSERT( 
		a1->findFinfo( "procout" )->
			add( a1, a2, a2->findFinfo( "dmap[2]" ) ),
			"Adding procout to dmap[2]"
		);
	ASSERT( 
		a2->findFinfo( "dmap[2]" )->
			add( a2, a1, a1->findFinfo( "dmap[2]" ) ),
			"Adding dmap[2] to dmap[2] after trigger"
		);
	// We have not made a finfo for a1->dmap[2]. Check that this
	// new one is used for the messaging.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 5, "New DynamicFinfo for dmap[2]" );

	// 5. a1 trigger message will call send on a2->dmap[3]. This goes
	//   to a1->dmap[3]. The send is created first. Check it.
	ASSERT( 
		a2->findFinfo( "dmap[3]" )->
			add( a2, a1, a1->findFinfo( "dmap[3]" ) ),
			"Adding dmap[3] to dmap[3] before trigger"
		);
	ASSERT( 
		a1->findFinfo( "procout" )->
			add( a1, a2, a2->findFinfo( "dmap[3]" ) ),
			"Adding procout to dmap[3]"
		);
	// We have not made a finfo for a1->dmap[3]. Check that this
	// new one is used for the messaging.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 5, "Old DynamicFinfo for dmap[3]" );

	unsigned int procOutSlot = lookuptestclass.getSlotIndex( "procout");

	send0( a1, procOutSlot ); // procout
	// Here a2->dval should simply become the sum of its lookup entries.
	// As this has just been initialized, the sum should be 1.0.
	// Bad Upi: should never test for equality of doubles.
	get< double >( a2, a2->findFinfo( "dval" ), dret );
	ASSERT( dret == 1.0, "test msg1");

	// proc on a2 will send this value of dval to a1->dmap[0]. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dmap[0]" ), dret );
	ASSERT( dret == 1.0, "test msg2");

	// a1 trigger message will call send on a2->dmap[1]. This goes
	//   to a1->dval. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dval" ), dret );
	ASSERT( dret == 0.2, "test msg3");

	// a1 trigger message will call send on a2->dmap[2]. This goes
	//   to a1->dmap[2]. The trigger is created first. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dmap[2]" ), dret );
	ASSERT( dret == 0.3, "test msg4");

	// a1 trigger message will call send on a2->dmap[3]. This goes
	//   to a1->dmap[3]. The send is created first. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dmap[3]" ), dret );
	ASSERT( dret == 0.4, "test msg5");

	// Check that there are no strange things happening with the
	// Finfos when the messaging is actually used.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 5, "Same DynamicFinfo for dmap[3]" );
}

#endif // DO_UNIT_TESTS
