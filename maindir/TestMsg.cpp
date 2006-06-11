/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "TestMsg.h"

//////////////////////////////////////////////////////////////////
// This is a class for testing messaging.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// TestMsg initialization
//////////////////////////////////////////////////////////////////

Finfo* TestMsg::fieldArray_[] = 
{
	new ValueFinfo< int >(
		"i", &TestMsg::getI, &TestMsg::setI, "int" ),
	new ValueFinfo< int >(
		"j", &TestMsg::getJ, &TestMsg::setJ, "int" ),
	new ValueFinfo< string >(
		"s", &TestMsg::getS, &TestMsg::setS, "string" ),
	new ArrayFinfo< int >(
		"v", &TestMsg::getV, &TestMsg::setV, "int" ),
	new NSrc1Finfo< int >(
		"out", &TestMsg::getOut, "in" ),
	new Dest1Finfo< int >(
		"in", &TestMsg::inFunc, &TestMsg::getInConn, "out" ),
	new NSrc1Finfo< int >(
		"out2", &TestMsg::getOut2, "in2" ),
	new Dest1Finfo< int >(
		"in2", &TestMsg::inFunc2, &TestMsg::getInConn2, "out2" ),

	new SingleSrc1Finfo< int >(
		"oneout", &TestMsg::getOneOut, "onein" ),
	new Dest1Finfo< int >(
		"onein", &TestMsg::inOneFunc, &TestMsg::getOneInConn, "oneout"),

	new ArrayFinfo< int >(
		"w", &TestMsg::getW, &TestMsg::setW, "int" ),
	new Synapse1Finfo< int >(
		"syn", &TestMsg::synFunc, &TestMsg::getSynConn, 
		&TestMsg::newSynConn, ""),
};

const Cinfo TestMsg::cinfo_(
	"TestMsg",
	"Upinder S. Bhalla, NCBS",
	"TestMsg class. Testing messaging.",
	"Element",
	TestMsg::fieldArray_,
	sizeof(TestMsg::fieldArray_)/sizeof(Finfo *),
	&TestMsg::create
);


//////////////////////////////////////////////////////////////////
// TestMsg functions
//////////////////////////////////////////////////////////////////

TestMsg::~TestMsg()
{
	;
}

// In this function we take the proto's value
Element* TestMsg::create(
			const string& name, Element* parent, const Element* proto)
{
	TestMsg* ret = new TestMsg(name);
	const TestMsg* p = dynamic_cast<const TestMsg *>(proto);
	if (p) {
		ret->i_ = p->i_;
		ret->j_ = p->j_;
		ret->s_ = p->s_;
	}
	if (parent->adoptChild(ret)) {
		return ret;
	} else {
		delete ret;
		return 0;
	}
	return 0;
}

//////////////////////////////////////////////////////////////////
// TestMsg array functions
//////////////////////////////////////////////////////////////////


/*
void TestMsg::setV(Conn* c, int i)
{
	RelayConn* rc = dynamic_cast<RelayConn *>(c);
	if (rc) {
		TestMsg* tf = static_cast< TestMsg* >( c->parent() );
		ArrayFinfo< int >* af = 
			dynamic_cast< ArrayFinfo< int >* >( rc->finfo() );
		if ( af ) {
			if ( tf->v_.size() > af->index() )
				tf->v_[ af->index() ] = i;
		} else {
			cout << "Error: TestMsg::setV: Failed to cast into ArrayFinfo\n";
		}
	}
}
*/

void TestMsg::setV(Element* e, unsigned long index, int value )
{
	TestMsg* tf = static_cast< TestMsg* >( e );
	if ( tf->v_.size() > index )
		tf->v_[ index ] = value;
	else
		cout << "Error:TestMsg::setV: Failed to cast into ArrayFinfo\n";
}

int TestMsg::getV( const Element* e, unsigned long index ) {
	const TestMsg* tf = static_cast< const TestMsg* >( e );
	if (index < tf->v_.size())
		return tf->v_[index];
	return 0;
}

//////////////////////////////////////////////////////////////////
// TestMsg synaptic functions
//////////////////////////////////////////////////////////////////

/*
void TestMsg::setW(Conn* c, int i)
{
	RelayConn* rc = dynamic_cast<RelayConn *>(c);
	if (rc) {
		TestMsg* tf = static_cast< TestMsg* >( c->parent() );
		RelayFinfo1< int >* rf = 
			dynamic_cast< RelayFinfo1< int >* >( rc->finfo() );
		if ( rf ) {
			ArrayFinfo< int >* af = 
				dynamic_cast< ArrayFinfo< int >* >( rf->innerFinfo() );
			if ( af ) {
				if ( tf->synConn_.size() > af->index() )
					tf->synConn_[ af->index() ]->value_ = i;
			} else {
				cout << "Error: TestMsg::setW: Failed to cast into ArrayFinfo\n";
			}
		}
	}
}
*/

void TestMsg::setW(Element* e, unsigned long index, int value )
{
	TestMsg* tf = static_cast< TestMsg* >( e );
	if ( tf->synConn_.size() > index )
		tf->synConn_[ index ]->value_ = value;
	else
		cout << "Error:TestMsg::setV: Failed to cast into ArrayFinfo\n";
}

int TestMsg::getW( const Element* e, unsigned long index )
{
	const TestMsg* tf = static_cast< const TestMsg* >( e );
	if (index < tf->synConn_.size())
		return tf->synConn_[index]->value_;
	return 0;
}

void TestMsg::synFunc( Conn* c, int i )
{
	SynConn< int >* s = static_cast< SynConn< int >* >( c );
	TestMsg* tf = static_cast< TestMsg* >( c->parent() );
	tf->j_ += s->value_ * i;
	// cout << "Got input " << i << " to synapse with wt " << s->value_ << "\n";

// If T was double wt, double delay, we could 
// queue up the pending events according to the
// delay. We could even look into subtracting a transit delay, but
// that would require that we take an 'elapsed time' argument.
}

unsigned long TestMsg::newSynConn( Element* e ) {
	TestMsg* t = static_cast< TestMsg* >( e );
	SynConn< int >* s = new SynConn< int >( e );
	// s->value_ = 1; // Synaptic weight.
	// s->value_ = t->synConn_.size() + 1; // Synaptic weight.
	t->synConn_.push_back( s );
	return t->synConn_.size() - 1;
}


//////////////////////////////////////////////////////////////////
// TestMsg friend functions
//////////////////////////////////////////////////////////////////


Element* inOneConnLookup( const Conn* c )
{
	static const unsigned long OFFSET = 
		FIELD_OFFSET( TestMsg, inOneConn_ );
		//(unsigned long) ( &TestMsg::inOneConn_ );
	return reinterpret_cast< TestMsg* >( ( unsigned long )c - OFFSET );
	// return  ( Element * )( ( unsigned long )c - OFFSET );
}

Element* outOneConnLookup( const Conn* c )
{
	static const unsigned long OFFSET = 
		FIELD_OFFSET( TestMsg, outOneConn_ );
//		(unsigned long) ( &TestMsg::outOneConn_ );
//		offsetof( TestMsg, outOneConn_ );
//		(void *)&( ( TestMsg * )0L )->outOneConn_ - (void * ) 0L ;

//	return  ( Element * )( ( unsigned long )c - OFFSET );
	return reinterpret_cast< TestMsg* >( ( unsigned long )c - OFFSET );
}

