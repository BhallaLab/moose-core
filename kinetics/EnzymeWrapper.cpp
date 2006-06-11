#include "header.h"
#include "Enzyme.h"
#include "EnzymeWrapper.h"


Finfo* EnzymeWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"k1", &EnzymeWrapper::getK1, 
		&EnzymeWrapper::setK1, "double" ),
	new ValueFinfo< double >(
		"k2", &EnzymeWrapper::getK2, 
		&EnzymeWrapper::setK2, "double" ),
	new ValueFinfo< double >(
		"k3", &EnzymeWrapper::getK3, 
		&EnzymeWrapper::setK3, "double" ),
	new ValueFinfo< double >(
		"Km", &EnzymeWrapper::getKm, 
		&EnzymeWrapper::setKm, "double" ),
	new ValueFinfo< int >(
		"mode", &EnzymeWrapper::getMode, 
		&EnzymeWrapper::setMode, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new SingleSrc2Finfo< double, double >(
		"enzOut", &EnzymeWrapper::getEnzSrc, 
		"processIn", 1 ),
	new SingleSrc2Finfo< double, double >(
		"cplxOut", &EnzymeWrapper::getCplxSrc, 
		"processIn", 1 ),
	new NSrc2Finfo< double, double >(
		"subOut", &EnzymeWrapper::getSubSrc, 
		"processIn", 1 ),
	new NSrc2Finfo< double, double >(
		"prdOut", &EnzymeWrapper::getPrdSrc, 
		"processIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest0Finfo(
		"reinitIn", &EnzymeWrapper::reinitFunc,
		&EnzymeWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &EnzymeWrapper::processFunc,
		&EnzymeWrapper::getProcessConn, "enzOut, cplxOut, subOut, prdOut", 1 ),
	new Dest1Finfo< double >(
		"enzIn", &EnzymeWrapper::enzFunc,
		&EnzymeWrapper::getEnzConn, "", 1 ),
	new Dest1Finfo< double >(
		"cplxIn", &EnzymeWrapper::cplxFunc,
		&EnzymeWrapper::getCplxConn, "", 1 ),
	new Dest1Finfo< double >(
		"subIn", &EnzymeWrapper::subFunc,
		&EnzymeWrapper::getSubConn, "", 1 ),
	new Dest1Finfo< double >(
		"intramolIn", &EnzymeWrapper::intramolFunc,
		&EnzymeWrapper::getIntramolInConn, "" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &EnzymeWrapper::getProcessConn,
		"processIn, reinitIn" ),
	new SharedFinfo(
		"enz", &EnzymeWrapper::getEnzConn,
		"enzIn, enzOut" ),
	new SharedFinfo(
		"cplx", &EnzymeWrapper::getCplxConn,
		"cplxIn, cplxOut" ),
	new SharedFinfo(
		"sub", &EnzymeWrapper::getSubConn,
		"subIn, subOut" ),
};

const Cinfo EnzymeWrapper::cinfo_(
	"Enzyme",
	"Upinder S. Bhalla, 2005, NCBS",
	"Enzyme: Irreversible enzymatic reaction that supports two forms of the \nMichaelis-Menten formulation for enzyme catalysis:\nE + S <======> E.S ------> E + P\nIn this enzyme, the forward rate for the complex formation is\nk1, the backward rate is k2, and the final rate for product\nformation is k3. In terms of Michaelis-Menten parameters,\nk3 = kcat, and\n(k3 + k2)/k1 = Km.\nIn all forms, the enzyme object should be considered as an\nenzymatic activity. It must be created in association with\nan enzyme molecule. The same enzyme molecule may have multiple\nactivities, for example, on a range of substrates.\nIn the explicit form (default) the enzyme substrate complex E.S\nis explictly created as a distinct molecular pool. This is\nperhaps more realistic in complex models where there are likely\nto be other substrates for this enzyme, and so enzyme \nsaturation effects must be accounted for. However the complex\nmolecule does not participate in any other reactions, which\nmay itself be a poor assumption. If this is a serious concern\nthen it is best to do the entire enzymatic process\nusing elementary reactions.\nIn the implicit form there is no actual enzyme-complex molecule.\nIn this form the rate term is\ncomputed using the Michaelis-Menten form\nrate = kcat * [E] * [S] / ( Km + [S] )\nHere the opposite problem from above applies: There is no\nexplicit complex, which means that the level of the free enzyme\nmolecule is unaffected even near saturation. However, other\nreactions involving the enzyme do see the entire enzyme\nconcentration. \nFor the record, I regard the explicit formulation as more\naccurate for complex simulations.",
	"Neutral",
	EnzymeWrapper::fieldArray_,
	sizeof(EnzymeWrapper::fieldArray_)/sizeof(Finfo *),
	&EnzymeWrapper::create
);

///////////////////////////////////////////////////
// Create function definition
///////////////////////////////////////////////////

Element* EnzymeWrapper::create(
	const string& name, Element* pa, const Element* proto )
{
	if ( pa->cinfo()->name() != "Molecule" ) {
		cerr << "Error: EnzymeWrapper::create: parent " << pa->path() <<
			" is not a Molecule\n";
		return 0;
	}
	EnzymeWrapper* e = new EnzymeWrapper(name);
	// Put proto initialization stuff here
	// Problem here: we cannot complete assignment of complex
	// till the parent is attached.
	const EnzymeWrapper* p = 
		dynamic_cast<const EnzymeWrapper *>(proto);
	if ( p ) {
		e->k1_ = p->k1_;
		e->k2_ = p->k2_;
		e->k3_ = p->k3_;
		e->Km_ = p->Km_;
		e->procFunc_ = p->procFunc_;
	}
	return new EnzymeWrapper(name);
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void EnzymeWrapper::makeComplex()
{
	string cplxName = name() + "_cplx";
	if ( relativeFind( cplxName ) )
		return;

	double vol = 0.0;
	if ( !Ftype1< double >::get( parent(), "volumeScale", vol ) ) {
		cerr << "Error: EnzymeWrapper::makeComplex(): Cannot get volumeScale from parent()\n";
		return;
	}

	Element* complex = 
		Cinfo::find("Molecule")->create( cplxName, this );
	Field f = complex->field( "reac" );
	field( "cplx" ).add( f );
	// AddMsg( e, "reac", complex, "reac" );
	Ftype1< double >::set( complex, "volumeScale", vol );
}

void EnzymeWrapper::innerSetMode( int mode )
{
	Km_ = ( k2_ + k3_ ) / k1_;
	if ( mode == innerGetMode() )
		return;

	if ( mode ) { 
		Element* cplx = relativeFind( name() + "_cplx" );
		if ( cplx )
			delete cplx;
		procFunc_ = &EnzymeWrapper::implicitProcFunc;
//		Km_ = ( k2_ + k3_ ) / k1_;
		sA_ = 0;
	} else { 
		procFunc_ = &EnzymeWrapper::explicitProcFunc;
		makeComplex();
	}
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////
void EnzymeWrapper::implicitProcFunc()
{
	B_ = s_ * e_ * k3_ * sk1_ / ( s_ + Km_ );
	s_ = 1.0;
	subSrc_.send( 0.0, B_ );
	prdSrc_.send( B_, 0.0 );
}

void EnzymeWrapper::explicitProcFunc()
{
	eA_ = sA_ + pA_;
	B_ = s_ * e_;
	subSrc_.send( sA_, B_ );
	prdSrc_.send( pA_, 0.0 );
	enzSrc_.send( eA_, B_ );
	cplxSrc_.send( B_, eA_ );
	s_ = k1_ * sk1_;
	sA_ = k2_;
	pA_ = k3_;
}

void EnzymeWrapper::processFuncLocal( ProcInfo info )
{
	(this->*procFunc_)();
}
void EnzymeWrapper::intramolFuncLocal( double n )
{
	if ( n > 0 )
		sk1_ = 1.0 / n;
	else
		sk1_ = 1.0;
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnEnzymeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( EnzymeWrapper, processConn_ );
	return reinterpret_cast< EnzymeWrapper* >( ( unsigned long )c - OFFSET );
}

Element* enzConnEnzymeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( EnzymeWrapper, enzConn_ );
	return reinterpret_cast< EnzymeWrapper* >( ( unsigned long )c - OFFSET );
}

Element* cplxConnEnzymeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( EnzymeWrapper, cplxConn_ );
	return reinterpret_cast< EnzymeWrapper* >( ( unsigned long )c - OFFSET );
}

Element* intramolInConnEnzymeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( EnzymeWrapper, intramolInConn_ );
	return reinterpret_cast< EnzymeWrapper* >( ( unsigned long )c - OFFSET );
}

