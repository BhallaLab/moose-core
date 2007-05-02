/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Enzyme.h"
#include "../element/Neutral.h"

const Cinfo* initEnzymeCinfo()
{
	static TypeFuncPair processTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &Enzyme::processFunc ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &Enzyme::reinitFunc ) ),
	};
	static TypeFuncPair substrateTypes[] =
	{
		TypeFuncPair( Ftype2< double, double >::global(), 0 ),
		TypeFuncPair( Ftype1< double >::global(),
			RFCAST( &Enzyme::substrateFunc ) ),
	};
	static TypeFuncPair enzTypes[] =
	{
		TypeFuncPair( Ftype2< double, double >::global(), 0 ),
		TypeFuncPair( Ftype1< double >::global(),
			RFCAST( &Enzyme::enzymeFunc ) ),
	};
	static TypeFuncPair cplxTypes[] =
	{
		TypeFuncPair( Ftype2< double, double >::global(), 0 ),
		TypeFuncPair( Ftype1< double >::global(),
			RFCAST( &Enzyme::complexFunc ) ),
	};

	static Finfo* enzymeFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "k1", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getK1 ), 
			RFCAST( &Enzyme::setK1 ) 
		),
		new ValueFinfo( "k2", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getK2 ), 
			RFCAST( &Enzyme::setK2 ) 
		),
		new ValueFinfo( "k3", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getK3 ), 
			RFCAST( &Enzyme::setK3 ) 
		),
		new ValueFinfo( "Km", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getKm ), 
			RFCAST( &Enzyme::setKm ) 
		),
		new ValueFinfo( "kcat", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getKcat ), 
			RFCAST( &Enzyme::setKcat ) 
		),
		new ValueFinfo( "mode", 
			ValueFtype1< bool >::global(),
			GFCAST( &Enzyme::getMode ), 
			RFCAST( &Enzyme::setMode ) 
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "prdSrc", Ftype2< double, double >::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "scaleKm", 
			Ftype1< double >::global(),
			RFCAST( &Enzyme::scaleKmFunc ) ),
		new DestFinfo( "scaleKcat", 
			Ftype1< double >::global(),
			RFCAST( &Enzyme::scaleKcatFunc ) ),
		new DestFinfo( "intramol", 
			Ftype1< double >::global(),
			RFCAST( &Enzyme::intramolFunc ) ),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "process", processTypes, 2 ),
		new SharedFinfo( "sub", substrateTypes, 2 ),
		new SharedFinfo( "enz", enzTypes, 2 ),
		new SharedFinfo( "cplx", cplxTypes, 2 ),
	};

	static  Cinfo enzymeCinfo(
		"Enzyme",
		"Upinder S. Bhalla, 2007, NCBS",
	"Enzyme: Irreversible enzymatic reaction that supports two forms of the \nMichaelis-Menten formulation for enzyme catalysis:\nE + S <======> E.S ------> E + P\nIn this enzyme, the forward rate for the complex formation is\nk1, the backward rate is k2, and the final rate for product\nformation is k3. In terms of Michaelis-Menten parameters,\nk3 = kcat, and\n(k3 + k2)/k1 = Km.\nIn all forms, the enzyme object should be considered as an\nenzymatic activity. It must be created in association with\nan enzyme molecule. The same enzyme molecule may have multiple\nactivities, for example, on a range of substrates.\nIn the explicit form (default) the enzyme substrate complex E.S\nis explictly created as a distinct molecular pool. This is\nperhaps more realistic in complex models where there are likely\nto be other substrates for this enzyme, and so enzyme \nsaturation effects must be accounted for. However the complex\nmolecule does not participate in any other reactions, which\nmay itself be a poor assumption. If this is a serious concern\nthen it is best to do the entire enzymatic process\nusing elementary reactions.\nIn the implicit form there is no actual enzyme-complex molecule.\nIn this form the rate term is\ncomputed using the Michaelis-Menten form\nrate = kcat * [E] * [S] / ( Km + [S] )\nHere the opposite problem from above applies: There is no\nexplicit complex, which means that the level of the free enzyme\nmolecule is unaffected even near saturation. However, other\nreactions involving the enzyme do see the entire enzyme\nconcentration. \nFor the record, I regard the explicit formulation as more\naccurate for complex simulations.",
		initNeutralCinfo(),
		enzymeFinfos,
		sizeof(enzymeFinfos)/sizeof(Finfo *),
		ValueFtype1< Enzyme >::global()
	);

	return &enzymeCinfo;
}

static const Cinfo* enzymeCinfo = initEnzymeCinfo();

static const unsigned int subSlot =
	initEnzymeCinfo()->getSlotIndex( "sub" );
static const unsigned int enzSlot =
	initEnzymeCinfo()->getSlotIndex( "enz" );
static const unsigned int cplxSlot =
	initEnzymeCinfo()->getSlotIndex( "cplx" );
static const unsigned int prdSlot =
	initEnzymeCinfo()->getSlotIndex( "prdSrc" );

///////////////////////////////////////////////////
// Enzyme class function definitions
///////////////////////////////////////////////////

Enzyme::Enzyme()
	: k1_(0.1), k2_(0.4), k3_(0.1),sk1_(1.0)
{
	;
}
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void Enzyme::setK1( const Conn& c, double value )
{
	static_cast< Enzyme* >( c.data() )->k1_ = value;
}

double Enzyme::getK1( const Element* e )
{
	return static_cast< Enzyme* >( e->data() )->k1_;
}

void Enzyme::setK2( const Conn& c, double value )
{
	static_cast< Enzyme* >( c.data() )->k2_ = value;
}

double Enzyme::getK2( const Element* e )
{
	return static_cast< Enzyme* >( e->data() )->k2_;
}

void Enzyme::setK3( const Conn& c, double value )
{
	static_cast< Enzyme* >( c.data() )->k3_ = value;
}

double Enzyme::getK3( const Element* e )
{
	return static_cast< Enzyme* >( e->data() )->k3_;
}

double Enzyme::getKm( const Element* e )
{
	return static_cast< Enzyme* >( e->data() )->Km_;
}
void Enzyme::setKm( const Conn& c, double value )
{
	static_cast< Enzyme* >( c.data() )->innerSetKm( value );
}
void Enzyme::innerSetKm( double value )
{
	if ( value > 0.0 ) {
		Km_ = value;
		k1_ = ( k2_ + k3_ ) / value;
	}
}


double Enzyme::getKcat( const Element* e )
{
	return static_cast< Enzyme* >( e->data() )->k3_;
}
void Enzyme::setKcat( const Conn& c, double value )
{
	static_cast< Enzyme* >( c.data() )->innerSetKcat( value );
}
void Enzyme::innerSetKcat( double value )
{
	if ( value > 0.0 && k3_ > 0.0 ) {
		k2_ *= value / k3_;
		k1_ *= value / k3_;
		k3_ = value;
	}
}
bool Enzyme::getMode( const Element* e )
{
	return static_cast< Enzyme* >( e->data() )->innerGetMode();
}
bool Enzyme::innerGetMode() const
{
	return ( procFunc_ == &Enzyme::implicitProcFunc );
}
void Enzyme::setMode( const Conn& c, bool value )
{
	static_cast< Enzyme* >( c.data() )->innerSetMode(
		       c.targetElement(), value );
}
void Enzyme::innerSetMode( Element* e, bool mode )
{
	Km_ = ( k2_ + k3_ ) / k1_;
	if ( mode == innerGetMode() )
		return;
	if ( mode ) { 
		unsigned int id = 
			Neutral::getChildByName( e, e->name() + "_cplx" );
		if ( id != BAD_ID ) {
			Element* cplx = Element::element( id );
			if ( cplx )
				delete cplx;
		}
		procFunc_ = &Enzyme::implicitProcFunc;
		sA_ = 0;
	} else { 
		procFunc_ = &Enzyme::explicitProcFunc;
		makeComplex( e );
	}
}

///////////////////////////////////////////////////
// Shared message function definitions
///////////////////////////////////////////////////

void Enzyme::innerProcessFunc( Element* e )
{
	(this->*procFunc_)( e );
}

void Enzyme::processFunc( const Conn& c, ProcInfo p )
{
	Element* e = c.targetElement();
	static_cast< Enzyme* >( e->data() )->innerProcessFunc( e );
}
void Enzyme::implicitProcFunc( Element* e )
{
	B_ = s_ * e_ * k3_ * sk1_ / ( s_ + Km_ );
	s_ = 1.0;
	send2< double, double >( e, subSlot, 0.0, B_ );
	send2< double, double >( e, prdSlot, B_, 0.0 );
}
void Enzyme::explicitProcFunc( Element* e )
{
	eA_ = sA_ + pA_;
	B_ = s_ * e_;
	send2< double, double >( e, subSlot, sA_, B_ );
	send2< double, double >( e, prdSlot, pA_, 0.0 );
	send2< double, double >( e, enzSlot, eA_, B_ );
	send2< double, double >( e, cplxSlot, B_, eA_ );
	s_ = k1_ * sk1_;
	sA_ = k2_;
	pA_ = k3_;
}

void Enzyme::innerReinitFunc(  )
{
	eA_ = sA_ = pA_ = B_ = e_ = 0.0;
	s_ = 1.0;
}
		
void Enzyme::reinitFunc( const Conn& c, ProcInfo p )
{
	static_cast< Enzyme* >( c.data() )->innerReinitFunc( );
}

void Enzyme::substrateFunc( const Conn& c, double n )
{
	static_cast< Enzyme* >( c.data() )->s_ *= n;
}

void Enzyme::enzymeFunc( const Conn& c, double n )
{
	static_cast< Enzyme* >( c.data() )->e_ = n;
}

void Enzyme::complexFunc( const Conn& c, double n )
{
	static_cast< Enzyme* >( c.data() )->sA_ *= n;
	static_cast< Enzyme* >( c.data() )->pA_ *= n;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Enzyme::intramolFunc( const Conn& c, double n )
{
	static_cast< Enzyme* >( c.data() )->innerIntramolFunc( n );
}
void Enzyme::innerIntramolFunc( double n )
{
	if ( n > 0 )
		sk1_ = 1.0 / n;
	else
		sk1_ = 1.0;
}
void Enzyme::scaleKmFunc( const Conn& c, double k )
{
	static_cast< Enzyme* >( c.data() )->innerScaleKmFunc( k );
}
void Enzyme::innerScaleKmFunc( double k )
{
	if ( k > 0 )
		s_ /= k;
	else
		cout << "Error: Enzyme::scaleKm msg: negative k = " <<
			k << endl;
}
void Enzyme::scaleKcatFunc( const Conn& c, double k )
{
	static_cast< Enzyme* >( c.data() )->pA_ *= k;
}

///////////////////////////////////////////////////////
// Other func definitions
///////////////////////////////////////////////////////

void Enzyme::makeComplex( Element* e )
{
	static const Finfo* cplxSrcFinfo = enzymeCinfo->findFinfo( "cplx" );
	string cplxName = e->name() + "_cplx";
	unsigned int id = Neutral::getChildByName( e, cplxName );
	if ( id != BAD_ID )
		return;

	double vol = 0.0;

	unsigned int parentId = Neutral::getParent( e );
	assert( parentId != BAD_ID );
	Element* parent = Element::element( parentId );

	bool ret = get< double >( parent, "volumeScale", vol );
	assert( ret );
	Element* complex = Neutral::create( "Molecule", cplxName, e );
	ret = cplxSrcFinfo->add( e, complex, complex->findFinfo( "reac" ) );
	assert( ret );
}

