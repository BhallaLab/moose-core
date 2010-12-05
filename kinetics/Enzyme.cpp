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
#include "Molecule.h"

extern double getVolScale( Eref e ); // defined in KinCompt.cpp

const Cinfo* initEnzymeCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &Enzyme::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &Enzyme::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	static Finfo* substrateShared[] =
	{
		new SrcFinfo( "reac", Ftype2< double, double >::global() ),
		new DestFinfo( "sub", Ftype1< double >::global(),
			RFCAST( &Enzyme::substrateFunc ) ),
	};
	static Finfo* enzShared[] =
	{
		new SrcFinfo( "reac", Ftype2< double, double >::global() ),
		new DestFinfo( "enz", Ftype1< double >::global(),
			RFCAST( &Enzyme::enzymeFunc ) ),
	};
	static Finfo* cplxShared[] =
	{
		new SrcFinfo( "reac", Ftype2< double, double >::global() ),
		new DestFinfo( "cplx", Ftype1< double >::global(),
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
		new ValueFinfo( "nInitComplex", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getNinitComplex ), 
			RFCAST( &Enzyme::setNinitComplex ),
			"This actually looks up the child molecule for the enz-substrate complex, if it exists. "
		        "Only expected to be used for initial conditions in a model."
		),
		new ValueFinfo( "concInitComplex", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getConcInitComplex ), 
			RFCAST( &Enzyme::setConcInitComplex )
		),
		new ValueFinfo( "x", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getX ), 
			RFCAST( &Enzyme::setX ),
			"X coordinate: for display."
		),
		new ValueFinfo( "y", 
			ValueFtype1< double >::global(),
			GFCAST( &Enzyme::getY ), 
			RFCAST( &Enzyme::setY ),
			"Y coordinate: for display."
		),
		new ValueFinfo( "xtree_textfg_req", 
			ValueFtype1< string >::global(),
			GFCAST( &Enzyme::getColor ), 
			RFCAST( &Enzyme::setColor ),
			"text Color : for display."
		),
		new ValueFinfo( "xtree_fg_req", 
			ValueFtype1< string >::global(),
			GFCAST( &Enzyme::getBgColor ), 
			RFCAST( &Enzyme::setBgColor ),
			"background Color : for display."
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "prd", Ftype2< double, double >::global() ),
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
		new DestFinfo( "rescaleRates", 
			Ftype1< double >::global(),
			RFCAST( &Enzyme::rescaleRates ) ),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "sub", substrateShared,
			sizeof( processShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "enz", enzShared,
			sizeof( processShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "cplx", cplxShared,
			sizeof( processShared ) / sizeof( Finfo* ) ),
	};

	// Schedule enzymes for slower clock, stage 1.
	static SchedInfo schedInfo[] = { { process, 0, 1 } };

	static string doc[] =
	{
		"Name", "Enzyme",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Enzyme: Irreversible enzymatic reaction that supports two forms of the "
				"Michaelis-Menten formulation for enzyme catalysis: "
				"E + S <======> E.S ------> E + P\n"
				"In this enzyme, the forward rate for the complex formation is "
				"k1, the backward rate is k2, and the final rate for product "
				"formation is k3. In terms of Michaelis-Menten parameters, "
				"k3 = kcat, and (k3 + k2)/k1 = Km. \n"
				"In all forms, the enzyme object should be considered as an "
				"enzymatic activity. It must be created in association with "
				"an enzyme molecule. The same enzyme molecule may have multiple "
				"activities, for example, on a range of substrates. "
				"In the explicit form (default) the enzyme substrate complex E.S "
				"is explictly created as a distinct molecular pool. This is "
				"perhaps more realistic in complex models where there are likely "
				"to be other substrates for this enzyme, and so enzyme "
				"saturation effects must be accounted for. However the complex "
				"molecule does not participate in any other reactions, which "
				"may itself be a poor assumption. If this is a serious concern "
				"then it is best to do the entire enzymatic process using elementary "
				"reactions. In the implicit form there is no actual enzyme-complex molecule. "
				"In this form the rate term is computed using the Michaelis-Menten form "
				"rate = kcat * [E] * [S] / ( Km + [S] )\n"
				"Here the opposite problem from above applies: There is no "
				"explicit complex, which means that the level of the free enzyme "
				"molecule is unaffected even near saturation. However, other "
				"reactions involving the enzyme do see the entire enzyme concentration. "
				"For the record, I regard the explicit formulation as more accurate for complex simulations.",
	};
	static  Cinfo enzymeCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		enzymeFinfos,
		sizeof(enzymeFinfos)/sizeof(Finfo *),
		ValueFtype1< Enzyme >::global(),
		schedInfo, 1
	);

	return &enzymeCinfo;
}

static const Cinfo* enzymeCinfo = initEnzymeCinfo();

static const Slot subSlot =
	initEnzymeCinfo()->getSlot( "sub.reac" );
static const Slot enzSlot =
	initEnzymeCinfo()->getSlot( "enz.reac" );
static const Slot cplxSlot =
	initEnzymeCinfo()->getSlot( "cplx.reac" );
static const Slot prdSlot =
	initEnzymeCinfo()->getSlot( "prd" );

///////////////////////////////////////////////////
// Enzyme class function definitions
///////////////////////////////////////////////////

Enzyme::Enzyme()
	: k1_(0.1), k2_(0.4), k3_(0.1),sk1_(1.0), Km_( 5.0 ),
		procFunc_( &Enzyme::implicitProcFunc ),
		x_( 0.0 ), y_( 0.0 )
{
	;
}
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void Enzyme::setK1( const Conn* c, double value )
{
	static_cast< Enzyme* >( c->data() )->k1_ = value;
}

double Enzyme::getK1( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->k1_;
}

void Enzyme::setK2( const Conn* c, double value )
{
	static_cast< Enzyme* >( c->data() )->k2_ = value;
}

double Enzyme::getK2( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->k2_;
}

void Enzyme::setK3( const Conn* c, double value )
{
	static_cast< Enzyme* >( c->data() )->k3_ = value;
}

double Enzyme::getK3( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->k3_;
}

double Enzyme::getKm( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->innerGetKm( e );
}
void Enzyme::setKm( const Conn* c, double value )
{
	static_cast< Enzyme* >( c->data() )->innerSetKm( c->target(), value );
}

/**
 * Handles Km in concentration units, typically uM
 */
double Enzyme::innerGetKm( Eref e )
{
	if ( Km_ <= 0.0 )
		return 0.0;
	unsigned int numSub = e.e->numTargets( subSlot.msg(), e.i );
	unsigned int numEnz = e.e->numTargets( enzSlot.msg(), e.i );
	if ( numEnz != 1 && !innerGetMode() ) {
		cout << "Error: Cannot get enzyme Km: Lacks enzyme complex.";
		return 1.0;
	}
	if ( numSub < 1 ) {
		cout << "Error: Cannot get enzyme Km: No substrate.";
		return 1.0;
	}
	double volScale = getVolScale( e );
	if ( numSub == 1 ) {
		return Km_ / volScale;
	} else {
		double ns = numSub;
		volScale = pow( volScale, ns );
		return Km_ / volScale;
	}
}

void Enzyme::innerSetKm( Eref e, double value )
{
	if ( value <= 0.0 ) {
		cout << "Error: Cannot set enzyme Km to negative value: " << 
			value << endl;
		return;
	}
	unsigned int numSub = e.e->numTargets( subSlot.msg(), e.i );
	unsigned int numEnz = e.e->numTargets( enzSlot.msg(), e.i );
	if ( numEnz != 1 && !innerGetMode() ) {
		cout << "Error: Cannot set enzyme Km: Lacks enzyme complex.";
		return;
	}
	if ( numSub < 1 ) {
		cout << "Error: Cannot set enzyme Km: No substrate.";
		return;
	}
	double volScale = getVolScale( e );
	volScale = pow( volScale, static_cast< double >( numSub ) );
	Km_ = value * volScale;
	k1_ = ( k2_ + k3_ ) / Km_;
}


double Enzyme::getKcat( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->k3_;
}
void Enzyme::setKcat( const Conn* c, double value )
{
	static_cast< Enzyme* >( c->data() )->innerSetKcat( value );
}
void Enzyme::innerSetKcat( double value )
{
	if ( value > 0.0 && k3_ > 0.0 ) {
		k2_ *= value / k3_;
		k1_ *= value / k3_;
		k3_ = value;
	}
}

/**
 * Enzyme mode is TRUE if it is an MM enzyme: implicit form.
 * By default we have an explicit enzyme with enz-sub complex: this is
 * mode FALSE.
 */
bool Enzyme::getMode( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->innerGetMode();
}
bool Enzyme::innerGetMode() const
{
	return ( procFunc_ == &Enzyme::implicitProcFunc );
}
void Enzyme::setMode( const Conn* c, bool value )
{
	static_cast< Enzyme* >( c->data() )->innerSetMode(
		       c->target(), value );
}
void Enzyme::innerSetMode( Eref e, bool mode )
{
	Km_ = ( k2_ + k3_ ) / k1_;
	if ( mode == innerGetMode() )
		return;
	if ( mode ) { 
		Id id = Neutral::getChildByName( e.e, e.e->name() + "_cplx" );
		if ( !id.bad() ) {
			Element* cplx = id();
			if ( cplx )
				set( cplx, "destroy" );
		}
		procFunc_ = &Enzyme::implicitProcFunc;
		sA_ = 0;
	} else { 
		procFunc_ = &Enzyme::explicitProcFunc;
		makeComplex( e );
	}
}

double Enzyme::getNinitComplex( Eref e )
{
	if ( getMode( e ) ) // No complex in implicit mm mode.
		return 0.0;

	Id cplxId = Neutral::getChildByName( e.e, e.e->name() + "_cplx" );
	assert( cplxId.good() );
	return Molecule::getNinit( cplxId.eref() );
}

void Enzyme::setNinitComplex( const Conn* c, double value )
{
	Eref e = c->target();
	if ( getMode( e ) ) {
		if ( value == 0.0 ) // No complex in implicit mm mode.
			return;
		// A bit of a hack here: we need to create a complex.
		setMode( c, 0 );
	}
	Id cplxId = Neutral::getChildByName( e.e, e.e->name() + "_cplx" );
	assert( cplxId.good() );
	set< double >( cplxId.eref(), "nInit", value );
}

double Enzyme::getConcInitComplex( Eref e )
{
	if ( getMode( e ) ) // No complex in implicit mm mode.
		return 0.0;

	Id cplxId = Neutral::getChildByName( e.e, e.e->name() + "_cplx" );
	assert( cplxId.good() );
	return Molecule::getConcInit( cplxId.eref() );
}

void Enzyme::setConcInitComplex( const Conn* c, double value )
{
	Eref e = c->target();
	if ( getMode( e ) ) {
		if ( value == 0.0 ) // No complex in implicit mm mode.
			return;
		// A bit of a hack here: we need to create a complex.
		setMode( c, 0 );
	}
	Id cplxId = Neutral::getChildByName( e.e, e.e->name() + "_cplx" );
	assert( cplxId.good() );
	set< double >( cplxId.eref(), "concInit", value );
}

///////////////////////////////////////////////////
// Shared message function definitions
///////////////////////////////////////////////////

void Enzyme::innerProcessFunc( Eref e )
{
	(this->*procFunc_)( e );
}

void Enzyme::processFunc( const Conn* c, ProcInfo p )
{
	Eref e = c->target();
	static_cast< Enzyme* >( c->data() )->innerProcessFunc( e );
}
void Enzyme::implicitProcFunc( Eref e )
{
	B_ = s_ * e_ * k3_ * sk1_ / ( s_ + Km_ );
	s_ = 1.0;
	send2< double, double >( e, subSlot, 0.0, B_ );
	send2< double, double >( e, prdSlot, B_, 0.0 );
}
void Enzyme::explicitProcFunc( Eref e )
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
		
void Enzyme::reinitFunc( const Conn* c, ProcInfo p )
{
	static_cast< Enzyme* >( c->data() )->innerReinitFunc( );
}

void Enzyme::substrateFunc( const Conn* c, double n )
{
	static_cast< Enzyme* >( c->data() )->s_ *= n;
}

void Enzyme::enzymeFunc( const Conn* c, double n )
{
	static_cast< Enzyme* >( c->data() )->e_ = n;
}

void Enzyme::complexFunc( const Conn* c, double n )
{
	static_cast< Enzyme* >( c->data() )->sA_ *= n;
	static_cast< Enzyme* >( c->data() )->pA_ *= n;
}
void Enzyme::setX( const Conn* c, double value )
{
	static_cast< Enzyme* >( c->data() )->x_ = value;
}

double Enzyme::getX( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->x_;
}

void Enzyme::setY( const Conn* c, double value )
{
	static_cast< Enzyme* >( c->data() )->y_ = value;
}

double Enzyme::getY( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->y_;
}

void Enzyme::setColor( const Conn* c, string value )
{
	static_cast< Enzyme* >( c->data() )->xtree_textfg_req_ = value;
}

string Enzyme::getColor( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->xtree_textfg_req_;
}


void Enzyme::setBgColor( const Conn* c, string value )
{
	static_cast< Enzyme* >( c->data() )->xtree_fg_req_ = value;
}

string Enzyme::getBgColor( Eref e )
{
	return static_cast< Enzyme* >( e.data() )->xtree_fg_req_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Enzyme::intramolFunc( const Conn* c, double n )
{
	static_cast< Enzyme* >( c->data() )->innerIntramolFunc( n );
}
void Enzyme::innerIntramolFunc( double n )
{
	if ( n > 0 )
		sk1_ = 1.0 / n;
	else
		sk1_ = 1.0;
}
void Enzyme::scaleKmFunc( const Conn* c, double k )
{
	static_cast< Enzyme* >( c->data() )->innerScaleKmFunc( k );
}
void Enzyme::innerScaleKmFunc( double k )
{
	if ( k > 0 )
		s_ /= k;
	else
		cout << "Error: Enzyme::scaleKm msg: negative k = " <<
			k << endl;
}
void Enzyme::scaleKcatFunc( const Conn* c, double k )
{
	static_cast< Enzyme* >( c->data() )->pA_ *= k;
}

/**
 * Ratio is ratio of new vol to old vol.
 * Need to scale k1_ and hence Km_ according to numSub. k2 and k3
 * have units of time only.
 */
void Enzyme::rescaleRates( const Conn* c, double ratio )
{
	Eref e = c->target();
	static_cast< Enzyme* >( e.data() )->innerRescaleRates( e, ratio );
}

void Enzyme::innerRescaleRates( Eref e, double ratio )
{
	assert( ratio > 0.0 );
	unsigned int numSub = e.e->numTargets( subSlot.msg(), e.i );
	unsigned int numEnz = e.e->numTargets( enzSlot.msg(), e.i );
	if ( numEnz != 1 && !innerGetMode() ) {
		cout << "Error: Cannot set enzyme Km: Lacks enzyme complex.";
		return;
	}
	if ( numSub < 1 ) {
		cout << "Error: Cannot set enzyme Km: No substrate.";
		return;
	}
	ratio = pow( ratio, static_cast< double >( numSub ) );
	Km_ = Km_ * ratio;
	k1_ = ( k2_ + k3_ ) / Km_;
}

///////////////////////////////////////////////////////
// Other func definitions
///////////////////////////////////////////////////////

void Enzyme::makeComplex( Eref er )
{
	Element* e = er.e;
	// static const Finfo* cplxSrcFinfo = enzymeCinfo->findFinfo( "cplx" );
	string cplxName = e->name() + "_cplx";
	Id id = Neutral::getChildByName( e, cplxName );
	if ( !id.bad() )
		return;

	double vol = 0.0;

	Id parentId = Neutral::getParent( e );
	assert( !parentId.bad() );
	Element* parent = parentId();

	bool ret = get< double >( parent, "volumeScale", vol );
	assert( ret );
	Element* complex = Neutral::create( "Molecule", cplxName, e->id(),
		Id::scratchId() );

	ret = er.add( "cplx", complex, "reac" );
	// ret = cplxSrcFinfo->add( e, complex, complex->findFinfo( "reac" ) );
	assert( ret );
}


#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"
#include "Molecule.h"

void testEnzyme()
{
	static double explicitSub[] = {
		0.9999, 0.92394, 0.879014, 0.849819, 0.828738,
		0.811911, 0.797361, 0.784069, 0.771509, 0.759406
	};
	static double explicitPrd[] = {
		0, 0.00401767, 0.013213, 0.0249653, 0.0379434,
		0.0514658, 0.0651816, 0.0789104, 0.0925594, 0.106081,
	};
	static double explicitEnz[] = {
		0.9999, 0.927958, 0.892226, 0.874783, 0.86668,
		0.863375, 0.86254, 0.862976, 0.864064, 0.865483,
	};

	static double implicitSub[] = {
		0.999983, 0.983432, 0.967111, 0.951017, 0.935149,
		0.919504, 0.904081, 0.888878, 0.873892, 0.859122,
	};
	static double implicitPrd[] = {
		0, 0.0165678, 0.0328894, 0.0489834, 0.0648518,
		0.0804965, 0.0959196, 0.111123, 0.126109, 0.140879,
	};
	static double implicitEnz[] = {
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
	};


	cout << "\nTesting Enzyme" << flush;

	Element* n = Neutral::create( "KinCompt", "n", Element::root()->id(),
		Id::scratchId() );
	Element* sub = Neutral::create( "Molecule", "sub", n->id(),
		Id::scratchId() );
	ASSERT( sub != 0, "creating molecule" );
	Element* prd = Neutral::create( "Molecule", "prd", n->id(),
		Id::scratchId() );
	ASSERT( prd != 0, "creating molecule" );
	Element* enzMol = Neutral::create( "Molecule", "enzMol", n->id(),
		Id::scratchId() );
	ASSERT( enzMol != 0, "creating molecule" );
	Element* enz = Neutral::create( "Enzyme", "enz", enzMol->id(),
		Id::scratchId() );
	ASSERT( enz != 0, "creating enzyme" );

	bool ret;

	ProcInfoBase p;
	SetConn csub( sub, 0 );
	SetConn cprd( prd, 0 );
	SetConn cenzMol( enzMol, 0 );
	SetConn cenz( enz, 0 );
	p.dt_ = 0.001;
	set< double >( sub, "concInit", 1.0 );
	set< int >( sub, "mode", 0 );
	set< double >( prd, "concInit", 0.0 );
	set< int >( prd, "mode", 0 );
	set< double >( enzMol, "concInit", 1.0 );
	set< int >( enzMol, "mode", 0 );
	set< double >( enz, "k1", 0.1 );
	set< double >( enz, "k2", 0.4 );
	set< double >( enz, "k3", 0.1 );
	ret = set< bool >( enz, "mode", 0 );
	ASSERT( ret, "setting enz mode" );
	Id cplxId = Neutral::getChildByName( enz, "enz_cplx" );
	ASSERT( !cplxId.bad(), "making Enzyme cplx" );
	Element* cplx = cplxId();
	SetConn ccplx( cplx, 0 );

	// ret = sub->findFinfo( "reac" )->add( sub, enz, enz->findFinfo( "sub" ));
	// ret = enz->findFinfo( "prd" )->add( enz, prd, prd->findFinfo( "prd" ) );
	// ret = enzMol->findFinfo( "reac" )->add( enzMol, enz, enz->findFinfo( "enz" ) );
	ret = Eref( sub ).add( "reac", enz, "sub" );
	ASSERT( ret, "adding substrate msg" );

	ret = Eref( enz ).add( "prd", prd, "prd" );
	ASSERT( ret, "adding product msg" );

	ret = Eref( enzMol ).add( "reac", enz, "enz" );
	ASSERT( ret, "adding enzMol msg" );

	// First, test charging curve for a single compartment
	// We want our charging curve to be a nice simple exponential
	// n = 0.5 + 0.5 * exp( - t * 0.2 );
	double delta = 0.0;
	Enzyme::reinitFunc( &cenz, &p );
	Molecule::reinitFunc( &csub, &p );
	Molecule::reinitFunc( &cprd, &p );
	Molecule::reinitFunc( &cenzMol, &p );
	unsigned int i = 0;
	unsigned int j = 0;
	cout << " ";
	for ( p.currTime_ = 0.0; p.currTime_ < 9.5; p.currTime_ += p.dt_ ) 
	{
		double nprd = Molecule::getN( prd );
		double nsub = Molecule::getN( sub );
		double nenz = Molecule::getN( enzMol );
		double temp = 0;
		if ( i++ % 1000 == 0 ) {
			temp = nprd - explicitPrd[ j ]; delta += temp * temp;
			temp = nsub - explicitSub[ j ]; delta += temp * temp;
			temp = nenz - explicitEnz[ j++ ]; delta += temp * temp;
		}
		//	cout << p.currTime_ << "	" << nprd << endl;

		Enzyme::processFunc( &cenz, &p );
		Molecule::processFunc( &ccplx, &p );
		Molecule::processFunc( &csub, &p );
		Molecule::processFunc( &cprd, &p );
		Molecule::processFunc( &cenzMol, &p );
	}
	ASSERT( delta < 5.0e-6, "Testing molecule and enzyme" );

	// Check set and get of nInitComplex
	ret = set< double >( Eref( enz ), "nInitComplex", 1234.5 );
	ASSERT( ret, "Setting nInitComplex" );
	double ncplx = 0.0;
	ret = get< double >( Eref( cplx ), "nInit", ncplx );
	ASSERT( fabs( ncplx - 1234.5 ) < 1e-9, "setting nInitComplex" );

	ret = set< double >( Eref( cplx ), "nInit", 3.1415 );
	ASSERT( ret, "Getting nInitComplex" );
	ret = get< double >( Eref( enz ), "nInitComplex", ncplx );
	ASSERT( fabs( ncplx - 3.1415 ) < 1e-9, "getting nInitComplex" );

	////////////////////////////////////////////////////////////////////
	// Testing implicit mode.
	////////////////////////////////////////////////////////////////////
	ret = set< bool >( enz, "mode", 1 );
	ASSERT( ret, "setting enz mode" );
	cplxId = Neutral::getChildByName( enz, "enz_cplx" );
	ASSERT( cplxId.bad() , "removing Enzyme cplx" );

	ret = set< bool >( enz, "mode", 0 );
	ASSERT( ret, "setting enz mode" );
	cplxId = Neutral::getChildByName( enz, "enz_cplx" );
	ASSERT( !cplxId.bad(), "adding Enzyme cplx" );

	ret = set< bool >( enz, "mode", 1 );
	ASSERT( ret, "setting enz mode" );
	cplxId = Neutral::getChildByName( enz, "enz_cplx" );
	ASSERT( cplxId.bad(), "removing Enzyme cplx" );

	Enzyme::reinitFunc( &cenz, &p );
	Molecule::reinitFunc( &csub, &p );
	Molecule::reinitFunc( &cprd, &p );
	Molecule::reinitFunc( &cenzMol, &p );
	i = 0;
	j = 0;
	delta = 0.0;
	for ( p.currTime_ = 0.0; p.currTime_ < 9.5; p.currTime_ += p.dt_ ) 
	{
		double nprd = Molecule::getN( prd );
		double nsub = Molecule::getN( sub );
		double nenz = Molecule::getN( enzMol );
		double temp = 0;
		if ( i++ % 1000 == 0 ) {
			temp = nprd - implicitPrd[ j ]; delta += temp * temp;
			temp = nsub - implicitSub[ j ]; delta += temp * temp;
			temp = nenz - implicitEnz[ j++ ]; delta += temp * temp;
		}
		//	cout << p.currTime_ << "	" << nprd << endl;

		Enzyme::processFunc( &cenz, &p );
		Molecule::processFunc( &csub, &p );
		Molecule::processFunc( &cprd, &p );
		Molecule::processFunc( &cenzMol, &p );
	}
	ASSERT( delta < 5.0e-6, "Testing molecule and enzyme" );

	// Check volume rescaling.
	Eref enzE( enz, 0 );
	double k1 = Enzyme::getK1( enzE );
	double k2 = Enzyme::getK2( enzE );
	double k3 = Enzyme::getK3( enzE );
	double Km = Enzyme::getKm( enzE );
	double nenz = Molecule::getN( enzMol );
	set< double >( n, "volume", 1e-15 );

	double num = Enzyme::getK1( enzE ); // Should be same as k1 * 6e5.
	ASSERT( fabs( 1.0 - num * Molecule::NA * 1e-18 / k1 ) < 1.0e-6, "enz vol rescaling" );
	num = Enzyme::getK2( enzE ); // Should be same as k2
	ASSERT( fabs( num - k2 ) < 1.0e-6, "enz vol rescaling" );
	num = Enzyme::getK3( enzE ); // Should be same as k3
	ASSERT( fabs( num - k3 ) < 1.0e-6, "enz vol rescaling" );
	num = Enzyme::getKm( enzE ); // Should be same as Km.
	ASSERT( fabs( num - Km ) < 1.0e-6, "enz vol rescaling" );

	num = Molecule::getN( enzMol ); // Should be same as nenz scaled by vol.
	ASSERT( fabs( num - nenz * Molecule::NA * 1e-18 ) < 1.0e-2, "enzmol num rescaling" );
	num = Molecule::getConc( enzMol ); // Should be same as nenz.
	ASSERT( fabs( num - nenz ) < 1.0e-6, "enzmol conc rescaling" );

	// Get rid of all the compartments.
	set( n, "destroy" );
}
#endif
