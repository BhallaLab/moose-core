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
#include <math.h>
#include "Molecule.h"
#include "KinCompt.h" // for getVolScale and setParentalVolScale

const double Molecule::EPSILON = 1.0e-15;
const double Molecule::NA = 6.02214179e23; // Avogadro's constant

const Cinfo* initMoleculeCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &Molecule::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &Molecule::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ),
		"Shared message to handle scheduling. 'process' calls the update operation on the molecule. "
		"'reinit' reinitializes the molecule concentration and updates attached reactions." );
	static Finfo* reacShared[] =
	{
		new DestFinfo( "reac", Ftype2< double, double >::global(),
			RFCAST( &Molecule::reacFunc ),
			"handles A and B (increment and decrement) values to n" ),
		new SrcFinfo( "n", Ftype1< double >::global(), 
		"sends out n, number of molecules, every timestep"
		)
	};
	static Finfo* extentShared[] =
	{
		new DestFinfo( "returnExtent", Ftype2< double, unsigned int >::global(),
			RFCAST( &Molecule::extentFunc ) ),
		new SrcFinfo( "requestExtent", Ftype0::global() )
	};
	static Finfo* moleculeFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "n", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getN ), 
			RFCAST( &Molecule::setN ),
			"Number of molecules."
		),
		new ValueFinfo( "nInit", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getNinit ), 
			RFCAST( &Molecule::setNinit ),
			"Initial number of molecules in this pool. Assigned to n every time the reInit function is "
			"called. Also sets the value of n the entire time the molecule is in buffered mode (mode 4)"
		),
		new ValueFinfo( "volumeScale",  // Deprecated. Used for BC only.
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getVolumeScale ), 
			RFCAST( &Molecule::setVolumeScale ),
			"Deprecated field. Used for backward compatibility for legacy kkit models. Defines the "
			"ratio of 'n' to 'conc'. When volumeScale is assigned a value, it looks up the nearest "
			"ancestor KinCompt to see if it can change the volume of the compartment in which the "
			"molecule is embedded.Obviously lots of scope for conflicting assignments here, hence "
			"DO NOT USE. Instead assign volume of the parent KinCompt."
		),
		new ValueFinfo( "mode", 
			ValueFtype1< int >::global(),
			GFCAST( &Molecule::getMode ), 
			RFCAST( &Molecule::setMode ),
			"Mode of functioning of molecule object.\n"
			"0: normal calculation mode.\n1, 2: deprecated.\n4: buffered mode. "
			"In buffered mode the value of n is always set to nInit."
		),
		new ValueFinfo( "slave_enable", 
			ValueFtype1< int >::global(),
			GFCAST( &Molecule::getMode ), 
			RFCAST( &Molecule::setMode ),
			"Alias for 'mode' field, used for backward compatibility with kkit. Deprecated. "
			"Mode of functioning of molecule object.\n0: normal calculation mode.\n1, 2: deprecated.\n"
			"4: buffered mode. In buffered mode the value of n is always set to nInit."
		),
		new ValueFinfo( "conc", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getConc ), 
			RFCAST( &Molecule::setConc ),
			"Concentration of molecule. Derived on the fly from n and from looking up parental KinCompt "
			"to obtain volume."
		),
		new ValueFinfo( "concInit", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getConcInit ), 
			RFCAST( &Molecule::setConcInit ),
			"Initial concentration of molecules in this pool. Derived on the fly from nInit and requires "
			"a lookup of parental KinCompt to obtain volume."
		),
		new ValueFinfo( "D", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getD ), 
			RFCAST( &Molecule::setD ),
			"Diffusion constant of molecule."
		),
		new ValueFinfo( "x", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getX ), 
			RFCAST( &Molecule::setX ),
			"X coordinate: for display."
		),
		new ValueFinfo( "y", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getY ), 
			RFCAST( &Molecule::setY ),
			"Y coordinate: for display."
		),
		new ValueFinfo( "xtree_textfg_req", 
			ValueFtype1< string >::global(),
			GFCAST( &Molecule::getColor ), 
			RFCAST( &Molecule::setColor ),
			"text color: for display."
		),
		new ValueFinfo( "xtree_fg_req", 
			ValueFtype1< string >::global(),
			GFCAST( &Molecule::getBgColor ), 
			RFCAST( &Molecule::setBgColor ),
			"text color: for display."
		),

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "nSrc", Ftype1< double >::global(),
		"Sends out 'n', number of molecules, every timestep" ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "prd",
			Ftype2< double, double >::global(),
			RFCAST( &Molecule::reacFunc ),
			"This is a backward compat feature to handle one-ended input from enzymes, but using the "
			"same reacFunc as the shared message for reac."
		),
	
		new DestFinfo( "sumTotal",
			Ftype1< double >::global(),
			RFCAST( &Molecule::sumTotalFunc ),
			"Sums up input values to assign to n."
		),

		new DestFinfo( "rescaleVolume",
			Ftype1< double >::global(),
			RFCAST( &Molecule::rescaleFunc ),
			"Increases volume by specified factor, so it scales up n and nInit. Should be called "
			"primarily by KinCompt to manage system volumes"
		),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "reac", reacShared,
			sizeof( reacShared ) / sizeof( Finfo* ),
			"Shared message to a Reaction or Enzyme object. Receives A and B terms,which are "
			"increments and decrements to # of molecules.Sends n, the number of molecules"),
		new SharedFinfo( "extent", extentShared,
			sizeof( extentShared ) / sizeof( Finfo* ),
			"Deprecated. Manages volume control through interacting with KinCompt: the kinetic "
			"compartment object, which in turn interfaces with surfaces" ),
	};

	// Schedule molecules for the slower clock, stage 0.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static string doc[] =
	{
		"Name", "Molecule",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Molecule: Pool of molecules.",
		"Details", "Internally manages number of molecules, n. Initial condition set by nInit.The molecule "
		"can be buffered, in which case n is always = nInit.Does NOT handle volumes: this is managed by the "
		"nearest ancestral KinCompt (Kinetic compartment) object. Calculates concentration (conc) on the "
		"fly from compartmental volume when needed. Has a default explicit Exponential Euler integration "
		"method, but is normally solved by a special solver such as ksolve (ODEs, uses GSL), or gsolve "
		"(Gillespie Stochastic Simulation Algorithm) or Smoldyn (Steven Andrews' Smoluchowski dynamics "
		"Monte Carlo simulator)"
	};
	static Cinfo moleculeCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		moleculeFinfos,
		sizeof( moleculeFinfos )/sizeof(Finfo *),
		ValueFtype1< Molecule >::global(),
			schedInfo, 1
	);

	return &moleculeCinfo;
}

static const Cinfo* moleculeCinfo = initMoleculeCinfo();

static const Slot reacSlot = initMoleculeCinfo()->getSlot( "reac.n" );
static const Slot nSlot = initMoleculeCinfo()->getSlot( "nSrc" );
static const Slot extentSlot =
	initMoleculeCinfo()->getSlot( "extent.requestExtent" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Molecule::Molecule()
	:
	nInit_( 0.0 ), 
	volumeScale_( 1.0 ),
	n_( 0.0 ),
	mode_( 0 ),
	total_( 0.0 ),
	A_( 0.0 ),
	B_( 0.0 ),
	D_( 0.0 ),
	x_( 0.0 ),
	y_( 0.0 )
{
		;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void Molecule::setNinit( const Conn* c, double value )
{
	if ( value >= 0.0 )
		static_cast< Molecule* >( c->data() )->nInit_ = value;
}

void Molecule::localSetNinit( double value )
{
	if ( value >= 0.0 )
		nInit_ = value;
}

double Molecule::getNinit( Eref e )
{
	return static_cast< Molecule* >( e.data() )->nInit_;
}

void Molecule::setVolumeScale( const Conn* c, double value )
{
	setParentalVolScale( c->target(), value );
/*
	if ( value >= 0.0 )
		static_cast< Molecule* >( c->data() )->volumeScale_ = value;
	double volScale = getVolScale( c->target() );
	if ( fabs( value - volScale ) / (value + volScale ) > 1e-6 )
		cout << "Warning: Molecule::setVolumeScale: Mismatch in deprecated assignment to\nvolScale: " << volScale << " != " << value << endl; 
		*/
}

double Molecule::getVolumeScale( Eref e )
{
	return getVolScale( e );
	// return static_cast< Molecule* >( e.data() )->volumeScale_;
}

void Molecule::setN( const Conn* c, double value )
{
	if ( value >= 0.0 )
		static_cast< Molecule* >( c->data() )->n_ = value;
}

double Molecule::getN( Eref e )
{
	return static_cast< Molecule* >( e.data() )->n_;
}

void Molecule::setMode( const Conn* c, int value )
{
	static_cast< Molecule* >( c->data() )->mode_ = value;
}

int Molecule::getMode( Eref e )
{
	return static_cast< Molecule* >( e.data() )->localGetMode( e );
}

int Molecule::localGetMode( Eref e )
{
	static const Finfo* sumTotFinfo = 
			Cinfo::find( "Molecule" )->findFinfo( "sumTotal" );

	unsigned int n = e.e->numTargets( sumTotFinfo->msg(), e.i );
	if ( mode_ == 0 && n > 0 )
		mode_ = 1;
	else if ( mode_ == 1 && n == 0 )
		mode_ = 0;

	return mode_;
}

/*
double Molecule::localGetConc() const
{
			if ( volumeScale_ > 0.0 )
				return n_ / volumeScale_ ;
			else
				return n_;
}
*/
double Molecule::getConc( Eref e )
{
	double volScale = getVolScale( e );
	return static_cast< Molecule* >( e.data() )->n_ / volScale;
	// return static_cast< Molecule* >( e.data() )->localGetConc();
}

/*
void Molecule::localSetConc( double value ) {
			if ( volumeScale_ > 0.0 )
				n_ = value * volumeScale_ ;
			else
				n_ = value;
}
*/

void Molecule::setConc( const Conn* c, double value )
{
	if ( value >= 0.0 ) {
		double volScale = getVolScale( c->target() );
		static_cast< Molecule* >( c->data() )->n_ = value * volScale;
		// static_cast< Molecule* >( c->data() )->localSetConc( value );
	}
}

/*
double Molecule::localGetConcInit() const
{
			if ( volumeScale_ > 0.0 )
				return nInit_ / volumeScale_ ;
			else
				return nInit_;
}
*/
double Molecule::getConcInit( Eref e )
{
	double volScale = getVolScale( e );
	return static_cast< Molecule* >( e.data() )->nInit_ / volScale;
	// return static_cast< Molecule* >( e.data() )->localGetConcInit();
}

/*
void Molecule::localSetConcInit( double value ) {
			if ( volumeScale_ > 0.0 )
				nInit_ = value * volumeScale_ ;
			else
				nInit_ = value;
}
*/
void Molecule::setConcInit( const Conn* c, double value )
{
	if ( value >= 0.0 ) {
		double volScale = getVolScale( c->target() );
		static_cast< Molecule* >( c->data() )->nInit_ = value * volScale;
	}
	// static_cast< Molecule* >( c->data() )->localSetConcInit( value );
}

void Molecule::setD( const Conn* c, double value )
{
	if ( value >= 0.0 )
		static_cast< Molecule* >( c->data() )->D_ = value;
	else
		static_cast< Molecule* >( c->data() )->D_ = 1.0;
}

double Molecule::getD( Eref e )
{
	return static_cast< Molecule* >( e.data() )->D_;
}
void Molecule::setX( const Conn* c, double value )
{
	static_cast< Molecule* >( c->data() )->x_ = value;
}

double Molecule::getX( Eref e )
{
	return static_cast< Molecule* >( e.data() )->x_;
}

void Molecule::setY( const Conn* c, double value )
{
	static_cast< Molecule* >( c->data() )->y_ = value;
}

double Molecule::getY( Eref e )
{
	return static_cast< Molecule* >( e.data() )->y_;
}

void Molecule::setColor( const Conn* c, string value )
{
	static_cast< Molecule* >( c->data() )->xtree_textfg_req_ = value;
}

string Molecule::getColor( Eref e )
{
	return static_cast< Molecule* >( e.data() )->xtree_textfg_req_;
}
void Molecule::setBgColor( const Conn* c, string value )
{
	static_cast< Molecule* >( c->data() )->xtree_fg_req_ = value;
}

string Molecule::getBgColor( Eref e )
{
	return static_cast< Molecule* >( e.data() )->xtree_fg_req_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


void Molecule::reacFunc( const Conn* c, double A, double B )
{
	static_cast< Molecule* >( c->data() )->A_ += A;
	static_cast< Molecule* >( c->data() )->B_ += B;
}

void Molecule::sumTotalFunc( const Conn* c, double n )
{
	Molecule* m = static_cast< Molecule* >( c->data() );
	m->total_ += n;
}

void Molecule::rescaleFunc( const Conn* c, double ratio )
{
	Molecule* m = static_cast< Molecule* >( c->data() );
	m->nInit_ *= ratio;
	m->n_ *= ratio;
}

/*
void Molecule::sumTotalConcFunc( const Conn* c, double conc )
{
	static_cast< Molecule* >( c->data() )->total_ += conc * volumeScale_;
}
*/

/*
void Molecule::sumProcessFuncLocal( )
{
		n_ = total_;
		total_ = 0.0;
}
void Molecule::sumProcessFunc( const Conn* c, ProcInfo info )
{
	static_cast< Molecule* >( c->data() )->sumProcessFuncLocal();
}
*/

void Molecule::reinitFunc( const Conn* c, ProcInfo info )
{
	static_cast< Molecule* >( c->data() )->reinitFuncLocal( c->target() );
}
void Molecule::reinitFuncLocal( Eref e )
{
	static const Finfo* sumTotFinfo = 
			Cinfo::find( "Molecule" )->findFinfo( "sumTotal" );
	unsigned int numTargets = e.e->numTargets( sumTotFinfo->msg() );

	A_ = B_ = total_ = 0.0;
	n_ = nInit_;
	if ( mode_ == 0 && numTargets > 0 )
		mode_ = 1;
	else if ( mode_ == 1 && numTargets == 0 )
		mode_ = 0;
	send1< double >( e, reacSlot, n_ );
	send1< double >( e, nSlot, n_ );
	send0( e, extentSlot ); // Request volume from the compartment.
							// The compartment will also push the
							// latest volume in at appropriate times.
}

void Molecule::processFunc( const Conn* c, ProcInfo info )
{
	Eref e = c->target();
	static_cast< Molecule* >( c->data() )->processFuncLocal( e, info );
}
void Molecule::processFuncLocal( Eref e, ProcInfo info )
{
			if ( mode_ == 0 ) {
				if ( n_ > EPSILON && B_ > EPSILON ) {
					double C = exp( -B_ * info->dt_ / n_ );
					n_ *= C + ( A_ / B_ ) * ( 1.0 - C );
				} else {
					n_ += ( A_ - B_ ) * info->dt_;
				}
				A_ = B_ = 0.0;
			} else if ( mode_ == 1 ) {
				n_ = total_;
				total_ = 0.0;
			} else if ( mode_ == 2 ) {
				n_ = total_ * volumeScale_;
				total_ = 0.0;
			} else { 
				n_ = nInit_;
			}
			send1< double >( e, reacSlot, n_ );
			send1< double >( e, nSlot, n_ );
}

void Molecule::extentFunc( const Conn* c, double size, unsigned int dim )
{
	static_cast< Molecule* >( c->data() )->extentFuncLocal( c->target(), size, dim );
}

void Molecule::extentFuncLocal(  Eref e, double size, unsigned int dim)
{
	// Assume that the units of conc are uM.
	if ( size > 0.0 ) {
		if ( dim == 3 ) // Regular 3-d scaling.
			//volumeScale_ = size / 6.023e20;
			volumeScale_ = size / 6.02214199e20;
		else // Here a quandary: Do we use areal density for readout?
			volumeScale_ = 1.0;
	} else {
		volumeScale_ = 1.0;
	}
}



#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"
#include "Reaction.h"

void testMolecule()
{
	cout << "\nTesting Molecule" << flush;

	Element* n = Neutral::create( "KinCompt", "n", Element::root()->id(),
		Id::scratchId() );
	Element* m0 = Neutral::create( "Molecule", "m0", n->id(),
		Id::scratchId() );
	ASSERT( m0 != 0, "creating molecule" );
	Element* m1 = Neutral::create( "Molecule", "m1", n->id(),
		Id::scratchId() );
	ASSERT( m1 != 0, "creating molecule" );
	Element* r0 = Neutral::create( "Reaction", "r0", n->id(),
		Id::scratchId() );
	ASSERT( r0 != 0, "creating reaction" );

	bool ret;

	ProcInfoBase p;
	SetConn cm0( m0, 0 );
	SetConn cm1( m1, 0 );
	SetConn cr0( r0, 0 );
	p.dt_ = 0.001;
	ret = set< double >( m0, "concInit", 1.0 );
	ASSERT( ret, "mol conc assignment" );
	double num = 0.0;
	ret = get< double >( m0, "nInit", num );
	ASSERT( fabs( num - 1.0 ) < 1.0e-6, "mol conc assignment" );
	set< int >( m0, "mode", 0 );
	set< double >( m1, "concInit", 0.0 );
	set< int >( m1, "mode", 0 );
	set< double >( r0, "kf", 0.1 );
	set< double >( r0, "kb", 0.1 );

	// ret = m0->findFinfo( "reac" )->add( m0, r0, r0->findFinfo( "sub" ) );
	ret = Eref( m0 ).add( "reac", r0, "sub" );
	ASSERT( ret, "adding msg 0" );
	// ret = m1->findFinfo( "reac" )->add( m1, r0, r0->findFinfo( "prd" ) );
	ret = Eref( m1 ).add( "reac", r0, "prd" );
	ASSERT( ret, "adding msg 1" );

	// First, test charging curve for a single compartment
	// We want our charging curve to be a nice simple exponential
	// n = 0.5 + 0.5 * exp( - t * 0.2 );
	double delta = 0.0;
	double n0 = 1.0;
	double n1 = 0.0;
	double y = 0.0;
	double y0 = 0.0;
	double y1 = 0.0;
	double tau = 5.0;
	double nMax = 0.5;
	Reaction::reinitFunc( &cr0, &p );
	Molecule::reinitFunc( &cm0, &p );
	Molecule::reinitFunc( &cm1, &p );
	for ( p.currTime_ = 0.0; p.currTime_ < 20.0; p.currTime_ += p.dt_ ) 
	{
		n0 = Molecule::getN( m0 );
		n1 = Molecule::getN( m1 );
//		cout << p.currTime_ << "	" << n1 << endl;

		y = nMax * exp( -p.currTime_ / tau );
		y0 = 0.5 + y;
		y1 = 0.5 - y;
		delta += ( n0 - y0 ) * ( n0 - y0 );
		delta += ( n1 - y1 ) * ( n1 - y1 );
		Reaction::processFunc( &cr0, &p );
		Molecule::processFunc( &cm0, &p );
		Molecule::processFunc( &cm1, &p );
	}
	ASSERT( delta < 5.0e-6, "Testing molecule and reacn" );

	// Check volume rescaling.
	n0 = Molecule::getN( m0 );
	n1 = Molecule::getN( m1 );
	set< double >( n, "volume", 1e-15 );
	num = Molecule::getConc( m0 ); // Should be same as n0.
	ASSERT( fabs( num - n0 ) < 1.0e-6, "mol num rescaling" );
	num = Molecule::getConc( m1 ); // Should be same as n0.
	ASSERT( fabs( num - n1 ) < 1.0e-6, "mol num rescaling" );
	
	num = Molecule::getN( m0 );
	ASSERT( fabs( num - n0 * Molecule::NA * 1e-18 ) < 1.0e-2, "mol num rescaling" );
	num = Molecule::getN( m1 );
	ASSERT( fabs( num - n1 * Molecule::NA * 1e-18 ) < 1.0e-2, "mol num rescaling" );

	num = Molecule::getNinit( m0 );
	ASSERT( fabs( num - Molecule::NA * 1e-18 ) < 1.0e-2, "mol num rescaling" );
	num = Molecule::getConcInit( m0 );
	ASSERT( fabs( num - 1.0 ) < 1.0e-6, "mol num rescaling" );

	// Get rid of all the compartments.
	set( n, "destroy" );
}
#endif
