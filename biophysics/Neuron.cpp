/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"
#include "ReadCell.h"
#include "../utility/Vec.h"
#include "SwcSegment.h"
#include "Neuron.h"

#include "../external/muparser/muParser.h"

const Cinfo* Neuron::initCinfo()
{
	/////////////////////////////////////////////////////////////////////
	// ValueFinfos
	/////////////////////////////////////////////////////////////////////
	static ValueFinfo< Neuron, double > RM( "RM",
		"Membrane resistivity, in ohm.m^2. Default value is 1.0.",
		&Neuron::setRM,
		&Neuron::getRM
	);
	static ValueFinfo< Neuron, double > RA( "RA",
		"Axial resistivity of cytoplasm, in ohm.m. Default value is 1.0.",
		&Neuron::setRA,
		&Neuron::getRA
	);
	static ValueFinfo< Neuron, double > CM( "CM",
		"Membrane Capacitance, in F/m^2. Default value is 0.01",
		&Neuron::setCM,
		&Neuron::getCM
	);
	static ValueFinfo< Neuron, double > Em( "Em",
		"Resting membrane potential of compartments, in Volts. "
		"Default value is -0.065.",
		&Neuron::setEm,
		&Neuron::getEm
	);
	static ValueFinfo< Neuron, double > theta( "theta",
		"Angle to rotate cell geometry, around long axis of neuron. "
		"Think Longitude. Units are radians. "
		"Default value is zero, which means no rotation. ",
		&Neuron::setTheta,
		&Neuron::getTheta
	);
	static ValueFinfo< Neuron, double > phi( "phi",
		"Angle to rotate cell geometry, around elevation of neuron. "
		"Think Latitude. Units are radians. "
		"Default value is zero, which means no rotation. ",
		&Neuron::setPhi,
		&Neuron::getPhi
	);

	static ValueFinfo< Neuron, string > sourceFile( "sourceFile",
		"Name of source file from which to load a model. "
		"Accepts swc and dotp formats at present. "
		"Both these formats require that the appropriate channel "
		"definitions should have been loaded into /library. ",
		&Neuron::setSourceFile,
		&Neuron::getSourceFile
	);

	static ValueFinfo< Neuron, double > compartmentLengthInLambdas( 
		"compartmentLengthInLambdas",
		"Electrotonic length to use for the largest compartment in the "
		"model. Used to define subdivision of branches into compartments. "
		"For example, if we set *compartmentLengthInLambdas*  to 0.1, "
		"and *lambda* (electrotonic length) is 250 microns, then it "
		"sets the compartment length to 25 microns. Thus a dendritic "
		"branch of 500 microns is subdivided into 20 commpartments. "
		"If the branch is shorter than *compartmentLengthInLambdas*, "
		"then it is not subdivided. "
		"If *compartmentLengthInLambdas* is set to 0 then the original "
		"compartmental structure of the model is preserved. "
		" Note that this routine does NOT merge branches, even if "
		"*compartmentLengthInLambdas* is bigger than the branch. "
		"While all this subdivision is being done, the Neuron class "
		"preserves as detailed a geometry as it can, so it can rebuild "
		"the more detailed version if needed. "
		"Default value of *compartmentLengthInLambdas* is 0. ",
		&Neuron::setCompartmentLengthInLambdas,
		&Neuron::getCompartmentLengthInLambdas
	);

	static ValueFinfo< Neuron, vector< string > > channelDistribution( 
		"channelDistribution",
		"Specification for channel distribution on this neuron. "
		"Each entry in the specification is a triplet of strings: \n"
		"	(name, path, function) \n"
		" which are collated into a 3N vector of strings to specify N "
		" channel distributions. The string arguments for each spec are: \n"
		"chanName, pathOnCell, function( r, L, len, dia ) \n"
		"The function uses arguments:\n"
		"	r: geometrical distance from soma, measured along dendrite, in metres.\n"
		"	L: electrotonic distance (# of lambdas) from soma, along dend. No unts.\n"
		"	len: length of compartment, in metres.\n"
		"	dia: for diameter of compartment, in metres.\n"
		"For Channels, the function does Gbar = func( r, L, len, dia).\n"
		"For CaConc, the function does B = func( r, L, len, dia).\n"
		"In both cases, if func() < 0 then the chan/CaConc is NOT created. "
		"\n\n"
		"For Rm, Ra, Cm, the function does func( r, L, len, dia ), " 
		"and if func() < 0.0 then the value takes the default value based "
		"on scaling of RM, RA, or CM by the geometry of the compartment "
		"\n\n"
		"Some example function forms might be: \n"
		" r < 10e-6 ? 400 * len * dia * pi : 0.0 \n"
		" ->Set channel Gbar to 400 S/m^2 only within 10 microns of soma\n"
		"\n"
		" L < 1.0 ? 100 * exp( -L ) * len * dia * pi : 0.0 \n"
		" ->Set channel Gbar to an exponentially falling function of "
		"electrotonic distance from soma, provided L is under 1.0 lambdas. "
		"\n"
		"The channelDistribution is parsed when the cell sourceFile is "
		"loaded. The parsing of this table is equivalent to repeated calls "
		"to *Neuron::assignChanDistrib()*, working through the entries in "
		"the *channelDistribution* vector. ",
		&Neuron::setChannelDistribution,
		&Neuron::getChannelDistribution
	);
	
	static ReadOnlyValueFinfo< Neuron, unsigned int > numCompartments( 
		"numCompartments",
		"Number of electrical compartments in model. ",
		&Neuron::getNumCompartments
	);

	static ReadOnlyValueFinfo< Neuron, unsigned int > numBranches( 
		"numBranches",
		"Number of branches in dendrites. ",
		&Neuron::getNumBranches
	);

	static ReadOnlyValueFinfo< Neuron, vector< double > > geomDistFromSoma( 
		"geometricalDistanceFromSoma",
		"geometrical distance of each segment from soma, as measured along "
		"the dendrite.",
		&Neuron::getGeomDistFromSoma
	);

	static ReadOnlyValueFinfo< Neuron, vector< double > > elecDistFromSoma( 
		"electrotonicDistanceFromSoma",
		"geometrical distance of each segment from soma, as measured along "
		"the dendrite.",
		&Neuron::getElecDistFromSoma
	);

	/////////////////////////////////////////////////////////////////////
	// DestFinfos
	/////////////////////////////////////////////////////////////////////
	static DestFinfo buildSegmentTree( "buildSegmentTree",
		"Build the reference segment tree structure using the child "
		"compartments of the current Neuron. Fills in all the coords and "
		"length constant information into the segments, for later use "
		"when we build reduced compartment trees and channel "
		"distributions. Should only be called once, since subsequent use "
	   "on a reduced model will lose the original full cell geometry. ",
		new EpFunc0< Neuron >( &Neuron::buildSegmentTree )
	);
	static DestFinfo assignChanDistrib( "assignChanDistrib",
		"Handles requests to assign the channel distribution. Args are "
		"chanName, pathOnCell, function( r, L, len, dia ) "
		"The function uses arguments:\n"
		"	r: geometrical distance from soma, measured along dendrite, in metres.\n"
		"	L: electrotonic distance (# of lambdas) from soma, along dend. No unts.\n"
		"	len: length of compartment, in metres.\n"
		"	dia: for diameter of compartment, in metres.\n"
		"For Channels, the function does Gbar = func( r, L, len, dia).\n"
		"For CaConc, the function does B = func( r, L, len, dia).\n"
		"In both cases, if func() < 0 then the chan/CaConc is NOT created. "
		"\n\n"
		"For Rm, Ra, Cm, the function does func( r, L, len, dia ), " 
		"and if func() < 0.0 then the value takes the default value based "
		"on scaling of RM, RA, or CM by the geometry of the compartment "
		"\n\n"
		"Some example function forms might be: \n"
		" r < 10e-6 ? 400 * len * dia * pi : 0.0 \n"
		" ->Set channel Gbar to 400 S/m^2 only within 10 microns of soma\n"
		"\n"
		" L < 1.0 ? 100 * exp( -e ) * len * dia * pi : 0.0 \n"
		" ->Set channel Gbar to an exponentially falling function of "
		"electrotonic distance from soma, provided L is under 1.0 lambdas. "
		"\n"
		"A dirty hack to get rid of a channel is to set the function to "
		" 0. But you should preferably use clearChanDistrib instead.\n"
		"\n\n Note that the pathOnCell is relative to the parent Neuron ",
		new EpFunc3< Neuron, string, string, string >(
			&Neuron::assignChanDistrib )
	);
	static DestFinfo clearChanDistrib( "clearChanDistrib",
		"Clears out a previously assigned channel distribution. "
		"Args are chanName, pathOnCell "
		"\n\n Note that the pathOnCell is relative to the parent Neuron ",
		new EpFunc2< Neuron, string, string >(
			&Neuron::clearChanDistrib )
	);
	static DestFinfo parseChanDistrib( "parseChanDistrib",
		"Parses a previously assigned vector of channel distributions. "
		"These are located in the field *channelDistribution*. "
		"When this function is called it builds the specified channels on "
		"the cell model loaded over self.",
		new EpFunc0< Neuron >(
			&Neuron::parseChanDistrib )
	);

	/*
	static DestFinfo decorateWithSpines( "decorateWithSpines",
					proto
					pathOnExistingCompts
					spacing, spacingDistrib
					sizeDistrib
					angle, angleDistrib
					rotation, rotationDistrib
					*/
	/*
	static DestFinfo rotateInSpace( "rotateInSpace",
		theta, phi
	static DestFinfo transformInSpace( "transformInSpace",
		transfMatrix(4x4)
	static DestFinfo saveAsNeuroML( "saveAsNeuroML", fname )
	static DestFinfo saveAsDotP( "saveAsDotP", fname )
	static DestFinfo saveAsSwc( "saveAsSwc", fname )
	*/
	
	static Finfo* neuronFinfos[] = 
	{ 	
		&RM,						// ValueFinfo
		&RA,						// ValueFinfo
		&CM,						// ValueFinfo
		&Em,						// ValueFinfo
		&theta,						// ValueFinfo
		&phi,						// ValueFinfo
		&sourceFile,				// ValueFinfo
		&compartmentLengthInLambdas,	// ValueFinfo
		&numCompartments,			// ReadOnlyValueFinfo
		&numBranches,				// ReadOnlyValueFinfo
		&geomDistFromSoma,			// ReadOnlyValueFinfo
		&elecDistFromSoma,			// ReadOnlyValueFinfo
		&channelDistribution,		// ValueFinfo
		&buildSegmentTree,			// DestFinfo
		&assignChanDistrib,			// DestFinfo
		&clearChanDistrib,			// DestFinfo
		&parseChanDistrib,			// DestFinfo
	};
	static string doc[] =
	{
		"Name", "Neuron",
		"Author", "C H Chaitanya, Upi Bhalla",
		"Description", "Neuron - Manager for neurons."
		" Handles high-level specification of channel distribution.",
	};
	static Dinfo<Neuron> dinfo;
	static Cinfo neuronCinfo(
				"Neuron",
				Neutral::initCinfo(),
				neuronFinfos, sizeof( neuronFinfos ) / sizeof( Finfo* ),
				&dinfo,
				doc,
				sizeof(doc)/sizeof(string)
	);

	return &neuronCinfo;
}

static const Cinfo* neuronCinfo = Neuron::initCinfo();

////////////////////////////////////////////////////////////////////////
Neuron::Neuron()
	: 
			RM_( 1.0 ),
			RA_( 1.0 ),
			CM_( 0.01 ),
			Em_( -0.065 ),
			theta_( 0.0 ),
			phi_( 0.0 ),
			sourceFile_( "" ),
			compartmentLengthInLambdas_( 0.2 )
{;}
////////////////////////////////////////////////////////////////////////
// ValueFinfos
////////////////////////////////////////////////////////////////////////

void Neuron::setRM( double v )
{
	if ( v > 0.0 )
		RM_ = v;
	else
		cout << "Warning:: Neuron::setRM: value must be +ve, is " << v << endl;
}
double Neuron::getRM() const
{
	return RM_;
}

void Neuron::setRA( double v )
{
	if ( v > 0.0 )
		RA_ = v;
	else
		cout << "Warning:: Neuron::setRA: value must be +ve, is " << v << endl;
}
double Neuron::getRA() const
{
	return RA_;
}

void Neuron::setCM( double v )
{
	if ( v > 0.0 )
		CM_ = v;
	else
		cout << "Warning:: Neuron::setCM: value must be +ve, is " << v << endl;
}
double Neuron::getCM() const
{
	return CM_;
}


void Neuron::setEm( double v )
{
	Em_ = v;
}
double Neuron::getEm() const
{
	return Em_;
}


void Neuron::setTheta( double v )
{
	theta_ = v;
	// Do stuff here for rotating it.
}
double Neuron::getTheta() const
{
	return theta_;
}


void Neuron::setPhi( double v )
{
	phi_ = v;
	// Do stuff here for rotating it.
}
double Neuron::getPhi() const
{
	return phi_;
}


void Neuron::setSourceFile( string v )
{
	sourceFile_ = v;
	// Stuff here for loading it.
}

string Neuron::getSourceFile() const
{
	return sourceFile_;
}


void Neuron::setCompartmentLengthInLambdas( double v )
{
	compartmentLengthInLambdas_ = v;
}
double Neuron::getCompartmentLengthInLambdas() const
{
	return compartmentLengthInLambdas_;
}

unsigned int Neuron::getNumCompartments() const
{
	return 0;
}

unsigned int Neuron::getNumBranches() const
{
	return 0;
}

vector< double> Neuron::getGeomDistFromSoma() const
{
	vector< double > ret( segs_.size(), 0.0 );
	for ( unsigned int i = 0; i < segs_.size(); ++i )
		ret[i] = segs_[i].getGeomDistFromSoma();
	return ret;
}

vector< double> Neuron::getElecDistFromSoma() const
{
	vector< double > ret( segs_.size(), 0.0 );
	for ( unsigned int i = 0; i < segs_.size(); ++i )
		ret[i] = segs_[i].getElecDistFromSoma();
	return ret;
}

void Neuron::setChannelDistribution( vector< string > v )
{
	if ( v.size() % 3 != 0 ) {
		cout << "Warning: Neuron::setChannelDistribution: vec must "
				"have 3N entries. \n" << v.size() << " entries found."
				"Value unchanged\n";
		return;
	}
	channelDistribution_ = v;
}

vector< string > Neuron::getChannelDistribution() const
{
	return channelDistribution_;
}


////////////////////////////////////////////////////////////////////////
// Stuff here for assignChanDistrib

static Id getComptParent( Id id )
{
	static const Finfo* axialFinfo = 
			Cinfo::find( "Compartment" )->findFinfo( "raxialOut" );
	static const Finfo* proximalFinfo = 
			Cinfo::find( "SymCompartment" )->findFinfo( "proximalOut" );

	if ( id.element()->cinfo()->isA( "CompartmentBase" ) ) {
		vector< Id > ret;
		id.element()->getNeighbors( ret, axialFinfo );
		if ( ret.size() == 1 )
			return ret[0];
		// If it didn't find an axial, maybe it is a symCompt
		if ( id.element()->cinfo()->isA( "SymCompartment" ) ) {
			id.element()->getNeighbors( ret, proximalFinfo );
			if ( ret.size() == 1 )
				return ret[0];
		}
	}
	return Id();
}

// Returns Id of soma
Id fillSegIndex( 
		const vector< Id >& kids, map< Id, unsigned int >& segIndex )
{
	Id soma;
	for ( unsigned int i = 0; i < kids.size(); ++i ) {
		const Id& k = kids[i];
		if ( k.element()->cinfo()->isA( "CompartmentBase" ) ) {
			segIndex[ k ] = i;
			if ( k.element()->getName() == "soma" ) {
				soma = k;
			}
		}
	}
	return soma;
}

static void fillSegments( vector< SwcSegment >& segs, 
	const map< Id, unsigned int >& segIndex,
	const vector< Id >& kids ) 
{
	segs.clear();
	for ( unsigned int i = 0; i < kids.size(); ++i ) {
		const Id& k = kids[i];
		if ( k.element()->cinfo()->isA( "CompartmentBase" ) ) {
			double x = Field< double >::get( k, "x" );
			double y = Field< double >::get( k, "y" );
			double z = Field< double >::get( k, "z" );
			double dia = Field< double >::get( k, "diameter" );
			Id pa = getComptParent( k );
			unsigned int paIndex = ~0U; // soma
			int comptType = 1; // soma
			if ( pa != Id() ) {
				map< Id, unsigned int >::const_iterator 
					j = segIndex.find( pa );
				if ( j != segIndex.end() ) {
					paIndex = j->second;
					comptType = 3; // generic dendrite
				}
			}
			segs.push_back( 
				SwcSegment( i, comptType, x, y, z, dia/2.0, paIndex ) );
		}
	}
}

/// Recursive function to fill in cumulative distances from soma.
static void traverseCumulativeDistance( 
	SwcSegment& self, vector< SwcSegment >& segs,
   	const vector< Id >& lookupId,  double rSoma, double eSoma )
{
	self.setCumulativeDistance( rSoma, eSoma );
	for ( unsigned int i = 0; i < self.kids().size(); ++i ) {
		SwcSegment& kid = segs[ self.kids()[i] ];
		double r = rSoma + kid.length( self );
		Id kidId = lookupId[ self.kids()[i] ];
		double Rm = Field< double >::get( kidId, "Rm" );
		double Ra = Field< double >::get( kidId, "Ra" );
		// Note that sqrt( Rm/Ra ) = lambda/length = 1/L.
		double e = eSoma + sqrt( Ra / Rm );
		traverseCumulativeDistance( kid, segs, lookupId, r, e );
	}
}

/// Fills up vector of segments. First entry is soma.
void Neuron::buildSegmentTree( const Eref& e )
{
	vector< Id > kids;
	Neutral::children( e, kids );
	map< Id, unsigned int > segIndex;

	Id soma = fillSegIndex( kids, segIndex );
	fillSegments( segs_, segIndex, kids );
	int numPa = 0;
	for ( unsigned int i = 0; i < segs_.size(); ++i ) {
		if ( segs_[i].parent() == ~0U ) {
			numPa++;
		} else {
			segs_[ segs_[i].parent() ].addChild( i );
		}
	}
	for ( unsigned int i = 0; i < segs_.size(); ++i ) {
		segs_[i].figureOutType();
	}

	if ( numPa != 1 ) {
		cout << "Warning: Neuron.cpp: buildTree: numPa = " << numPa << endl;
	}
	segId_.clear();
	segId_.resize( segIndex.size(), Id() );
	for ( map< Id, unsigned int >::const_iterator 
			i = segIndex.begin(); i != segIndex.end(); ++i ) {
		assert( i->second < segId_.size() );
		segId_[ i->second ] = i->first;
	}
	traverseCumulativeDistance( segs_[0], segs_, segId_, 0, 0 );
}

/////////////////////////////////////////////////////////////////////

static void buildChildDistanceMap( 
		const Eref& e, map< ObjId, pair< double, double > >& m )
{
	// Hack for testing: just find the geometrical distance to the soma,
	// and assume electronic length is 0.5 mm.
	
	vector< Id > kids;
	Neutral::children( e, kids );
	Id soma;
	for ( unsigned int i = 0; i < kids.size(); ++i ) {
		if ( kids[i].element()->getName() == "soma" ) {
			soma = kids[i];
			break;
		}
	}

	double x = Field< double >::get( soma, "x" );
	double y = Field< double >::get( soma, "y" );
	double z = Field< double >::get( soma, "z" );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i ) {
		if ( i->element()->cinfo()->isA( "CompartmentBase" ) ) {
			double cx = Field< double >::get( *i, "x0" );
			double cy = Field< double >::get( *i, "y0" );
			double cz = Field< double >::get( *i, "z0" );
			double dist = 
				sqrt( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) );
			m[ *i ] = pair< double, double >( dist, dist / 0.5e-3 );
		}
	}
}

static Id acquireChannel( Shell* shell, const string& name, ObjId compt )
{
	Id chan = Neutral::child( compt.eref(), name );
	if ( chan == Id() ) { // Need to make it from prototype.
		Id proto( "/library/" + name );
		if ( proto == Id() ) {
			cout << "Warning: channel proto '" << name << "' not found\n";
			return Id();
		}
		chan = shell->doCopy( proto, compt, name, 1, false, false );
		// May need to do additional msgs depending on the chan type.
		if ( chan.element()->cinfo()->isA( "ChanBase" ) ) {
			shell->doAddMsg( "Single", compt, "channel", chan, "channel" );
		}
		ReadCell::addChannelMessage( chan );
	}
	return chan;
}

static void assignParam( Shell* shell, ObjId compt
				, double val, const string& name )
{
	// Only permit chans with val greater than zero.
	if ( val > 0.0 ) {
		if ( name == "Rm" ) {
			Field< double >::set( compt, "Rm", val );
		} else if ( name == "Ra" ) {
			Field< double >::set( compt, "Ra", val );
		} else if ( name == "Cm" ) {
			Field< double >::set( compt, "Cm", val );
		} else {
			Id chan = acquireChannel( shell, name, compt );
			if ( chan.element()->cinfo()->isA( "ChanBase" ) ) {
				Field< double >::set( chan, "Gbar", val );
			} else if ( chan.element()->cinfo()->isA( "CaConcBase" ) ) {
				Field< double >::set( chan, "B", val );
			}
		}
	}
}

static void evalChanParams( 
	const string& name, const string& func,
	vector< ObjId >& elist,
   	const map< ObjId, pair< double, double > >& m )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	// Build the function
	double r = 0; // geometrical distance arg
	double L = 0; // electrical distance arg
	double len = 0; // Length of compt in metres
	double dia = 0; // Diameter of compt in metres
	try {
		mu::Parser parser;
		parser.DefineVar( "r", &r );
		parser.DefineVar( "L", &L );
		parser.DefineVar( "len", &len );
		parser.DefineVar( "dia", &dia );
		parser.SetExpr( func );

		// Go through the elist checking for the channels. If not there,
		// build them. 
		for ( vector< ObjId >::iterator 
						i = elist.begin(); i != elist.end(); ++i) {
			if ( i->element()->cinfo()->isA( "CompartmentBase" ) ) {
				dia = Field< double >::get( *i, "diameter" );
				len = Field< double >::get( *i, "length" );
   				map< ObjId, pair< double, double > >::const_iterator j =
						m.find( *i );
				assert ( j != m.end() );
				r = j->second.first;
				L = j->second.second;

				double val = parser.Eval();
				assignParam( shell, *i, val, name );
			}
		}
	}
	catch ( mu::Parser::exception_type& err )
	{
		cout << err.GetMsg() << endl;
	}
}

void Neuron::assignChanDistrib( const Eref& e, 
				string name, string path, string func )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	// Go through all child compts recursively, figures out geom and
	// electrotonic distance to the child. Puts in a map.
	map< ObjId, pair< double, double > > m;
	buildChildDistanceMap( e, m );

	// build the elist of affected compartments.
	vector< ObjId > elist;
	ObjId oldCwe = shell->getCwe();
	shell->setCwe( e.objId() );
	wildcardFind( path, elist );
	shell->setCwe( oldCwe );
	if ( elist.size() == 0 )
		return;
	evalChanParams( name, func, elist, m );
}

void Neuron::clearChanDistrib( const Eref& e, string name, string path )
{
	assignChanDistrib( e, name, path, "0" );
}

void Neuron::parseChanDistrib( const Eref& e )
{
	for ( unsigned int i = 0; 3 * i < channelDistribution_.size(); ++i ) {
		assignChanDistrib( e, 
						channelDistribution_[i * 3],
						channelDistribution_[i * 3 + 1],
						channelDistribution_[i * 3 + 2] );
	}
}

