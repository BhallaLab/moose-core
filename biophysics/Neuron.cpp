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
#include "../randnum/Normal.h"
#include "../randnum/randnum.h"
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
		"Units: meters (SI). \n"
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
		"Note that the Gbar is automatically scaled by the area of the "
		"compartment, you do not have to compute this.\n"
		"For CaConc, the function does B = func( r, L, len, dia).\n"
		"In both cases, if func() < 0 then the chan/CaConc is NOT created. "
		"\n\n"
		"For RM, RA, CM, Em, initVm, the function does \n"
		"	func( r, L, len, dia ) \n" 
		"with automatic scaling of the RM, RA, or CM by the geometry "
		"of the compartment. As before, "
		"if func() < 0.0 then the default value is used. "
		"\n\n"
		"Some example function forms might be for a channel Gbar: \n"
		" r < 10e-6 ? 400 : 0.0 \n"
		"equivalently, "
		" ( sign(10e-6 - r) + 1) * 200 \n"
		"Both of these forms instruct the function to "
		"set channel Gbar to 400 S/m^2 only within 10 microns of soma\n"
		"\n"
		" L < 1.0 ? 100 * exp( -L ) : 0.0 \n"
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
	static ValueFinfo< Neuron, vector< string > > spineSpecification( 
		"spineSpecification",
		"Specification for spine creation on this neuron. "
		"Each entry in the specification is a string with the following "
		"arguments: \n"
		"protoName pathOnCell [spacing] [spacingDistrib] [sizeDistrib] "
		"[angle] [angleDistrib] [rotation] [rotationDistrib]\n"
		"Here the items in brackets are optional arguments. The default "
		"spacing is 1e-6 metres and the rest of the defaults are zero. "
		"Length units are metres and angle units are radians.",
		&Neuron::setSpineSpecification,
		&Neuron::getSpineSpecification
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
	static ReadOnlyValueFinfo< Neuron, vector< ObjId > > compartments( 
		"compartments",
		"Vector of ObjIds of electricalcompartments. Order matches order "
		"of segments, and also matches the order of the electrotonic and "
		"geometricalDistranceFromSoma vectors. ",
		&Neuron::getCompartments
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
	static DestFinfo insertSpines( "insertSpines",
		"This function inserts spines on a neuron, placing them "
		"perpendicular to the dendrite axis and spaced away from the axis "
		"by the diameter of the dendrite. The placement of spines is "
		"controlled by a range of parameters. "
		"Arguments:\n "
		"spineid: is the parent object of the prototype spine. The "
		" spine prototype can include any number of compartments, and each"
		" can have any number of voltage and ligand-gated channels, as "
		" well as CaConc and other mechanisms.\n"
		"path: is a wildcard path string for parent compartments for "
		" the new spines. "
		"placement: is a vector of doubles with the parameters for "
		" placing the spines. Zero to six arguments may be provided. "
		" Default spacing is 1 micron, remaining defaults are 0.0. "
		" The parameters within placement are: \n"
		"    placement[0] = spacing: distance (metres) between spines. \n"
		"    placement[1] = spacingDistrib (metres): the width of a \n"
	    "    normal distribution around the spacing. \n"
		"    placement[2] = sizeDistrib: Random scaling of spine size.\n"
		"    placement[3] = angle: Angular rotation around the axis of "
		"    the dendrite. 0 radians is facing away from the soma. \n"
		"    placement[4] = angleDistrib: Scatter around angle. "
		"    The simplest way to put the spine in any random position is "
		" to have an angleDistrib of 2 pi. The algorithm selects any "
		" angle in the linear range of the angle distrib to add to the "
		" specified angle.\n"
		"    placement[5] = rotation (radians): With each position along "
		"    the dendrite the algorithm computes a new spine direction, "
		"    using rotation to increment the angle. "
		"    placement[6] = rotationDistrib. Scatter in rotation. ",
		new EpFunc3< Neuron, Id, string, vector< double > >(
			&Neuron::insertSpines )
	);
	static DestFinfo clearSpines( "clearSpines",
		"Clears out all spines. ",
		new EpFunc0< Neuron >(
			&Neuron::clearSpines )
	);
	static DestFinfo parseSpines( "parseSpines",
		"Parses a previously assigned vector of spine specifications. "
		"This is located in the field *spineSpecification*. "
		"When this function is called it builds the specified spines on "
		"the cell model.",
		new EpFunc0< Neuron >(
			&Neuron::parseSpines )
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
		&compartments,				// ReadOnlyValueFinfo
		&channelDistribution,		// ValueFinfo
		&spineSpecification,		// ValueFinfo
		&buildSegmentTree,			// DestFinfo
		&insertSpines,				// DestFinfo
		&clearSpines,				// DestFinfo
		&parseSpines,				// DestFinfo
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
			compartmentLengthInLambdas_( 0.2 ),
			spineIndex_( 0 )
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
	return segId_.size();
}

unsigned int Neuron::getNumBranches() const
{
	return branches_.size();
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

vector< ObjId > Neuron::getCompartments() const
{
	vector< ObjId > ret( segId_.size() );
	for ( unsigned int i = 0; i < segId_.size(); ++i )
		ret[i] = segId_[i];
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

void Neuron::setSpineSpecification( vector< string > v )
{
	spineSpecification_ = v;
}

vector< string > Neuron::getSpineSpecification() const
{
	return spineSpecification_;
}

////////////////////////////////////////////////////////////////////////
// Stuff here for parsing the compartment tree
////////////////////////////////////////////////////////////////////////

static Id getComptParent( Id id )
{
	// raxial points towards soma.
	static const Finfo* raxialFinfo = 
			Cinfo::find( "Compartment" )->findFinfo( "raxialOut" );
	static const Finfo* proximalFinfo = 
			Cinfo::find( "SymCompartment" )->findFinfo( "proximalOut" );

	if ( id.element()->cinfo()->isA( "CompartmentBase" ) ) {
		vector< Id > ret;
		id.element()->getNeighbors( ret, raxialFinfo );
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
	segIndex.clear();
	Id fatty;
	double maxDia = 0.0;
	for ( unsigned int i = 0; i < kids.size(); ++i ) {
		const Id& k = kids[i];
		if ( k.element()->cinfo()->isA( "CompartmentBase" ) ) {
			segIndex[ k ] = i;
			const string& s = k.element()->getName();
			if ( s.find( "soma" ) != s.npos ||
				s.find( "Soma" ) != s.npos ||
				s.find( "SOMA" ) != s.npos ) {
				soma = k;
			}
			double dia = Field< double >::get( k, "diameter" );
			if ( dia > maxDia ) {
				maxDia = dia;
				fatty = k;
			}
		}
	}
	if ( soma != Id() )
		return soma;
	return fatty; // Fallback.
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

	Id soma = fillSegIndex( kids, segIndex_ );
	if ( kids.size() == 0 || soma == Id() ) {
		cout << "Error: Neuron::buildSegmentTree( " << e.id().path() <<
				" ): \n		Valid neuronal model not found.\n";
		return;
	}
	fillSegments( segs_, segIndex_, kids );
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
	segId_.resize( segIndex_.size(), Id() );
	for ( map< Id, unsigned int >::const_iterator 
			i = segIndex_.begin(); i != segIndex_.end(); ++i ) {
		assert( i->second < segId_.size() );
		segId_[ i->second ] = i->first;
	}
	traverseCumulativeDistance( segs_[0], segs_, segId_, 0, 0 );
}

////////////////////////////////////////////////////////////////////////
// Stuff here for assignChanDistrib
////////////////////////////////////////////////////////////////////////

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
				, double val, const string& name, double len, double dia )
{
	// Only permit chans with val greater than zero.
	if ( val > 0.0 ) {
		if ( name == "Rm" || name == "RM" ) {
			Field< double >::set( compt, "Rm", val / ( len  * dia * PI ) );
		} else if ( name == "Ra" || name == "RA" ) {
			Field< double >::set( compt, "Ra", val*len*4 / (dia*dia*PI) );
		} else if ( name == "Cm" || name == "CM" ) {
			Field< double >::set( compt, "Cm", val * len * dia * PI );
		} else if ( name == "Em" || name == "EM" ) {
			Field< double >::set( compt, "Em", val );
		} else if ( name == "initVm" || name == "INITVM" ) {
			Field< double >::set( compt, "initVm", val );
		} else {
			Id chan = acquireChannel( shell, name, compt );
			if ( chan.element()->cinfo()->isA( "ChanBase" ) ) {
				Field< double >::set( chan, "Gbar", val * len * dia * PI );
			} else if ( chan.element()->cinfo()->isA( "CaConcBase" ) ) {
				Field< double >::set( chan, "B", val );
			}
		}
	}
}

void Neuron::evalChanParams( 
	const string& name, const string& func,
	vector< ObjId >& elist )
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
				map< Id, unsigned int >:: const_iterator j = 
					segIndex_.find( *i );
				assert( j != segIndex_.end() );
				assert( j->second < segs_.size() );
				r = segs_[ j->second ].getGeomDistFromSoma();
				L = segs_[ j->second ].getElecDistFromSoma();

				double val = parser.Eval();
				assignParam( shell, *i, val, name, len, dia );
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
	if ( segIndex_.size() == 0 && segs_.size() == 0 )
		buildSegmentTree( e );

	// build the elist of affected compartments.
	vector< ObjId > elist;
	ObjId oldCwe = shell->getCwe();
	shell->setCwe( e.objId() );
	wildcardFind( path, elist );
	shell->setCwe( oldCwe );
	if ( elist.size() == 0 )
		return;
	evalChanParams( name, func, elist );
}

void Neuron::clearChanDistrib( const Eref& e, string name, string path )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< ObjId > elist;
	ObjId oldCwe = shell->getCwe();
	shell->setCwe( e.objId() );
	path = path + "/" + name;
	wildcardFind( path, elist );
	shell->setCwe( oldCwe );
	if ( elist.size() == 0 )
		return;
	for ( vector< ObjId >::iterator 
		i = elist.begin(); i != elist.end(); ++i) {
			shell->doDelete( *i );
	}
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

////////////////////////////////////////////////////////////////////////
// Stuff here for inserting spines.
////////////////////////////////////////////////////////////////////////

/**
 * Utility function to return a coordinate system where 
 * z is the direction of a dendritic segment, 
 * x is the direction of spines outward from soma and perpendicular to z
 * and y is the perpendicular to x and z.
 */
static double coordSystem( Id soma, Id dend, Vec& x, Vec& y, Vec& z )
{
	static const double EPSILON = 1e-20;
	double x0 = Field< double >::get( dend, "x0" );
	double y0 = Field< double >::get( dend, "y0" );
	double z0 = Field< double >::get( dend, "z0" );
	double x1 = Field< double >::get( dend, "x" );
	double y1 = Field< double >::get( dend, "y" );
	double z1 = Field< double >::get( dend, "z" );

	Vec temp( x1-x0, y1-y0, z1-z0 );
	double len = temp.length();
	z = Vec( temp.a0()/len, temp.a1()/len, temp.a2()/len );

	double sx0 = Field< double >::get( soma, "x0" );
	double sy0 = Field< double >::get( soma, "y0" );
	double sz0 = Field< double >::get( soma, "z0" );
	Vec temp2( x0 - sx0, y0-sy0, z0-sz0 );
	Vec y2 = temp.crossProduct( z );
	double ylen = y2.length();
	double ytemp = 1.0;
	while ( ylen < EPSILON ) {
		Vec t( ytemp , sqrt( 2.0 ), 0.0 );
		y2 = t.crossProduct( z );
		ylen = y2.length();
		ytemp += 1.0;
	}
	y = Vec( y2.a0()/ylen, y2.a1()/ylen, y2.a2()/ylen );
	x = z.crossProduct( y );
	assert( doubleEq( x.length(), 1.0 ) );
	return len;
}

/**
 * Utility function to resize electrical compt electrical properties,
 * including those of its child channels and calcium conc.
 */
static void scaleSpineCompt( Id compt, double size )
{
	vector< ObjId > chans;
	allChildren( compt, "ISA=ChanBase", chans );
	// wildcardFind( compt.path() + "/##[ISA=ChanBase]", chans );
	double a = size * size;
	for ( vector< ObjId >::iterator 
					i = chans.begin(); i != chans.end(); ++i )
	{
		double gbar = Field< double >::get( *i, "Gbar" );
		Field< double >::set( *i, "Gbar", gbar * a );
	}

	double v = size * size * size;
	vector< ObjId > concs;
	allChildren( compt, "ISA=CaConcBase", concs );
	// wildcardFind( compt.path() + "/##[ISA=CaConcBase]", concs );
	for ( vector< ObjId >::iterator 
					i = concs.begin(); i != concs.end(); ++i )
	{
		double B = Field< double >::get( *i, "B" );
		Field< double >::set( *i, "B", B * v );
	}

	double Rm = Field< double >::get( compt, "Rm" );
	Field< double >::set( compt, "Rm", Rm / a );
	double Cm = Field< double >::get( compt, "Cm" );
	Field< double >::set( compt, "Cm", Cm * a );
	double Ra = Field< double >::get( compt, "Ra" );
	Field< double >::set( compt, "Ra", Ra / size );
}


/**
 * Utility function to change coords of spine so as to reorient it.
 */
static void reorientSpine( vector< Id >& spineCompts, 
				vector< Vec >& coords, 
				Vec& parentPos, double pos, double angle, 
				Vec& x, Vec& y, Vec& z )
{
	double c = cos( angle );
	double s = sin( angle );
	double omc = 1.0 - c;

	Vec rot0( 		z.a0()*z.a0()*omc + c, 
					z.a1()*z.a0()*omc - z.a2()*s ,
					z.a2()*z.a0()*omc + z.a1()*s );

	Vec rot1( 		z.a0()*z.a1()*omc + z.a2()*s, 
					z.a1()*z.a1()*omc + c,
            		z.a2()*z.a1()*omc - z.a0()*s );

	Vec rot2(		z.a0()*z.a2()*omc - z.a1()*s, 
					z.a1()*z.a2()*omc + z.a0()*s, 
					z.a2()*z.a2()*omc + c );

    Vec translation = z * pos + parentPos;
    // Vec translation = parentPos;
	vector< Vec > ret( coords.size() );
	for ( unsigned int i = 0; i < coords.size(); ++i ) {
		ret[i] = Vec( 	rot0.dotProduct( coords[i] ) + translation.a0(), 
						rot1.dotProduct( coords[i] ) + translation.a1(), 
						rot2.dotProduct( coords[i] ) + translation.a2() );
		
	}
    assert( spineCompts.size() * 2 == ret.size() );

	for ( unsigned int i = 0; i < spineCompts.size(); ++i ) {
		unsigned int j = 2 * i;
		Field< double >::set( spineCompts[i], "x0", ret[j].a0() );
		Field< double >::set( spineCompts[i], "y0", ret[j].a1() );
		Field< double >::set( spineCompts[i], "z0", ret[j].a2() );
		// cout << "(" << ret[j].a0() << ", " << ret[j].a1() << ", " << ret[j].a2() << ")";
            j = j + 1;
		Field< double >::set( spineCompts[i], "x", ret[j].a0() );
		Field< double >::set( spineCompts[i], "y", ret[j].a1() );
		Field< double >::set( spineCompts[i], "z", ret[j].a2() );
		// cout << "(" << ret[j].a0() << ", " << ret[j].a1() << ", " << ret[j].a2() << ")\n";
	}
}

/** 
 * Utility function to add a single spine to the given parent.
 * parent is parent compartment for this spine.
 * spineProto is just that.
 * pos is position (in metres ) along parent compartment
 * angle is angle (in radians) to rotate spine wrt x in plane xy.
 * Size is size scaling factor, 1 leaves as is.
 * x, y, z are unit vectors. Z is along the parent compt.
 * We first shift the spine over so that it is offset by the parent compt
 * diameter.
 * We then need to reorient the spine which lies along (i,0,0) to
 * lie along x. X is a unit vector so this is done simply by 
 * multiplying each coord of the spine by x.
 * Finally we rotate the spine around the z axis by the specified angle
 * k is index of this spine.
 */

static void addSpine( Id parentCompt, Id spineProto, 
		double pos, double angle, 
		Vec& x, Vec& y, Vec& z, 
		double size, 
		unsigned int k )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	Id parentObject = Neutral::parent( parentCompt );
	stringstream sstemp;
	sstemp << k;
	string kstr = sstemp.str();
	Id spine = shell->doCopy( spineProto, parentObject, "_spine" + kstr, 
					1, false, false );
	vector< Id > kids;
	Neutral::children( spine.eref(), kids );
	double x0 = Field< double >::get( parentCompt, "x0" );
	double y0 = Field< double >::get( parentCompt, "y0" );
	double z0 = Field< double >::get( parentCompt, "z0" );
	double parentRadius = Field< double >::get( parentCompt, "diameter" )/2;
	Vec ppos( x0, y0, z0 );
	// First, build the coordinates vector for the spine. Assume that its
	// major axis is along the unit vector [1,0,0].
	vector< Vec > coords;
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i )
	{
		if ( i->element()->cinfo()->isA( "CompartmentBase" ) ) {
			i->element()->setName( i->element()->getName() + kstr );
			x0 = Field< double >::get( *i, "x0" ) * size;
			y0 = Field< double >::get( *i, "y0" ) * size;
			z0 = Field< double >::get( *i, "z0" ) * size;
			coords.push_back( Vec( x0 + parentRadius, y0, z0 ) );
			double x = Field< double >::get( *i, "x" ) * size;
			double y = Field< double >::get( *i, "y" ) * size;
			double z = Field< double >::get( *i, "z" ) * size;
			double dia = Field< double >::get( *i, "diameter" ) * size;
			Field< double >::set( *i, "diameter", dia );
			coords.push_back( Vec( x + parentRadius, y, z ) );
			scaleSpineCompt( *i, size );
			shell->doMove( *i, parentObject );
		}
	}
	// Then, take the projection of this along the x vector passed in.
	for( vector< Vec >::iterator i = coords.begin(); i != coords.end(); ++i)
		*i = x * i->a0();
	shell->doDelete( spine ); // get rid of the holder for the spine copy.
	shell->doAddMsg( "Single", parentCompt, "axial", kids[0], "raxial" );
	reorientSpine( kids, coords, ppos, pos, angle, x, y, z );
}

static void makeSpacingDistrib( vector< double >& pos, 
				double spacing, double spacingDistrib )
{
	unsigned int num = pos.size();
	if ( spacingDistrib > 0.0 ) {
		/*
		 * Here again we need to get rid of the Normal RNG due to speed.
		Normal ns( spacing, spacingDistrib * spacingDistrib );
		// We can't have a simple normal distrib, have to guarantee that
		// we always move forward.
		double x = 0.0;
		while ( x <= 0.0 )
			x = ns.getNextSample() / 2.0;
		for ( unsigned int j = 0; j < num; ++j ) {
			pos[j] = x;
			double temp = 0.0;
			while ( temp <= 0.0 )
				temp = ns.getNextSample();
			x += temp;
		}
		*/
		if ( spacingDistrib > spacing ) {
			cout << "Warning: Neuron::makeSpacingDistrib: Distribution = "
					<< spacingDistrib << " must be < spacing = " <<
					spacing << ". Using " << spacing << endl;
			spacingDistrib = spacing;
		}
		for ( unsigned int j = 0; j < num; ++j ) {
			pos[j] = spacing * ( 0.5 + j ) + 
					( 2.0*mtrand() - 1.0 ) * spacingDistrib;
		}
	} else {
		for ( unsigned int j = 0; j < num; ++j ) {
			pos[j] = spacing * ( 0.5 + j );
		}
	}
}

// All angles in radians.
static void makeAngleDistrib( vector< double >& theta, 
				double angle, double angleDistrib, 
				double rotation, double rotationDistrib )
{
	if ( angleDistrib > 0.0 )
		angle += mtrand() * angleDistrib;
	unsigned int num = theta.size();
	if ( rotationDistrib > 0.0 ) {
		/* Get rid of Normal RNG
		Normal nr( rotation, rotationDistrib* rotationDistrib );
		double x = angle;
		for ( unsigned int j = 0; j < num; ++j ) {
			theta[j] = x;
			x += nr.getNextSample();
		}
		*/
		for ( unsigned int j = 0; j < num; ++j ) {
			theta[j] = angle + rotation * ( 0.5 + j ) +
				(2.0*mtrand() - 1.0)  * rotationDistrib;
		}
	} else {
		for ( unsigned int j = 0; j < num; ++j ) {
			theta[j] = angle + rotation * ( 0.5 + j );
		}
	}
}

/**
 * API function to add a series of spines.
 * 
 * The spineid is the parent object of the prototype spine. The 
 * spine prototype can include any number of compartments, and each
 * can have any number of voltage and ligand-gated channels, as well
 * as CaConc and other mechanisms.
 * The parentList is a list of Object Ids for parent compartments for
 * the new spines
 * The spacingDistrib is the width of a normal distribution around
 * the spacing. Both are in metre units.
 * The reference angle of 0 radians is facing away from the soma.
 * In all cases we assume that the spine will be rotated so that its
 * axis is perpendicular to the axis of the dendrite.
 * The simplest way to put the spine in any random position is to have
 * an angleDistrib of 2 pi. The algorithm selects any angle in the
 * linear range of the angle distrib to add to the specified angle.
 * With each position along the dendrite the algorithm computes a new
 * spine direction, using rotation to increment the angle.
 * Returns list of spines.
 */

void Neuron::insertSpines( const Eref& e, Id spineProto, string path, 
				vector< double > placement )
{
	vector< ObjId > parentList;
	double args[] = {1.0e-6, 0,0,0,0,0,0};
	double& spacing = args[0];
	double& spacingDistrib = args[1];
	double& sizeDistrib = args[2];
	double& angle = args[3];
	double& angleDistrib = args[4];
	double& rotation = args[5];
	double& rotationDistrib = args[6];
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );

	vector< ObjId > somaList;
	wildcardFind( e.id().path() + "/#soma#", somaList );
	Id soma = somaList[0];

	for ( unsigned int i = 0; i < 7 && i < placement.size(); ++i ) {
		args[i] = placement[i];
	}

	// Do this juggle with cwe so that we can handle rel as well as
	// absolute paths.
	ObjId oldCwe = shell->getCwe();
	shell->setCwe( e.objId() );
	wildcardFind( path, parentList );
	shell->setCwe( oldCwe );
	for ( unsigned int i = 0; i < parentList.size(); ++i ) {
		Vec x, y, z;
		double dendLength = coordSystem( soma, parentList[i], x, y, z );

		// Have extra num entries to allow for the random spacing.
		int num = dendLength / spacing + 2.0; 

		vector< double > pos( num );
		makeSpacingDistrib( pos, spacing, spacingDistrib );

		vector< double > theta( num );
		makeAngleDistrib( theta, angle, angleDistrib, 
						rotation, rotationDistrib );

		vector< double > size( num, 1.0 );

		/* The use of this RNG causes a 100x slowdown in spine setup!
		 * It uses the alias method, I would have thought Box-Muller would
		 * be faster but I won't worry about it for now.
		if ( sizeDistrib > 0.0 ) {
			Normal nz( 1.0, sizeDistrib* sizeDistrib );
			for ( int j = 0; j < num; ++j ) {
				double s = 0.0;
				while( s <= 0.1 ) // Arb cutoff. Exclude tiny spines.
					s = nz.getNextSample();
				size[j] = s;
			}
		}
		*/
		if ( sizeDistrib > 0.0 ) {
			if ( sizeDistrib > 0.9 ) {
				cout << "Warning: Neuron::insertSpines: sizeDistrib = " <<
					   sizeDistrib << "	too big, using 0.9\n";
				sizeDistrib = 0.9;
			}
			for ( int j = 0; j < num; ++j ) {
				size[j] = mtrand() * 2.0 * sizeDistrib + 1.0 - sizeDistrib;
			}
		}
		for ( unsigned int j = 0; j < pos.size(); ++j ) {
			if ( pos[j] > dendLength || pos[j] < 0.0 )
				break;
			addSpine( parentList[i], spineProto, pos[j], theta[j], 
							x, y, z, size[j], spineIndex_++ );
		}
	}
	cout << "Neuron::insertSpines: " << spineIndex_ << " spines inserted\n";
}

void Neuron::parseSpines( const Eref& e )
{
	for ( unsigned int i = 0; i < spineSpecification_.size(); ++i ) {
		stringstream ss( spineSpecification_[i] );
		string proto;
		string path;
		vector< double > args( 7, 0.0 );
		args[0] = 1e-6;
		ss >> proto >> path >> args[0] >> args[1] >> args[2] >> args[3] >> args[4] >> args[5] >> args[6];

		Id spineProto;
		if ( proto[0] == '/' )
			spineProto = Id( proto );
		else
			spineProto = Id( "/library/" + proto );

		if ( spineProto == Id() ) {
			cout << "Warning: Neuron::parseSpines: Unable to find prototype spine: " << proto << endl;
			return;
		}

		insertSpines( e, spineProto, path, args );
	}
}

void Neuron::clearSpines( const Eref& e )
{
}
