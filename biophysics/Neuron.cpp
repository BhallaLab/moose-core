/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Neuron.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"

#include "../external/muparser/muParser.h"

static const double MinGbar = 0.01; // Physiol units per unit area. 

/**
 * The initCinfo() function sets up the Compartment class.
 * This function uses the common trick of having an internal
 * static value which is created the first time the function is called.
 * There are several static arrays set up here. The ones which
 * use SharedFinfos are for shared messages where multiple kinds
 * of information go along the same connection.
 */
const Cinfo* Neuron::initCinfo()
{
	static DestFinfo updateChanDistrib( "updateChanDistrib",
	"Handles requests to update the channel distribution. Args are "
	"chanName, Gbar, pathOnCell, function( geomPos, elecPos )"
	" Note that the pathOnCell is relative to the parent Neuron ",
	new EpFunc4< Neuron, string, double, string, string >(
			&Neuron::updateChanDistrib )
	);
	
	static Finfo* neuronFinfos[] = 
	{ 	
		&updateChanDistrib,			// DestFinfo
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
		shell->doAddMsg( "Single", compt, "channel", chan, "channel" );
	}
	return chan;
}

void Neuron::updateChanDistrib( const Eref& e, 
				string name, double max, string path, string func )
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

	// Build the function
	double x = 0; // geometrical distance arg
	double y = 0; // electrical distance arg
	try {
		mu::Parser parser;
		parser.DefineVar( "g", &x );
		parser.DefineVar( "e", &y );
		parser.SetExpr( func );

		// Go through the elist checking for the channels. If not there,
		// build them. 
		for ( vector< ObjId >::iterator 
						i = elist.begin(); i != elist.end(); ++i) {
			if ( i->element()->cinfo()->isA( "CompartmentBase" ) ) {
				double dia = Field< double >::get( *i, "diameter" );
				double len = Field< double >::get( *i, "length" );
				double area = len * dia * dia * PI / 4.0;

				const pair< double, double >& dist = m[*i];
	
				x = dist.first;
				y = dist.second;
				double gbar = parser.Eval();
				// Only permit chans with cond greater than MinGbar
				if ( gbar > MinGbar ) {
					gbar *= area;

					Id chan = acquireChannel( shell, name, *i );
					Field< double >::set( chan, "Gbar", gbar );
				}
			}
		}
	}
	catch ( mu::Parser::exception_type& err )
	{
		cout << err.GetMsg() << endl;
	}
}

