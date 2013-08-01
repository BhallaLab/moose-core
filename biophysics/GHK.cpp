/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "GHK.h"
/*
 * NOT FULLY TESTED YET.
 *
 *
 */

static SrcFinfo1< double >* VmOut()
{
	static SrcFinfo1< double > VmOut( "VmOut", 
		"Relay of membrane potential Vm." );
	return &VmOut;
}

static SrcFinfo2< double, double >* channelOut()
{
	static SrcFinfo2< double, double > channelOut( "channelOut", 
		"Sends channel variables Gk and Ek to compartment" );
	return &channelOut;
}

static SrcFinfo1< double >* IkOut()
{
	static SrcFinfo1< double > IkOut( "IkOut", 
		"MembraneCurrent." );
	return &IkOut;
}

const Cinfo* GHK::initCinfo()
{
	/////////////////////////////////////////////////////////////////////
	// Shared messages
	/////////////////////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles process call",
		new ProcOpFunc< GHK >( &GHK::process ) );
	static DestFinfo reinit( "reinit", 
		"Handles reinit call",
		new ProcOpFunc< GHK >( &GHK::reinit ) );
	static Finfo* processShared[] =
	{
		&process, &reinit
	};
	static SharedFinfo proc( "proc", 
			"This is a shared message to receive Process message from the"
			"scheduler. The first entry is a MsgDest for the Process "
			"operation. It has a single argument, ProcInfo, which "
			"holds lots of information about current time, thread, dt and"
			"so on.\n The second entry is a MsgDest for the Reinit "
			"operation. It also uses ProcInfo.",
		processShared, sizeof( processShared ) / sizeof( Finfo* )
	);

	/////////////////////////////////////////////////////////////////////
	/// ChannelOut SrcFinfo defined above.
	static DestFinfo handleVm( "handleVm", 
		"Handles Vm message coming in from compartment",
		new EpFunc1< GHK, double >( &GHK::handleVm ) );

	static Finfo* channelShared[] =
	{
		channelOut(), &handleVm
	};
	static SharedFinfo channel( "channel", 
		"This is a shared message to couple channel to compartment. "
		"The first entry is a MsgSrc to send Gk and Ek to the compartment "
		"The second entry is a MsgDest for Vm from the compartment.",
		channelShared, sizeof( channelShared ) / sizeof( Finfo* )
	);

	///////////////////////////////////////////////////////
	static DestFinfo addPermeability( "addPermeability", 
		"Handles permeability message coming in from channel",
		new OpFunc1< GHK, double >( &GHK::addPermeability ) );

	/// Permability SrcFinfo defined above.
	static Finfo* ghkShared[] =
	{
		VmOut(), &addPermeability
	};
	static SharedFinfo ghk( "ghk", 
		"Message from channel to current Goldman-Hodgkin-Katz object"
		"This shared message connects to an HHChannel. "
		"The first entry is a MsgSrc which relays the Vm received from "
		"a compartment. The second entry is a MsgDest which receives "
		"channel conductance, and interprets it as permeability.",
		ghkShared, sizeof( ghkShared ) / sizeof( Finfo* ) );

	/////////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< GHK, double > Ik( "Ik", 
		"Membrane current",
    	&GHK::getIk
	);
	static ReadOnlyValueFinfo< GHK, double > Gk( "Gk", 
		"Conductance",
		&GHK::getGk
	);
	static ReadOnlyValueFinfo< GHK, double > Ek( "Ek", 
		"Reversal Potential",
		&GHK::getEk
	);
	static ValueFinfo< GHK, double > T( "T", 
		"Temperature of system",
		&GHK::setTemperature,
		&GHK::getTemperature
	);
	static ValueFinfo< GHK, double > p( "p", 
		"Permeability of channel",
		&GHK::setPermeability,
		&GHK::getPermeability
	);
	static ValueFinfo< GHK, double > Vm( "Vm", 
		"Membrane potential",
		&GHK::setVm,
		&GHK::getVm
	);
	static ValueFinfo< GHK, double > Cin( "Cin", 
		"Internal concentration",
		&GHK::setCin,
		&GHK::getCin
	);
	static ValueFinfo< GHK, double > Cout( "Cout", 
		"External ion concentration",
		&GHK::setCout,
		&GHK::getCout
	);
	static ValueFinfo< GHK, double > valency( "valency", 
		"Valence of ion",
		&GHK::setValency,
		&GHK::getValency
	);
	/////////////////////////////////////////////////////////////////////
	// DestFinfos
	/////////////////////////////////////////////////////////////////////
	static DestFinfo CinDest( "CinDest", 
		"Alias for set_Cin",
		new OpFunc1< GHK, double >( &GHK::setCin ) );
	static DestFinfo CoutDest( "CoutDest", 
		"Alias for set_Cout",
		new OpFunc1< GHK, double >( &GHK::setCout ) );

	/////////////////////////////////////////////////////////////////////
  static Finfo* GHKFinfos[] =
    {
          &proc,			// Shared
	  &channel,			// Shared
	  &ghk,				// Shared
	  &Ik,				// ReadOnlyValue
	  &Gk,				// ReadOnlyValue
	  &Ek,				// ReadOnlyValue
	  &T,				// Value
	  &p,				// Value
	  &Vm,				// Value
	  &Cin,				// Value
	  &Cout,				// Value
	  &valency,				// Value
	  &CinDest,				// Dest
	  &CoutDest,			// Dest
	  &addPermeability,		// Dest
	  IkOut()				// Src
    };
  static string doc[] =
    {
      "Name", "GHK",
                "Author", "Johannes Hjorth, 2009, KTH, Stockholm",
                "Description", 
      "Calculates the Goldman-Hodgkin-Katz (constant field) equation "
      "for a single ionic species.  Provides current as well as "
      "reversal potential and slope conductance.",
    };

	static Cinfo GHKCinfo(
		"GHK",
		Neutral::initCinfo(),
		GHKFinfos, sizeof( GHKFinfos )/sizeof(Finfo *),
		new Dinfo< GHK >()
	);

	return &GHKCinfo;
}

static const Cinfo* GHKCinfo = GHK::initCinfo();

GHK::GHK()
	:	Ik_( 0.0 ), 
		Gk_( 0.0 ), 
		Ek_( 0.0 ), 
		p_( 0.0 ), 
		Cin_( 50e-6 ), 
		Cout_( 2 )
{;}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


double GHK::getEk() const
{
        return Ek_;
}


double GHK::getIk() const
{
        return Ik_;
}


double GHK::getGk() const
{
        return Gk_;
}


void GHK::setTemperature( double T )
{
        T_ = T;
}
double GHK::getTemperature() const
{
        return T_;
}


void GHK::setPermeability( double p )
{
	p_ = p;
}

void GHK::addPermeability( double p )
{
	p_ += p;
}

double GHK::getPermeability() const
{
        return p_;
}


void GHK::setVm( double Vm )
{
        Vm_ = Vm;
}
double GHK::getVm() const
{
        return Vm_;
}


void GHK::setCin( double Cin )
{
        Cin_ = Cin;
}
double GHK::getCin() const
{
        return Cin_;
}


void GHK::setCout( double Cout )
{
        Cout_ = Cout;
}
double GHK::getCout() const
{
        return Cout_;
}


void GHK::setValency( double valency )
{
        valency_ = valency;
}
double GHK::getValency() const
{
        return valency_;
}



///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


void GHK::handleVm( const Eref& e, const Qinfo* q, double Vm )
{
        Vm_ = Vm;
		VmOut()->send( e, q->threadNum(), Vm );
}

void GHK::process( const Eref& e, ProcPtr info )
{
	// Code for process adapted from original GENESIS ghk.c
	Ek_ = log(Cout_/Cin_)/GHKconst_;
	
	double exponent = GHKconst_*Vm_;
	double e_to_negexp = exp(-exponent);

	if ( fabs(exponent) < 0.00001 ) {
		/* exponent near zero, calculate current some other way */
		
		/* take Taylor expansion of V'/[exp(V') - 1], where
		* V' = constant * Vm
		*  First two terms should be enough this close to zero
		*/
	
		Ik_ = -valency_ * p_ * FARADAY *
		(Cin_ - (Cout_ * e_to_negexp)) / (1-0.5 * exponent);
	} else {       /* exponent far from zero, calculate directly */
		Ik_ = -p_ * FARADAY * valency_ * exponent *
		(Cin_ - (Cout_ * e_to_negexp)) / (1.0 - e_to_negexp);
	}

	/* Now calculate the chord conductance, but
	* check the denominator for a divide by zero.  */

	exponent = Ek_ - Vm_;
	if ( fabs(exponent) < 1e-12 ) {
		/* we are very close to the rest potential, so just set the
		* current and conductance to zero.  */
		Ik_ = Gk_ = 0.0;
	} else { /* calculate in normal way */
		Gk_ = Ik_ / exponent;
	}
	channelOut()->send( e, info->threadIndexInGroup, Gk_, Ek_ );
	IkOut()->send( e, info->threadIndexInGroup, Ik_ );

	// Set permeability to 0 at each timestep
	p_ = 0;
}

void GHK::reinit( const Eref& e, ProcPtr info )
{
	GHKconst_ =  F_OVER_R*valency_/ (T_ + ZERO_CELSIUS);

	if( fabs(valency_) == 0) {
		std::cerr << "GHK warning, valency set to zero" << std::endl;
	}

	if( Cin_ < 0) {
		std::cerr << "GHK error, invalid Cin set" << std::endl;
	}

	if( Cout_ < 0) {
		std::cerr << "GHK error, invalid Cout set" << std::endl;
	}

	if( T_ + ZERO_CELSIUS <= 0) {
		std::cerr << "GHK is freezing, please raise temperature" << std::endl;
	}

	if( p_ < 0) {
		std::cerr << "GHK error, invalid permeability" << std::endl;
	}

	channelOut()->send( e, info->threadIndexInGroup, Gk_, Ek_ );
	IkOut()->send( e, info->threadIndexInGroup, Ik_ );
}


/*
// This function should be called from TestBiophysics.cpp

void testGHK()
{
cout << "\nTesting GHK";

Element* n = Neutral::create( "Neutral", "n", Element::root()->id(),
                            Id::scratchId() );

Element* g = Neutral::create( "GHK", "ghk", n->id(), Id::scratchId() );

Element* chan = Neutral::create( "HHChannel", "Na", compt->id(),
                               Id::scratchId() );

bool ret = Eref( compt ).add( "channel", chan, "channel" );

// How do I connect the compartment -> HH -> GHK for unit testing
// I do not want HH to couple back to the compartment
}
*/
