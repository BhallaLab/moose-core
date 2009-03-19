/*
 * NOT FULLY TESTED YET.
 *
 *
 */


#include <math.h>
#include "moose.h"
#include "GHK.h"

const Cinfo* initGHKCinfo()
{

  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     RFCAST( &GHK::processFunc ) ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &GHK::reinitFunc ) ),
    };

  static Finfo* process =
    new SharedFinfo( "process", processShared,
                     sizeof( processShared ) / sizeof( Finfo* ) );

  static Finfo* channelShared[] =
    {
      new SrcFinfo( "channel", Ftype2< double, double >::global() ),
      new DestFinfo( "Vm", Ftype1< double >::global(),
                     RFCAST( &GHK::channelFunc ) ),
    };
  
  static Finfo* ghkShared[] =
    {
      new SrcFinfo( "Vm", Ftype1< double >::global() ),
      new DestFinfo( "permeability", Ftype1< double >::global(),
                     RFCAST( &GHK::addPermeability ) ),
    };

  //!! Need to add channelFunc

  static Finfo* GHKFinfos[] =
    {
      new ValueFinfo( "Ik", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getIk ),
                      &dummyFunc
                      ),
      new ValueFinfo( "Gk", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getGk ),
                      &dummyFunc
                      ),
      new ValueFinfo( "Ek", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getEk ),
                      &dummyFunc
                      ),
      new ValueFinfo( "T", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getTemperature ),
                      RFCAST( &GHK::setTemperature )
                      ),
      new ValueFinfo( "p", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getPermeability ),
                      RFCAST( &GHK::setPermeability )
                      ),
      new ValueFinfo( "Vm", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getVm ),
                      RFCAST( &GHK::setVm )
                      ),
      new ValueFinfo( "Cin", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getCin ),
                      RFCAST( &GHK::setCin )
                      ),
      new ValueFinfo( "Cout", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getCout ),
                      RFCAST( &GHK::setCout )
                      ),
      new ValueFinfo( "valency", ValueFtype1< double >::global(),
                      GFCAST( &GHK::getValency ),
                      RFCAST( &GHK::setValency )
                      ),
      process,

      new SrcFinfo( "IkSrc", Ftype1< double >::global() ),

      new DestFinfo( "CinDest", Ftype1< double >::global(),
                     RFCAST( &GHK::setCin ) ),
      new DestFinfo( "CoutDest", Ftype1< double >::global(),
                     RFCAST( &GHK::setCout ) ),
      new DestFinfo( "pDest", Ftype1< double >::global(),      
                     RFCAST( &GHK::addPermeability ) ),
      new SharedFinfo( "channel", channelShared,
                       sizeof( channelShared ) / sizeof( Finfo* ),
                       "This is a shared message to couple channel to compartment. "
                       "The first entry is a MsgSrc to send Gk and Ek to the compartment "
                       "The second entry is a MsgDest for Vm from the compartment." ),
      new SharedFinfo( "ghk", ghkShared,
                       sizeof( ghkShared ) / sizeof( Finfo* ),
                       "This shared message connects to an HHChannel. "
					   "The first entry is a MsgSrc which relays the Vm received from "
					   "a compartment. The second entry is a MsgDest which receives "
					   "channel conductance, and interprets it as permeability." ),
    };

  // Order of updates: (t0) Compartment -> (t1) HHChannel -> (t2) GHK
  static SchedInfo schedInfo[] = { { process, 0, 2 } };

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
                doc,
                sizeof( doc ) / sizeof( string ),
                initNeutralCinfo(),
                GHKFinfos,
                sizeof( GHKFinfos )/sizeof(Finfo *),
                ValueFtype1< GHK >::global(),
                schedInfo, 1
        );

  return &GHKCinfo;

}

static const Cinfo* GHKCinfo = initGHKCinfo();


static const Slot ikSlot =
        initGHKCinfo()->getSlot( "IkSrc" );

static const Slot channelSlot =
        initGHKCinfo()->getSlot( "channel.channel" );

static const Slot vmSlot =
        initGHKCinfo()->getSlot( "ghk.Vm" );


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


double GHK::getEk( Eref e )
{
        return static_cast< GHK* >( e.data() )->Ek_;
}


double GHK::getIk( Eref e )
{
        return static_cast< GHK* >( e.data() )->Ik_;
}


double GHK::getGk( Eref e )
{
        return static_cast< GHK* >( e.data() )->Gk_;
}


void GHK::setTemperature( const Conn* c, double T )
{
        static_cast< GHK* >( c->data() )->T_ = T;
}
double GHK::getTemperature( Eref e )
{
        return static_cast< GHK* >( e.data() )->T_;
}


void GHK::setPermeability( const Conn* c, double p )
{
    static_cast< GHK* >( c->data() )->p_ = p;
}

void GHK::addPermeability( const Conn* c, double p )
{
  static_cast< GHK* >( c->data() )->p_ += p;
  
}

double GHK::getPermeability( Eref e )
{
        return static_cast< GHK* >( e.data() )->p_;
}


void GHK::setVm( const Conn* c, double Vm )
{
        static_cast< GHK* >( c->data() )->Vm_ = Vm;
}
double GHK::getVm( Eref e )
{
        return static_cast< GHK* >( e.data() )->Vm_;
}


void GHK::setCin( const Conn* c, double Cin )
{
        static_cast< GHK* >( c->data() )->Cin_ = Cin;
}
double GHK::getCin( Eref e )
{
        return static_cast< GHK* >( e.data() )->Cin_;
}


void GHK::setCout( const Conn* c, double Cout )
{
        static_cast< GHK* >( c->data() )->Cout_ = Cout;
}
double GHK::getCout( Eref e )
{
        return static_cast< GHK* >( e.data() )->Cout_;
}


void GHK::setValency( const Conn* c, double valency )
{
        static_cast< GHK* >( c->data() )->valency_ = valency;
}
double GHK::getValency( Eref e )
{
        return static_cast< GHK* >( e.data() )->valency_;
}


void GHK::channelFunc( const Conn* c, double Vm )
{
        static_cast< GHK* >( c->data() )->Vm_ = Vm;
		send1< double >( c->target(), vmSlot, Vm );
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


void GHK::processFunc( const Conn* c, ProcInfo p )
{
        static_cast< GHK* >( c->data() )->innerProcessFunc( c->target(), p );
}


void GHK::innerProcessFunc( Eref e, ProcInfo info )
{
  // Code for process adapted from original GENESIS ghk.c

  Ek_ = log(Cout_/Cin_)/GHKconst_;

  double exponent = GHKconst_*Vm_;
  double e_to_negexp = exp(-exponent);


  if ( abs(exponent) < 0.00001 ) {
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
    if ( abs(exponent) < 1e-12 ) {
      /* we are very close to the rest potential, so just set the
       * current and conductance to zero.  */

      Ik_ = Gk_ = 0.0;
    } else { /* calculate in normal way */
      Gk_ = Ik_ / exponent;
    }

    send2< double, double >( e, channelSlot, Gk_, Ek_ );
    send1< double >( e, ikSlot, Ik_ );

    // Set permeability to 0 at each timestep
    p_ = 0;

}



void GHK::reinitFunc( const Conn* c, ProcInfo p )
{
  static_cast< GHK* >( c->data() )->innerReinitFunc( c->target(), p );
}

void GHK::innerReinitFunc( Eref e, ProcInfo info )
{
  GHKconst_ =  F_OVER_R*valency_/ (T_ + ZERO_CELSIUS);

  if(abs(valency_) == 0) {
    std::cerr << "GHK warning, valency set to zero" << std::endl;
  }

  if(Cin_ < 0) {
    std::cerr << "GHK error, invalid Cin set" << std::endl;
  }

  if(Cout_ < 0) {
    std::cerr << "GHK error, invalid Cout set" << std::endl;
  }

  if(T_ + ZERO_CELSIUS <= 0) {
    std::cerr << "GHK is freezing, please raise temperature" << std::endl;
  }

  if(p_ < 0) {
    std::cerr << "GHK error, invalid permeability" << std::endl;
  }

    send2< double, double >( e, channelSlot, Gk_, Ek_ );
    send1< double >( e, ikSlot, Ik_ );

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















