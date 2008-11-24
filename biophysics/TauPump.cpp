#include <math.h>
#include "moose.h"
#include "TauPump.h"

const Cinfo* initTauPumpCinfo()
{
      
  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     RFCAST( &TauPump::processFunc ) ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &TauPump::reinitFunc ) ),
    };

  static Finfo* process = 
    new SharedFinfo( "process", processShared,
                     sizeof( processShared ) / sizeof( Finfo* ) );

	//We receive Vm from the compartment, and we send pump rate (kP) and equilibrium concentration to diffshell
  static Finfo* TauPumpFinfos[] =
    {
      new ValueFinfo( "pumpRate", ValueFtype1< double >::global(),
                      GFCAST( &TauPump::getPumpRate ),
                      RFCAST( &TauPump::setPumpRate )
                      ),
      new ValueFinfo( "eqConc", ValueFtype1< double >::global(),
                      GFCAST( &TauPump::getEqConc ),
                      RFCAST( &TauPump::setEqConc )
                      ),
      new ValueFinfo( "TA", ValueFtype1< double >::global(),
                      GFCAST( &TauPump::getTA ),
                      RFCAST( &TauPump::setTA )
                      ),
      new ValueFinfo( "TB", ValueFtype1< double >::global(),
                      GFCAST( &TauPump::getTB ),
                      RFCAST( &TauPump::setTB )
                      ),
      new ValueFinfo( "TC", ValueFtype1< double >::global(),
                      GFCAST( &TauPump::getTC ),
                      RFCAST( &TauPump::setTC )
                      ),
      new ValueFinfo( "TV", ValueFtype1< double >::global(),
                      GFCAST( &TauPump::getTV ),
                      RFCAST( &TauPump::setTV )
                      ),
      process,
      
      new SrcFinfo( "pumpData", Ftype2< double, double >::global() ),

      new DestFinfo( "Vm", Ftype1< double >::global(),
                     RFCAST( &TauPump::setVm ) ),

    };

  // !!! Is this the right place in the scheduling?
  static SchedInfo schedInfo[] = { { process, 0, 1 } };

  static string doc[] =
	{
		"Name", "TauPump",
		"Author", "Johannes Hjorth, 2008, Stockholm Brain Institute, KTH",
		"Description", "TauPump:: Implementation of a simple pump with a variable time "
				"constant of removal. Should be coupled to a difshell, where "
				"the change in concentration is computed.",
	};

  static Cinfo TauPumpCinfo(
                doc,
		sizeof( doc ) / sizeof( string ),                
		initNeutralCinfo(),
                TauPumpFinfos,
                sizeof( TauPumpFinfos )/sizeof(Finfo *),
                ValueFtype1< TauPump >::global(),
                schedInfo, 1				
        );

  return &TauPumpCinfo;

}

static const Cinfo* tauPumpCinfo = initTauPumpCinfo();


static const Slot pumpDataSlot =
  initTauPumpCinfo()->getSlot( "pumpData" );


void TauPump::setPumpRate( const Conn* c, double pumpRate )
{
        static_cast< TauPump* >( c->data() )->rate_ = pumpRate;
}
double TauPump::getPumpRate( Eref e )
{
        return static_cast< TauPump* >( e.data() )->rate_;
}



void TauPump::setEqConc( const Conn* c, double eqConc )
{
        static_cast< TauPump* >( c->data() )->eqConc_ = eqConc;
}
double TauPump::getEqConc( Eref e )
{
        return static_cast< TauPump* >( e.data() )->eqConc_;
}

void TauPump::setTA( const Conn* c, double TA )
{
        static_cast< TauPump* >( c->data() )->TA_ = TA;
}
double TauPump::getTA( Eref e )
{
        return static_cast< TauPump* >( e.data() )->TA_;
}

void TauPump::setTB( const Conn* c, double TB )
{
        static_cast< TauPump* >( c->data() )->TB_ = TB;
}
double TauPump::getTB( Eref e )
{
        return static_cast< TauPump* >( e.data() )->TB_;
}

void TauPump::setTC( const Conn* c, double TC )
{
        static_cast< TauPump* >( c->data() )->TC_ = TC;
}
double TauPump::getTC( Eref e )
{
        return static_cast< TauPump* >( e.data() )->TC_;
}

void TauPump::setTV( const Conn* c, double TV )
{
        static_cast< TauPump* >( c->data() )->TV_ = TV;
}
double TauPump::getTV( Eref e )
{
        return static_cast< TauPump* >( e.data() )->TV_;
}

void TauPump::setVm( const Conn* c, double Vm )
{
        static_cast< TauPump* >( c->data() )->Vm_ = Vm;
}





void TauPump::processFunc( const Conn* c, ProcInfo p )
{
  static_cast< TauPump* >( c->data() )->innerProcessFunc( c->target(), p );
}

void TauPump::innerProcessFunc( Eref e, ProcInfo info )
{

  // Check that there is a voltage message in reinit.

  if(useVm_) 
    {
      tau_ = TA_*exp((Vm_ - TV_)/TB_) + TC_;
    }
  else 
    {
      if(TC_ != 0)
        {
          tau_ = TC_;
        }
      else
        {
          tau_ = TA_;
        }
    }

  rate_ = 1.0/tau_;
}

void TauPump::reinitFunc( const Conn* c, ProcInfo p )
{
  static_cast< TauPump* >( c->data() )->innerReinitFunc( c->target(), p );
}

void TauPump::innerReinitFunc( Eref e, ProcInfo info )
{
  // If there are atleast one Vm message then we will use it.
  useVm_ = e->numTargets("Vm");

}



