#include <math.h>
#include "moose.h"
#include "Calculator.h"


const Cinfo* initCalculatorCinfo()
{
  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     RFCAST( &Calculator::processFunc ) ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &Calculator::reinitFunc ) ),
    };

  static Finfo* process =
    new SharedFinfo( "process", processShared,
                     sizeof( processShared ) / sizeof( Finfo* ) );

  static Finfo* CalculatorFinfos[] =
    {
      new ValueFinfo( "initValue", ValueFtype1< double >::global(),
                      GFCAST( &Calculator::getInitValue ),
                      RFCAST( &Calculator::setInitValue )
                      ),

      new ValueFinfo( "value", ValueFtype1< double >::global(),
                      GFCAST( &Calculator::getValue ),
                      &dummyFunc
                      ),

      process,

      new SrcFinfo( "valueSrc", Ftype1< double >::global() ),

      new DestFinfo( "mul", Ftype1< double >::global(),
                     RFCAST( &Calculator::mulValue ) ),
      new DestFinfo( "div", Ftype1< double >::global(),
                     RFCAST( &Calculator::divValue ) ),
      new DestFinfo( "add", Ftype1< double >::global(),
                     RFCAST( &Calculator::addValue ) ),
      new DestFinfo( "sub", Ftype1< double >::global(),
                     RFCAST( &Calculator::subValue ) ),


    };

  // We want the updates after the compartments are done.
  static SchedInfo schedInfo[] = { { process, 0, 1 } };


  static string doc[] =
    {
      "Name", "Calculator",
                "Author", "Johannes Hjorth, 2009, KTH, Stockholm",
                "Description",

      "Adds, subtracts, multiplies, and divides using messages.",
    };


  static Cinfo CalculatorCinfo(
                doc,
                sizeof( doc ) / sizeof( string ),
                initNeutralCinfo(),
                CalculatorFinfos,
                sizeof( CalculatorFinfos )/sizeof(Finfo *),
                ValueFtype1< Calculator >::global(),
                schedInfo, 1
        );

  return &CalculatorCinfo;


}

static const Cinfo* CalculatorCinfo = initCalculatorCinfo();


static const Slot valueSlot =
        initCalculatorCinfo()->getSlot( "valueSrc" );

void Calculator::setInitValue( const Conn* c, double v )
{
        static_cast< Calculator* >( c->data() )->initVal_ = v;
}

double Calculator::getInitValue( Eref e )
{
        return static_cast< Calculator* >( e.data() )->initVal_;
}

double Calculator::getValue( Eref e )
{
  // As we reset the val_ in each loop, we need to be able to show the
  // last timesteps calculated val_, ie prevVal_

  return static_cast< Calculator* >( e.data() )->prevVal_;
}


void Calculator::mulValue( const Conn* c, double factor )
{
  static_cast< Calculator* >( c->data() )->val_ *= factor;
}

void Calculator::divValue( const Conn* c, double divisor )
{
  static_cast< Calculator* >( c->data() )->val_ /= divisor;
}

void Calculator::addValue( const Conn* c, double term )
{
  static_cast< Calculator* >( c->data() )->val_ += term;
}

void Calculator::subValue( const Conn* c, double term )
{
  static_cast< Calculator* >( c->data() )->val_ -= term;
}


void Calculator::processFunc( const Conn* c, ProcInfo p )
{
        static_cast< Calculator* >( c->data() )->innerProcessFunc( c->target(), p );
}

void Calculator::innerProcessFunc( Eref e, ProcInfo info )
{
  //std::cerr << "Calculator sending out : " << val_ << std::endl;
  
  // Send the value out
  send1< double >( e, valueSlot, val_ );

  prevVal_ = val_;

  // Reset the value 
  val_ = initVal_;
}

void Calculator::reinitFunc( const Conn* c, ProcInfo p )
{
  static_cast< Calculator* >( c->data() )->innerReinitFunc( c->target(), p );
}

void Calculator::innerReinitFunc( Eref e, ProcInfo info )
{
  // Setting value to 0, since the first timestep will pass this value along
  // which causes problems if initVal_ is non-zero and the mul-message is at
  // or close to 0.
  val_ = 0; //initVal_;

}
