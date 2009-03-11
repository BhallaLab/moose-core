#ifndef _CALCULATOR_H
#define _CALCULATOR_H

class Calculator {

 public:
  Calculator()
    {
     initVal_ = 0;
     val_ = 0;
     prevVal_ = 0;
    }

  static void setInitValue( const Conn* c, double v );
  static double getInitValue( Eref );
  static double getValue( Eref );

  static void processFunc( const Conn* c, ProcInfo p );
  static void reinitFunc( const Conn* c, ProcInfo p );

  static void mulValue( const Conn* c, double factor );
  static void divValue( const Conn* c, double divisor );
  static void addValue( const Conn* c, double term );
  static void subValue( const Conn* c, double term );

 private:

  void innerProcessFunc( Eref e, ProcInfo p );
  void innerReinitFunc( Eref e, ProcInfo p );

  double val_,initVal_,prevVal_;
  

};

extern const Cinfo* initCalculatorCinfo();



#endif
