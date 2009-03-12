#ifndef _GHK_H
#define _GHK_H

#define GAS_CONSTANT	8.314			/* (V * C)/(deg K * mol) */
#define FARADAY		9.6487e4			/* C / mol */
#define ZERO_CELSIUS	273.15			/* deg */
#define R_OVER_F        8.6171458e-5		/* volt/deg */
#define F_OVER_R        1.1605364e4		/* deg/volt */


class GHK {

 public:

  GHK() :
  Ik_( 0.0 ), Gk_( 0.0 ), Ek_( 0.0 ), p_( 0.0 ), Cin_( 50e-6 ), Cout_( 2 )
    {
      ;
    }

  static double getIk( Eref );
  static double getGk( Eref );
  static double getEk( Eref );

  static void setTemperature( const Conn* c, double T );
  static double getTemperature( Eref );

  static void setPermeability( const Conn* c, double p );
  static double getPermeability( Eref );
  static void addPermeability( const Conn* c, double p );


  static void setVm( const Conn* c, double Vm );
  static double getVm( Eref );

  static void setCin( const Conn* c, double Cin );
  static double getCin( Eref );

  static void setCout( const Conn* c, double Cout );
  static double getCout( Eref );

  static void setValency( const Conn* c, double valecny );
  static double getValency( Eref );

  static void processFunc( const Conn* c, ProcInfo p );
  static void reinitFunc( const Conn* c, ProcInfo p );

  static void channelFunc( const Conn* c, double Vm );

 private:

  void innerProcessFunc( Eref e, ProcInfo p );
  void innerReinitFunc( Eref e, ProcInfo p );

  double Ik_, Gk_, Ek_, p_, T_, Vm_, Cin_, Cout_, valency_,GHKconst_;

};

extern const Cinfo* initGHKCinfo();



#endif // _GHK_H
