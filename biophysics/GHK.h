#ifndef _GHK_H
#define _GHK_H

#define GAS_CONSTANT 8.314472
#define FARADAY  96485.3399
#define ZERO_CELSIUS 273.15
#define R_OVER_F        8.617342e-5            /* volt/deg */
#define F_OVER_R        11604.506              /* deg/volt */


class GHK {

 public:

  GHK() :
  Gk_( 0.0 ), Ek_( 0.0 ), p_( 0.0 )
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

  double Ik_, p_, Gk_, Ek_, T_, Vm_, Cin_, Cout_, valency_,GHKconst_;

};

extern const Cinfo* initGHKCinfo();



#endif // _GHK_H
