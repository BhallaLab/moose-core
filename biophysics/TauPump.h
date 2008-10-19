#ifndef _TAUPUMP_h
#define _TAUPUMP_h

class TauPump {

 public:

  TauPump() 
    {
      ;
    }

  static void setPumpRate( const Conn* c, double rate );
  static double getPumpRate( Eref );

  static void setEqConc( const Conn* c, double conc );
  static double getEqConc( Eref );

  static void setTA( const Conn* c, double ta );
  static double getTA( Eref );

  static void setTB( const Conn* c, double tb );
  static double getTB( Eref );

  static void setTV( const Conn* c, double tv );
  static double getTV( Eref );

  static void setTC( const Conn* c, double tc );
  static double getTC( Eref );
  
  static void setVm( const Conn* c, double Vm );

  static void processFunc( const Conn* c, ProcInfo p );
  static void reinitFunc( const Conn* c, ProcInfo p );

 private:

  void innerProcessFunc( Eref e, ProcInfo p );
  void innerReinitFunc( Eref e, ProcInfo p );


  double rate_, eqConc_, TA_, TB_, TC_, TV_, Vm_, tau_;
  int useVm_;

};

extern const Cinfo* initTauPumpCinfo();



#endif // _TAUPUMP_h
