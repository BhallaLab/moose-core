#ifndef _MUSIC_INPUT_EVENT_PORT_H
#define _MUSIC_INPUT_EVENT_PORT_H


class InputEventPort : public MUSIC::EventHandlerLocalIndex
{

 public:
  InputEventPort() 
    {
      maxBuffered_ = 100;
      accLatency_ = 1e-3;
      isMapped_ = 0;
    }

  virtual ~InputEventPort() { ; }

  void operator () ( double t, MUSIC::LocalIndex id );

  //////////////////////////////////////////////////////////////////
  // Message dest functions.
  //////////////////////////////////////////////////////////////////
  
  static void reinitFunc( const Conn* c, ProcInfo p );

  static void initialiseFunc( const Conn* c,
                              unsigned int width, 
                              unsigned int offset,
                              MUSIC::EventInputPort* mPort);

  static unsigned int getWidth( Eref e);
  static unsigned int getIsConnected( Eref e);

  static double getAccLatency(Eref e);
  static void setAccLatency(const Conn* c, double accLatency);
  static int getMaxBuffered(Eref e);
  static void setMaxBuffered(const Conn* c, int maxBuffered);

 protected:

 private:

  MUSIC::EventInputPort* mPort_;
  vector < Eref > channels_;

  unsigned int myOffset_, myWidth_;
  double accLatency_;
  int maxBuffered_;
  bool isMapped_;

  void innerInitialiseFunc( Eref e, unsigned int width, unsigned int offset,
                            MUSIC::EventInputPort* mPort);

  void innerReinitFunc();

};



#endif // _MUSIC_INPUT_EVENT_PORT_H
