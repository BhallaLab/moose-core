#ifndef _MUSIC_INPUT_EVENT_PORT_H
#define _MUSIC_INPUT_EVENT_PORT_H


class InputEventPort : public MUSIC::event_handler_local_index
{

 public:
  InputEventPort() 
    {
      maxBuffered_ = 100;
	  accLatency_ = 1e-3;
    }

  void operator () ( double t, MUSIC::local_index id );

  //////////////////////////////////////////////////////////////////
  // Message dest functions.
  //////////////////////////////////////////////////////////////////
  
  static void reinitFunc( const Conn* c, ProcInfo p );

  static void initialiseFunc( const Conn* c,
                              unsigned int width, 
                              unsigned int offset,
                              MUSIC::event_input_port* mPort);

  static unsigned int getWidth( Eref e);
  static unsigned int getIsConnected( Eref e);

  static double getAccLatency(Eref e);
  static void setAccLatency(const Conn* c, double accLatency);
  static int getMaxBuffered(Eref e);
  static void setMaxBuffered(const Conn* c, int maxBuffered);

 protected:

 private:

  MUSIC::event_input_port* mPort_;
  vector < Id > channels_;

  unsigned int myOffset_, myWidth_;
  double accLatency_;
  int maxBuffered_;


  void innerInitialiseFunc( Eref e, unsigned int width, unsigned int offset,
                            MUSIC::event_input_port* mPort);

  void innerReinitFunc();

};



#endif // _MUSIC_INPUT_EVENT_PORT_H
