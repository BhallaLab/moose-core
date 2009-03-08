#ifndef _MUSIC_OUTPUT_EVENT_PORT_H
#define _MUSIC_OUTPUT_EVENT_PORT_H


class OutputEventPort 
{

 public:
  OutputEventPort() 
    {
      maxBuffered_ = 1;
      isMapped_ = 0;
    }

  //////////////////////////////////////////////////////////////////
  // Message dest functions.
  //////////////////////////////////////////////////////////////////
  
  static void reinitFunc( const Conn* c, ProcInfo p );

  static void initialiseFunc( const Conn* c,
                              unsigned int width, 
                              unsigned int offset,
                              MUSIC::EventOutputPort* mPort);

  static unsigned int getWidth( Eref e );
  static unsigned int getIsConnected( Eref e);

  static int getMaxBuffered(Eref e);
  static void setMaxBuffered(const Conn* c, int maxBuffered);


 protected:

 private:

  MUSIC::EventOutputPort* mPort_;

  int maxBuffered_;
  unsigned int myOffset_, myWidth_;
  bool isMapped_;

  vector < Eref > channels_;

  void innerReinitFunc();

  void innerInitialiseFunc( Eref e, 
                            unsigned int width, 
                            unsigned int offset,
                            MUSIC::EventOutputPort* mPort);

};



#endif // _MUSIC_OUTPUT_EVENT_PORT_H
