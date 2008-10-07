#ifndef _MUSIC_INPUT_EVENT_PORT_H
#define _MUSIC_INPUT_EVENT_PORT_H

class event_handler {
 public:
  virtual void operator () (double t, int id) = 0;
  
};


//class InputEventPort : public MUSIC::event_handler
class InputEventPort : public event_handler
{

 public:
  InputEventPort() 
    {
      
    }

  void operator () ( double t, int id );

  //////////////////////////////////////////////////////////////////
  // Message dest functions.
  //////////////////////////////////////////////////////////////////
  
  static void reinitFunc( const Conn* c, ProcInfo p );

  static void setWidth( const Conn* c, unsigned int width);
  static unsigned int getWidth( const Conn* c);


 protected:

 private:

  vector < Id > channels_;

  void innerSetWidth( Eref e, unsigned int width);


};



#endif // _MUSIC_INPUT_EVENT_PORT_H
