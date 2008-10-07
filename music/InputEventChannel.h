#ifndef _MUSIC_INPUT_EVENT_CHANNEL_H
#define _MUSIC_INPUT_EVENT_CHANNEL_H

class InputEventChannel
{

 public:
  InputEventChannel() 
    {

    }

  //////////////////////////////////////////////////////////////////
  // Message dest functions.
  //////////////////////////////////////////////////////////////////

  void innerProcessFunc( const Conn* c, ProcInfo p );
  static void processFunc( const Conn* c, ProcInfo p );

  static void reinitFunc( const Conn* c, ProcInfo p );

 protected:

 private:

};

extern const Cinfo* initInputEventChannelCinfo();



#endif // _MUSIC_INPUT_EVENT_CHANNEL_H
