#ifndef _MUSIC_OUTPUT_EVENT_CHANNEL_H
#define _MUSIC_OUTPUT_EVENT_CHANNEL_H

class OutputEventChannel
{

 public:
 OutputEventChannel() : localId_(MUSIC::local_index(0))
    {

    }

  static void insertEvent(const Conn* c, double time);

  static void initialise(const Conn*, unsigned int id, 
                         MUSIC::event_output_port* mPort);

  //////////////////////////////////////////////////////////////////
  // Message dest functions.
  //////////////////////////////////////////////////////////////////

 protected:

 private:

  MUSIC::local_index localId_;
  MUSIC::event_output_port* mPort_;

  void innerInsertEvent(double time);

};

extern const Cinfo* initOutputEventChannelCinfo();



#endif // _MUSIC_OUTPUT_EVENT_CHANNEL_H
