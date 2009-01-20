#ifndef _MUSIC_OUTPUT_EVENT_CHANNEL_H
#define _MUSIC_OUTPUT_EVENT_CHANNEL_H

class OutputEventChannel
{

 public:
 OutputEventChannel() : localId_(MUSIC::LocalIndex(0))
    {

    }

  static void insertEvent(const Conn* c, double time);

  static void initialise(const Conn*, unsigned int id, 
                         MUSIC::EventOutputPort* mPort);

  //////////////////////////////////////////////////////////////////
  // Message dest functions.
  //////////////////////////////////////////////////////////////////

 protected:

 private:

  MUSIC::LocalIndex localId_;
  MUSIC::EventOutputPort* mPort_;

  void innerInsertEvent(double time);

};

extern const Cinfo* initOutputEventChannelCinfo();



#endif // _MUSIC_OUTPUT_EVENT_CHANNEL_H
