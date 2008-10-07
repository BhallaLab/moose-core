/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
**             and Niraj Dudani and Johannes Hjorth, KTH, Stockholm
**
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Music.h"
#include "InputEventPort.h"
#include "InputEventChannel.h"
#include <sstream>
#include "element/Neutral.h"

const Cinfo* initInputEventPortCinfo()
{

  /**
   * This is a shared message to receive Process messages from
   * the scheduler objects.
   */

  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     dummyFunc ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &InputEventPort::reinitFunc ) )
    };

  static Finfo* process = 
    new SharedFinfo( "process", processShared,
                     sizeof( processShared ) / sizeof( Finfo* ) );


  static Finfo* inputEventPortFinfos[] =
    {
      new ValueFinfo( "width", ValueFtype1< unsigned int >::global(),
                      GFCAST( &InputEventPort::getWidth ),
                      RFCAST( &InputEventPort::setWidth )
                      ),

      //////////////////////////////////////////////////////////////////
      // SharedFinfos
      //////////////////////////////////////////////////////////////////
      process,

    };

  
  static Cinfo inputEventPortCinfo("InputEventPort",
                             "Niraj Dudani and Johannes Hjorth",
                             "InputEventPort for communciation with the MUSIC API",
                             initNeutralCinfo(),
                             inputEventPortFinfos,
                             sizeof( inputEventPortFinfos ) / sizeof( Finfo* ),
                             ValueFtype1< InputEventPort >::global() );
  
  
  return &inputEventPortCinfo;

}

static const Cinfo* inputEventPortCinfo = initInputEventPortCinfo();

// This is from Channel and not from the port
static const Slot eventSlot =
        initInputEventChannelCinfo()->getSlot( "event" );



void InputEventPort::reinitFunc( const Conn* c, ProcInfo p ) 
{
  
}

void InputEventPort::operator () ( double t, int id ) 
{
  send1 < double > ( channels_[id](), eventSlot, t );

}


void InputEventPort::setWidth( const Conn* c, unsigned int width)
{
  static_cast < InputEventPort* > (c->data())->innerSetWidth(c->target(),
                                                             width);
}

void InputEventPort::innerSetWidth( Eref e, unsigned int width) 
{
  if(channels_.size() > width)
    {
      cerr << "InputEventPort::setWidth can not reduce number of channels"
           << " from " << channels_.size() << " to " << width << endl;
      return;
    }

  for(unsigned int i = channels_.size(); i < width; i++)
    {
      ostringstream name;

      name << "channel[" << i << "]";

      Element* channel = Neutral::create( "InputEventChannel", name.str(),
                                          e.id(), Id::scratchId() );
      channels_.push_back(channel->id());
    }

}

unsigned int InputEventPort::getWidth( const Conn* c) 
{
  return static_cast < InputEventPort* > (c->data())->channels_.size();
}
