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
// #include "Music.h"
#include "InputEventChannel.h"

const Cinfo* initInputEventChannelCinfo()
{

  /**
   * This is a shared message to receive Process messages from
   * the scheduler objects.
   */

  /*
  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     RFCAST( &InputEventChannel::processFunc ) ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &InputEventChannel::reinitFunc ) ),
    };
  */

  static Finfo* process = 
    new SharedFinfo( "process", processShared,
                     sizeof( processShared ) / sizeof( Finfo* ) );


  static Finfo* inputEventChannelFinfos[] =
    {
      //////////////////////////////////////////////////////////////////
      // SharedFinfos
      //////////////////////////////////////////////////////////////////
      // process,

      ///////////////////////////////////////////////////////
      // MsgSrc definitions
      ///////////////////////////////////////////////////////
      // Sends out a trigger for an event. The time is not
      // sent - everyone knows the time.
      new SrcFinfo( "event", Ftype1< double >::global() ),
      

    };

  
  static Cinfo inputEventChannelCinfo("InputEventChannel",
                                      "Niraj Dudani and Johannes Hjorth",
                                      "InputEventChannel for communciation with the MUSIC API",
                                      initNeutralCinfo(),
                                      inputEventChannelFinfos,
                                      sizeof( inputEventChannelFinfos ) / sizeof( Finfo* ),
                                      ValueFtype1< InputEventChannel >::global() );
  
  
  return &inputEventChannelCinfo;

}

static const Cinfo* inputEventChannelCinfo = initInputEventChannelCinfo();

