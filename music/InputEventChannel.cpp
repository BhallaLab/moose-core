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


  static Finfo* inputEventChannelFinfos[] =
    {
      ///////////////////////////////////////////////////////
      // MsgSrc definitions
      ///////////////////////////////////////////////////////
      new SrcFinfo( "event", Ftype1< double >::global() ),
      

    };

  static string doc[] =
	{
		"Name", "InputEventChannel",
		"Author", "Niraj Dudani and Johannes Hjorth",
		"Description", "InputEventChannel for communciation with the MUSIC API",
	};
  static Cinfo inputEventChannelCinfo(                                  
                                      doc,
				      sizeof( doc ) / sizeof( string ),                                      
				      initNeutralCinfo(),
                                      inputEventChannelFinfos,
                                      sizeof( inputEventChannelFinfos ) / sizeof( Finfo* ),
                                      ValueFtype1< InputEventChannel >::global() );
  
  
  return &inputEventChannelCinfo;

}

static const Cinfo* inputEventChannelCinfo = initInputEventChannelCinfo();

