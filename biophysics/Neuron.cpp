/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Neuron.h"


/**
 * The initCinfo() function sets up the Compartment class.
 * This function uses the common trick of having an internal
 * static value which is created the first time the function is called.
 * There are several static arrays set up here. The ones which
 * use SharedFinfos are for shared messages where multiple kinds
 * of information go along the same connection.
 */
const Cinfo* Neuron::initCinfo()
{
	static Finfo* neuronFinfos[] = 
	{ 
	  NULL

	};
	static string doc[] =
	{
		"Name", "Neuron",
		"Author", "C H Chaitanya",
		"Description", "Neuron - A compartment container",
	};	
	static Cinfo neuronCinfo(
				"Neuron",
				Neutral::initCinfo(),
				NULL,
				0,
				new Dinfo< Neuron >(),
                doc,
                sizeof(doc)/sizeof(string)
	);

	return &neuronCinfo;
}

static const Cinfo* neuronCinfo = Neuron::initCinfo();
