/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef ACKPICKDATA_H
#define ACKPICKDATA_H

// This structure is so named because it acts both as the acknowledgement message
// from glclient to GLcell (in response to the latter's PROCESS or PROCESSSYNC
// message to the former) and as a container for the id of any compartment that the
// user may have picked with the mouse in glcellclient's display window.
//
// If there has been a picking event, the first element stores true and the second
// stores the id of the compartment that was picked. If there has been no picking
// event, the first element stores false and the second stores zero.

struct AckPickData
{
	int msgType;
	bool wasSomethingPicked;
	unsigned int idPicked;

	template< typename Archive >
	void serialize( Archive& ar, const unsigned int version)
	{
		ar & msgType;
		ar & wasSomethingPicked;
		ar & idPicked;
	}
};

#endif // ACKPICKDATA_H
