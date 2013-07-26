/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TRY_CONN_H
#define _TRY_CONN_H

class Sender
{
	public:
		Sender();

		// Note that nothing actually says that these things are destFinfos, 
		// except the initCinfo function.
		void reinit( const Eref&, ProcPtr info );
		void process( const Eref&, ProcPtr info );

		static const Cinfo *initCinfo();

		// These are required for ValueFinfos, I think.
		// They will again be registered as ValueFinfos in initCinfo.
		double getX() const;
		void setX( double );

		// This is a srcFinfo, meaning it can send out values.
		// The function simply returns a pointer to a SrcFinfo1 object.
		// The actual sending of data happens in the reinit and process
		// functions.
		static SrcFinfo1< double >* XOut();

	private:
		double X_; // Some random value field;
};

class Receiver
{
	public:
		Receiver() {};

		// This class's objects don't need to be processed at every time step.
		// They simply capture and print the value of X every time it is given
		// by Sender.
		static const Cinfo *initCinfo();

		// I don't need anything other than a destFinfo to handle the received
		// X value here.
		void handleX( double );
};

#endif // _TRY_CONN_H
