/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class ProcInfoBase
{
	public:
		ProcInfoBase( const string& shell, 
			double dt = 1.0, double currTime = 0.0 )
			: currTime_( currTime ), dt_( dt ), shell_( shell )
		{
			;
		}

		Element* shell() {
			return Element::root()->relativeFind( shell_ );
		}

		void setResponse( const string& s ) {
			// Element* sh = Element::root()->relativeFind( shell_ );
			vector< string > vs;
			vs.push_back( s );
			Field sh( shell_ + "/echoIn" );
			if ( sh.good() )
				Ftype2< vector< string >*, int >::set( sh.getElement(), 
					sh.getFinfo(), &vs, 0 );
		}

		double currTime_;
		double dt_;

	private:
		string shell_;
};

typedef ProcInfoBase* ProcInfo;
