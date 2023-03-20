/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef  PROCINFO_INC
#define  PROCINFO_INC
class ProcInfo
{
    public:
        ProcInfo()
            : dt( 1.0 ), currTime( 0.0 ), status_( 0 )
        {;}
        double dt;
        double currTime;
		bool isFirstStep() const { return (status_==0x01); }; // first start
		bool isStart() const { return (status_ & 0x03); }; // any start call
		bool isContinue() const { return (status_ == 0x02); }; // Start call to continue from current nonzero time of simulation
		bool isReinit() const { return (status_ == 0x04); };
		void setFirstStep() { status_ = 0x01 ; };
		void setContinue() { status_ = 0x02 ; };
		void setRunning() { status_ = 0x0 ; };
		void setReinit() { status_ = 0x04 ; };

	private:
		unsigned int status_;	// bit 0: firstStep. bit 1: continue Bit 2: reinit.
};

typedef const ProcInfo* ProcPtr;
#endif   /* ----- #ifndef PROCINFO_INC  ----- */
