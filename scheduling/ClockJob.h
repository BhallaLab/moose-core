#ifndef _ClockJob_h
#define _ClockJob_h
class ClockJob
{
	friend class ClockJobWrapper;
	public:
		ClockJob()
			: runTime_( 0.0 ), currentTime_( 0.0 ),
			nSteps_( 0 ), currentStep_( 0 )
		{
		}

	private:
		double runTime_;
		double currentTime_;
		int nSteps_;
		int currentStep_;
};
#endif // _ClockJob_h
