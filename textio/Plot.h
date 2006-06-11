#ifndef _Plot_h
#define _Plot_h
class Plot
{
	friend class PlotWrapper;
	public:
		Plot()
		{
			currTime_ = 0.0;
			plotName_ = "data";
			jagged_ = 0;
		}

	private:
		double currTime_;
		string plotName_;
		int npts_;
		int jagged_;
		vector < double > x_;
		vector < double > y_;
		bool prime_;
};
#endif // _Plot_h
