#ifndef _Enzyme_h
#define _Enzyme_h
class Enzyme
{
	friend class EnzymeWrapper;
	public:
		Enzyme()
		{
			k1_ = 0.1;
			k2_ = 0.4;
			k3_ = 0.1;
			sk1_ = 1.0;
			Km_ = 5;
		}

	private:
		double k1_;
		double k2_;
		double k3_;
		double Km_;
		double sA_;
		double pA_;
		double eA_;
		double B_;
		double e_;
		double s_;
		double sk1_;
		void innerSetKm( double Km ) {
			Km_ = Km;
			k1_ = ( k2_ + k3_ ) / Km;
		}
};
#endif // _Enzyme_h
