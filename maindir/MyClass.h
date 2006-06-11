#ifndef _MyClass_h
#define _MyClass_h
class MyClass
{
	friend class MyClassWrapper;
	public:
		MyClass()
		{
			Cm_ = 1e-6;
			values_.reserve( 10 ) ;
		}

	private:
		double Vm_;
		double Cm_;
		double Rm_;
		static const double pi_;
		double Ra_;
		double inject_;
		vector < double > coords_;
		vector < double > values_;
		double I_;
		double Ca_;
		double volscale_;
		double Erest_;
};
#endif // _MyClass_h
