#ifndef _MyClass_h
#define _MyClass_h
class MyClass: public MyBaseClass
{
	friend class MyClassWrapper;
	public:
		MyClass()
		{
			Cm = 1e-6;
			values.reserve( 10 ) ;
		}

	private:
		double Vm_;
		double Cm_;
		double Rm_;
		const double pi_;
		double Ra_;
		double inject_;
		vector < double > coords_;
		vector < double > values_;
};
#endif // _MyClass_h
