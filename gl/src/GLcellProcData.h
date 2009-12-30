#ifndef GLCELLPROCDATA_H
#define GLCELLPROCDATA_H

struct GLcellProcData
{
	std::string strName;
	unsigned int id;
	std::string strPathName;
	std::vector< unsigned int > vecNeighbourIds;

	double diameter;
	double length;
	double x0;
	double y0;
	double z0;
	double x;
	double y;
	double z;
	
	template< typename Archive > 
	void serialize( Archive& ar, const unsigned int version )
	{
		ar & strName;
		ar & id;
		ar & strPathName;
		ar & vecNeighbourIds;
		
		ar & diameter;
		ar & length;
		ar & x0;
		ar & y0;
		ar & z0;
		ar & x;
		ar & y;
		ar & z;
	}
};

#endif // GLCELLPROCDATA_H

