#ifndef _CABLE_H
#define _CABLE_H
using namespace std;
class Cable
{
	protected:
		//string id;
		//string name_;		
		vector< string > groups_;
	public:
		//static void setId( const Conn* c,const std::string& id );		
		//static const std::string& getId( Eref );
		//static void setName( const Conn* c,std::string name );
		//static std::string getName( Eref );
		static void setGroups( const Conn* c,vector< string > group );
		static vector< string > getGroups( Eref );
		
		
};
extern const Cinfo* initCableCinfo();
#endif // _CABLE_H
