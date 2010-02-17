#ifndef _NCable_H
#define _NCable_H
#include <vector>
using namespace std;
class NCable
{
	protected:
		string id;
		string name;		
		vector< string > groups_;
	public:
		const std::string& getId() const;
		const std::string& getName() const;
		bool isSetId() const;
		bool isSetName() const;
		void setId(const std::string& id);
		void setName(const std::string& name);
		void unsetId();
		void unsetName();		
		vector< string > getGroups() const;
		void setGroups( string group );
		void unsetGroups();

};
#endif // _NCable_H
