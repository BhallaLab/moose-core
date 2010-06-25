#ifndef _IONPOOL_H
#define _IONPOOL_H
using namespace std;
class PoolDefinition
{
	public:
		PoolDefinition():
			setpoolnamespace(false)			
		{;}
		void readDefinition( std::string& filename);
		xmlXPathObjectPtr getnodeset (xmlChar *xpath); 		
		int register_PoolNamespaces();
		bool issetPoolNamespaces() ;	
		const std::string& getIon() const;
		void setIon(const std::string& ion);
		double getCharge();
		void setCharge(double value );
		const std::string& getPoolName() const;
		void setPoolName(const std::string& name);
		const std::string& getStatus() const;
		void setStatus(const std::string& status);
		double getResting_conc();
		void setResting_conc(double value );
		double getDecay_constant();
		void setDecay_constant(double value );
		double getShell_thickness();
		void setShell_thickness(double value );
	private:
		string ion;
		double charge;
		string pool_name;
		string status;
		double resting_conc;
		double decay_constant;
		double shell_thickness;
		xmlDocPtr docptr;
		xmlXPathContextPtr cxtptr;
		xmlTextReaderPtr rdrptr;
		bool setpoolnamespace;
		
};
class IonPool
{
	public:
		IonPool() {;}	
		~IonPool() {;}	
		const std::string& getName() const;
		void setName(const std::string& name);
		bool isSetName() const;
		void unsetName();
		const double getScalingFactor()const;
		void setScalingFactor(double value);
		const std::string& getScaling() const;
		void setScaling(const std::string& scale);
		vector< string > getGroups() const;
		void setGroups( string group );
		void unsetGroups();
		const std::string& getIon() const;
		void setIon(const std::string& ion);
		double getCharge();
		void setCharge(double value );
		const std::string& getPoolName() const;
		void setPoolName(const std::string& name);
		const std::string& getStatus() const;
		void setStatus(const std::string& status);
		double getResting_conc();
		void setResting_conc(double value );
		double getDecay_constant();
		void setDecay_constant(double value );
		double getShell_thickness();
		void setShell_thickness(double value );
		vector< string > getSegGroups() const;
		void setSegGroups( string seg );
		void unsetSegGroups();
	
	private:
		string name;		
		double B;
		vector< string > groups;
		vector< string > segGroups;
		string scaling;
		PoolDefinition* pooldefinition_;
		static map< std::string,PoolDefinition > lookup_ ;
		static PoolDefinition* lookupDefinition( const std::string& name );
};
#endif // _IONPOOL_H
