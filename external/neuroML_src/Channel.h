#ifndef _CHANNEL_H
#define _CHANNEL_H
#include <map>	// Somehow this is needed in gcc 4.2.4, even if it is included in the .cpp file.
using namespace std;
/**
 * 
 * 
 */
struct Transition
{
	string name;
	string from;
	string to;
	string expr_form;
	double rate;
	double scale;
	double midpoint;
	string expr;
	double xmin;
	double xmax;
	vector< double > tableEntry;
};
/**
 * 
 * 
 */
class Gate
{
	public:
		const std::string& getGateName();
		void setGateName(const std::string& value );
		double getInstances();
		void setInstances(double value );
		const std::string& getClosestateId();
		void setClosestateId(const std::string& value );
		const std::string& getOpenstateId();
		void setOpenstateId(const std::string& value );
		const std::string& getX_variable();
		void setX_variable(const std::string& value );
		const std::string& getY_variable();
		void setY_variable(const std::string& value );
		Transition alpha,beta;
	private:
		string gateName;
		double instances;
		string closeId;
		string openId;
		string x_variable;
		string y_variable;
		
		
};
/**
 * 
 * 
 */
class ChannelDefinition
{
	private:
		string channel_type;
		bool density;
		string claw;
		string ion;
		double erev;
		double gmax;	
		string conc_name;
		string conc_ion;
		string variable_name;
		double charge;
		double depen_charge;
		double min_conc;
		double max_conc;
		bool fixed_erev;	
		xmlDocPtr docptr;
		xmlXPathContextPtr cxtptr;
		xmlTextReaderPtr rdrptr;
		bool setchlnamespaces;
		bool setconc_dependence;	
		bool setCharge;
		double max_v;
		double min_v;
		double divs;
		
	public:
		ChannelDefinition():
			setchlnamespaces(false),
			setconc_dependence(false),
			setCharge(false)
		{;}
		void readDefinition( std::string& filename);
		const std::string& getChannel_type();		
		void setChannel_type(const std::string& value );
		bool getDensity();
		void setDensity( bool density );
		const std::string& getCond_law();
		void setCond_law(const std::string& value );
		const std::string& getIon();
		void setIon(const std::string& value );
		double getDefault_erev();
		void setDefault_erev(double value );
		double getDefault_gmax();
		void setDefault_gmax(double value );
		int register_channelNamespaces();
		bool issetChannelNamespaces() ;
		bool isSetCharge();
		const std::string& getConc_name();
		void setConc_name(const std::string& value );
		const std::string& getConc_ion();
		void setConc_ion(const std::string& value );
		const std::string& getVariable_name();
		void setVariable_name(const std::string& value );
		double getVolt_Charge();
		void setVolt_Charge(double value );
		double getDepen_Charge();
		void setDepen_Charge(double value );
		double getMin_conc();
		void setMin_conc(double value );
		double getMax_conc();
		void setMax_conc(double value );
		bool isSetConc_dependence();
		bool getFixed_erev();
		void setFixed_erev( bool value );
		double getMax_v();
		void setMax_v( double value );
		double getMin_v();
		void setMin_v( double value );
		double getDivs();
		void setDivs( double value );
		xmlXPathObjectPtr getnodeset (xmlChar *xpath) ;
		unsigned int getNumGates();
		Gate* getGate(int n);
		Gate gate;
};
/**
 * 
 * 
 */
class Channel
{
	public:
		Channel() {;}	
		~Channel() {;}	
		//void setType(string value);
		//const std::string& getType()const;
		//const std::string& getParameterName()const;
		//void setParameterName(string value);
		const double getGmax()const;
		void setGmax(double value);
		vector< string > getGroups() const;
		void setGroups( string group );
		const std::string& getName() const;
		bool isSetName() const;
		void setName(const std::string& name);
		void unsetName();
		bool getPassivecond()const;
		void setPassivecond(bool value);
		const std::string& getChannel_type();		
		void setChannel_type(const std::string& value );
		bool getDensity();
		void setDensity( bool density );
		const std::string& getCond_law();
		void setCond_law(const std::string& value );
		const std::string& getIon();
		void setIon(const std::string& value );
		double getDefault_erev();
		void setDefault_erev(double value );
		double getDefault_gmax();
		void setDefault_gmax(double value );
		const std::string& getGateName();
		void setGateName(const std::string& value );
		double getInstances();
		void setInstances(double value );
		const std::string& getClosestateId();
		void setClosestateId(const std::string& value );
		const std::string& getOpenstateId();
		void setOpenstateId(const std::string& value );
		const std::string& getX_variable();
		void setX_variable(const std::string& value );
		const std::string& getY_variable();
		void setY_variable(const std::string& value );
		unsigned int getNumGates() ;
		const std::string& getConc_name();
		void setConc_name(const std::string& value );
		const std::string& getConc_ion();
		void setConc_ion(const std::string& value );
		const std::string& getVariable_name();
		void setVariable_name(const std::string& value );
		double getVolt_Charge();
		void setVolt_Charge(double value );
		double getDepen_Charge();
		void setDepen_Charge(double value );
		double getMin_conc();
		void setMin_conc(double value );
		double getMax_conc();
		void setMax_conc(double value );
		bool isSetConc_dependence();
		bool isSetCharge();
		bool getFixed_erev();
		void setFixed_erev( bool value );
		double getMax_v();
		void setMax_v( double value );
		double getMin_v();
		void setMin_v( double value );
		double getDivs();
		void setDivs( double value );
		void unsetGroups();
		Gate* getGate(int n);
		vector< string > getSegGroups() const;
		void setSegGroups( string seg );
		void unsetSegGroups();

	private:
		string name;		
		double gmax;
		//string type;
		//string parmName;
		//string group;
		vector< string > groups;
		vector< string > segGroups;
		bool passivecond;
		ChannelDefinition* definition_;
		static map< std::string,ChannelDefinition > lookup_ ;
		static ChannelDefinition* lookupDefinition( const std::string& name );
};
#endif // _CHANNEL_H

