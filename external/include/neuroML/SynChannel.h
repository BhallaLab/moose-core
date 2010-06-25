#ifndef _SYNCHANNEL_H
#define _SYNCHANNEL_H
#include <map>	// Somehow this is needed in gcc 4.2.4, even if it is included in the .cpp file.
using namespace std;

class SynChannelDefinition
{
	private:
		string synapse_type;
		double max_conductance;
		double rise_time;
		double decay_time;
		double reversal_potential;
		double conc;
		double eta;
		double gamma;	
		xmlDocPtr syndocptr;
		xmlXPathContextPtr syncxtptr;
		xmlTextReaderPtr synrdrptr;
		bool setsynnamespaces;
		
	public:
		SynChannelDefinition():
			setsynnamespaces(false)
		{;}
		bool isblock;
		void readSynChlDefinition( std::string& filename);
		const std::string& getChannel_type();		
		void setChannel_type(const std::string& value );
		int register_synchannelNamespaces();
		bool issetsynChannelNamespaces();
		xmlXPathObjectPtr getnodeset (xmlChar *xpath);
		double getMax_Conductance();
		void setMax_Conductance( double value );
		double getRise_Time();
		void setRise_Time( double value );
		double getDecay_Time();
		void setDecay_Time( double value );
		double getReversal_Potential();
		void setReversal_Potential( double value );
		double getMgConc();
		void setMgConc( double value );
		double getEta();
		void setEta( double value );
		double getGamma();
		void setGamma( double value );
		

};
class SynChannel
{
	public:
		SynChannel() {;}	
		~SynChannel() {;}	
		void setSynType(string value);
		const std::string& getSynType()const;
		vector< string > getGroups() const;
		void setGroups( string group );
		void unsetGroups();
		const std::string& getChannel_type();		
		void setChannel_type(const std::string& value );
		double getMax_Conductance();
		void setMax_Conductance( double value );
		double getRise_Time();
		void setRise_Time( double value );
		double getDecay_Time();
		void setDecay_Time( double value );
		double getReversal_Potential();
		void setReversal_Potential( double value );
		double getMgConc();
		void setMgConc( double value );
		double getEta();
		void setEta( double value );
		double getGamma();
		void setGamma( double value );
		bool isMgblock();
	private:
		string type;	
		vector< string > groups;
		SynChannelDefinition* syndefinition_;
		static map< std::string,SynChannelDefinition > synlookup_ ;
		static SynChannelDefinition* synlookupDefinition( const std::string& name );
};

#endif // _SYNCHANNEL_H
