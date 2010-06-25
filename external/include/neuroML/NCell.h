#ifndef _NCELL_H
#define _NCELL_H
#include <string>
#include <libxml/xmlreader.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include "Segment.h"
#include "NCable.h"
#include "Channel.h"
#include "SynChannel.h"
#include "IonPool.h"
using namespace std;
class NCell
{
	public:
		NCell():
			setnamespaces(false) 
		{;}
		~NCell() {;}	
		//xmlDocPtr getDocument(string filename);
		bool isSetNamespaces() ;	
		int register_namespaces();
		xmlXPathObjectPtr getnodeset (xmlChar *xpath);
		const Segment* getSegment(string id);
		const Segment* getSegment(int n);
		const NCable* getCable(int n);
		Channel* getChannel(int n);
		SynChannel* getSynChannel(int n);
		IonPool* getPool(int n);
		//void getChannel(std::string filename,int n);
		unsigned int getNumSegments();
		unsigned int getNumCables();
		unsigned int getNumChannels();
		unsigned int getNumSynChannels(); 
		unsigned int getNumPools(); 
		double getInit_memb_potential();
		double getSpec_capacitance();
		double getSpec_axial_resistance();
		void getGate(int n);
		unsigned int getNumGates();
		void setXmldoc(xmlDocPtr& xmlDoc);
		xmlDocPtr& getXmldoc();
		void setContext(xmlXPathContextPtr& context);
		xmlXPathContextPtr& getContext();
		void setReaderptr(xmlTextReaderPtr& readerPtr);
		xmlTextReaderPtr& getReaderptr();
		const std::string& getLengthUnits();
		void setLengthUnits(const std::string& value );
		string getBiophysicsUnits();
		//bool hasSynapse();
		//void setBiophysicsUnits(const std::string& value );

	private:	
		Segment seg_;
		NCable cab_;
		Channel chl_;
		SynChannel synchl_;
		IonPool ion_;
		xmlDocPtr xmlDoc;
		xmlXPathContextPtr context;
		xmlTextReaderPtr readerPtr;
		bool setnamespaces;
		//bool synapticChl;
		//string biophysicsunits;
		string lengthunits;
		vector< IonPool > vec_pool;
		vector< Channel > vec_channel;
};
#endif // _NCELL_H

