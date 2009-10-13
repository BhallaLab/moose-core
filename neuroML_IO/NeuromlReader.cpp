#include "moose.h"
#include "element/Neutral.h"
#include <iostream>
#include "neuroML_IO/Cable.h"
#include <neuroML/NCell.h>
#include <neuroML/NBase.h>
#include "Cable.h"
#include "NeuromlReader.h"
using namespace std;
const double NeuromlReader::PI = 3.14159;
void setupSegments();
void setupCables();
void NeuromlReader::readModel( string filename,Id location )
{
   static const Cinfo* compartmentCinfo = initCompartmentCinfo();
   static const Finfo* lengthFinfo = compartmentCinfo->findFinfo( "length" );
   static const Finfo* diameterFinfo = compartmentCinfo->findFinfo( "diameter" );
   static const Finfo* x0Finfo = compartmentCinfo->findFinfo( "x0" );
   static const Finfo* y0Finfo = compartmentCinfo->findFinfo( "y0" );
   static const Finfo* z0Finfo = compartmentCinfo->findFinfo( "z0" );
   static const Finfo* xFinfo = compartmentCinfo->findFinfo( "x" );
   static const Finfo* yFinfo = compartmentCinfo->findFinfo( "y" );
   static const Finfo* zFinfo = compartmentCinfo->findFinfo( "z" );
   static const Finfo* RaFinfo = compartmentCinfo->findFinfo( "Ra" );
   static const Finfo* initVmFinfo = compartmentCinfo->findFinfo( "initVm" );
   static const Finfo* CmFinfo = compartmentCinfo->findFinfo( "Cm" );
   static const Finfo* EmFinfo = compartmentCinfo->findFinfo( "Em" );
   static const Cinfo* cableCinfo = initCableCinfo();
   NBase nb;   
   //NCell* c;
   ncl_= nb.readNeuroML (filename);
   ncl_->register_namespaces();
   string lengthunits = ncl_->getLengthUnits();
   double x,y,z,x1,y1,z1,diam1,diam2,diameter,length,initVm,r,ca,Cm,Ra;
   initVm = ncl_->getInit_memb_potential();
   ca = ncl_->getSpec_capacitance();
   r = ncl_->getSpec_axial_resistance();
  // ca /= 100; //si unit
  // r *= 10; //si
   static const Finfo* nameFinfo = cableCinfo->findFinfo( "name" );
   static const Finfo* groupsFinfo = cableCinfo->findFinfo( "groups" );
   unsigned int num_cables = ncl_->getNumCables();
   unsigned int num_segments = ncl_->getNumSegments();
   Id loc("/cables");  
   if ( loc.good() ) 
      cerr << "Warning: cables already exists. You are trying to overwrite it" << endl;	
   Element* locE = Neutral::create( "Neutral", "cables", Id(), Id::scratchId() );
   loc = locE->id();
   map< string,vector<string> > groupcableMap; 
   map< string,Id > cabMap;	
   for( int cb = 1; cb <= num_cables; cb++ )
   {
	const NCable * cab;
	cab =ncl_->getCable(cb);
	std::string cid = "", cname = "";
	cid = cab->getId();
	cname = cab->getName(); 
	//cout << "id " << cid << " name " << cname << endl;
	cable_ = Neutral::create( "Cable",cid,loc,Id::scratchId() ); 
	cabMap[cid] = cable_->id();
	set< string >( cable_,nameFinfo,cid );
	std::vector< std::string > groups;
	groups = cab->getGroups();
	set< vector< string > >( cable_,groupsFinfo,groups );
	for ( int itr = 0; itr < groups.size(); itr++ )
	     groupcableMap[groups[itr]].push_back(cid);
   }
   map< string,vector< string > > cablesegMap;
   //map< string,Id > segMap;	
   for( int i = 1; i<= num_segments; i ++ )
   {
		
	const Segment * seg ;
	seg =ncl_->getSegment(i);
	std::string id = "", name = "", cable = "", parent = ""; 
	id = seg->getId();
	name = seg->getName();   
	cable = seg->getCable();
	parent = seg->getParent();
	//cout << "id " << id << " name " << name << " cable " << cable << endl; 
	compt_ = Neutral::create( "Compartment",id,location,Id::scratchId() );
	segMap_[id] = compt_->id(); 
	if (parent != "")
		Eref(compt_).add("raxial",segMap_[parent](),"axial",ConnTainer::Default ); 
	Eref(cabMap[cable]()).add("compartment",Eref(compt_),"cable",ConnTainer::Default ); 
	cablesegMap[cable].push_back(id);
	x = seg->proximal.getX();
	y = seg->proximal.getY();
	z = seg->proximal.getZ();
	x1 = seg->distal.getX();
	y1 = seg->distal.getY();
	z1 = seg->distal.getZ();
	diam1 = seg->proximal.getDiameter();
	diam2 = seg->distal.getDiameter();
    	set< double >( compt_, x0Finfo, x );
	set< double >( compt_, y0Finfo, y );
 	set< double >( compt_, z0Finfo, z );
 	set< double >( compt_, xFinfo, x1  );
	set< double >( compt_, yFinfo, y1 );
	set< double >( compt_, zFinfo, z1 );
	diameter = (diam1 + diam2 )/2;
	length = sqrt(pow((x1-x),2) + pow((y1-y),2) + pow((z1-z),2));
	diameter *= 1e-4; //physiological unit
	length *= 1e-4;
	//length *= 1e-6;//si
	//diameter *= 1e-6;//si
	set< double >( compt_, diameterFinfo, diameter );
	set< double >( compt_, lengthFinfo, length );
	if ( length == 0 ){
		Cm = ca * PI * diameter * diameter;
		Ra = 13.50 * r /( diameter * PI );
		//sa = 4 * PI * diameter * diameter/4;
	}
	else{
        	Cm = ca * length * PI * diameter;
        	Ra = (r * length )/(PI * diameter * diameter/4);
		//sa = (2 * PI * (diameter * diameter)/4)+(2 * PI * diameter/2 * length);
	}
	//cout << "suraface area " << sa << endl;
        set< double >( compt_, CmFinfo, Cm );
        set< double >( compt_, RaFinfo, Ra ); 
	set< double >( compt_, initVmFinfo, initVm );
        //set< double >( compt_, initVmFinfo, initVm/1000);//si unit 
	set< double >( compt_, EmFinfo, initVm );
    }
   setupChannels(groupcableMap,cablesegMap);
   setupSynChannels(groupcableMap,cablesegMap);
    
   
}
double NeuromlReader::calcSurfaceArea(double length,double diameter)
{
	double sa ;
	if( length == 0 )
		sa = 4 * PI * diameter * diameter/4;
	else
		sa = (2 * PI * (diameter * diameter)/4)+(2 * PI * diameter/2 * length);
	return sa;	
}
void NeuromlReader::setupChannels(map< string,vector<string> > &groupcableMap,map< string,vector< string > > &cablesegMap)
{
	//NCell* c;
	static const Cinfo* compartmentCinfo = initCompartmentCinfo();
	static const Finfo* EmFinfo = compartmentCinfo->findFinfo( "Em" );
        static const Finfo* RmFinfo = compartmentCinfo->findFinfo( "Rm" );	
	static const Cinfo* channelCinfo = initHHChannelCinfo();
	static const Finfo* gbarFinfo = channelCinfo->findFinfo( "Gbar" );
	static const Finfo* ekFinfo = channelCinfo->findFinfo( "Ek" );
	static const Finfo* xpowerFinfo = channelCinfo->findFinfo( "Xpower" );
	static const Finfo* ypowerFinfo = channelCinfo->findFinfo( "Ypower" );
	static const Finfo* zpowerFinfo = channelCinfo->findFinfo( "Zpower" );
	unsigned int num_channels = ncl_->getNumChannels();
	map< string,vector<string> >::iterator cmap_iter;
    	map< string,vector<string> >::iterator smap_iter; 
    	for( int ch = 1; ch <= num_channels; ch++ )
    	{
    		Channel * chl ;
		chl =ncl_->getChannel( ch );
		string name = "", group = "";
		double gmax,ek;
		Id loc( "/library" );
		name = chl->getName();
		bool is2DChannel = chl->isSetConc_dependence();
		if ( is2DChannel )
			channel_ = Neutral::create( "HHChannel2D",name,loc,Id::scratchId() );	
		else	
 	 		channel_ = Neutral::create( "HHChannel",name,loc,Id::scratchId() );
	 	gmax = chl->getParameterValue();
	 	//gmax *= 10;//si
	 	ek = chl->getDefault_erev();
		set< double >( channel_, ekFinfo, ek );//si
	 	double xmin,xmax;
	 	int xdivs;
	 	string biophysicsunits = ncl_->getBiophysicsUnits();
		//cout << " unit " << biophysicsunits << endl;
	 	if ( biophysicsunits == "Physiological Units" ){
	 		xmin = -100;
	 		xmax = 50;
		}
	 	else if ( biophysicsunits == "SI Units" ){
	 		xmin = -0.1;
	 		xmax = 0.05;
		}
		xdivs = 3000;
		string path = Eref(channel_).id().path();
		vector< string > gatename;
		gatename.push_back("/xGate");
		gatename.push_back("/yGate");
		gatename.push_back("/zGate");
		unsigned int  num = chl->getNumGates();
	 	for( int i=1;i<=num;i++ )
	 	{	
			Gate* gat ;		
			gat = chl->getGate(i);
			string name = gat->getGateName();
			double power = gat->getInstances();
			if ( i == 1 )
				set< double >( channel_, xpowerFinfo, power );
			else if ( i == 2 )
				set< double >( channel_, ypowerFinfo, power );
			else if ( i == 3 )
				set< double >( channel_, zpowerFinfo, power );
			string gatepath = path + gatename[i-1];
			//cout << " gate path : " << gatepath << endl;
		 	Id gateid(gatepath);
			vector< double > result;
			result.clear();
			string Aexpr_form = gat->alpha.expr_form;
			double Arate = gat->alpha.rate;
			//Arate *= 1000;
			double Ascale = gat->alpha.scale;
			//Ascale /= 1000;
			double Amidpoint = gat->alpha.midpoint;
			//Amidpoint /= 1000;
			pushtoVector(result,Aexpr_form,Arate,Ascale,Amidpoint);
			string Bexpr_form = gat->beta.expr_form;
			double Brate = gat->beta.rate;
			//Brate *= 1000;
			double Bscale = gat->beta.scale;
			//Bscale /= 1000;
			double Bmidpoint = gat->beta.midpoint;
			//Bmidpoint /= 1000;
			pushtoVector(result,Bexpr_form,Brate,Bscale,Bmidpoint);
			result.push_back(xdivs);
			result.push_back(xmin);
			result.push_back(xmax);
			//cout << "result size : "<< result.size() << endl;
			set< vector< double > >( Eref(gateid()),"setupAlpha",result );
	 	}
		bool passive = chl->getPassivecond();
		group = chl->getGroup();
		// cout<< " group: " << group ;
		vector< string > cId ;
		vector< string > sId;
		cmap_iter = groupcableMap.find( group );
		if ( cmap_iter != groupcableMap.end() )
		   cId = cmap_iter->second;
		else {
	   		cout << "Error: CableId is empty" << endl;
	   		return;
	 	}
		for( int i = 0; i < cId.size(); i++ )
		{
		   //cout << " cable id:  " << cId[i];
		    smap_iter = cablesegMap.find( cId[i] );
		    if ( smap_iter != cablesegMap.end() )
			sId = smap_iter->second;
		    else{
			cout << "Error:SegmentId is empty" << endl;
			return;
		    }
		    for( int j = 0; j < sId.size(); j++ ){
			//cout << " seg Id:  " << sId[j] << endl; 
			Id comptEl = segMap_[sId[j]];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double sa = calcSurfaceArea( len,dia );
		 	double gbar = gmax * sa;
		 	set< double >( channel_, gbarFinfo, gbar );
			if (passive){
			
		   		double Rm = 1/(gmax*sa);
		   		set< double >( comptEl(), RmFinfo, Rm );
				set< double >( comptEl(), EmFinfo, ek );
		 	}
			else{
				Element* copyEl = channel_->copy(comptEl(),channel_->name());
				Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default ); 
			}
           	}
	 
     	}
}  
}

void NeuromlReader::setupSynChannels(map< string,vector<string> > &groupcableMap,map< string,vector< string > > &cablesegMap)
{
	//NCell* c;	
	unsigned int numsynchans =ncl_->getNumSynChannels(); 
	static const Cinfo* synchanCinfo = initSynChanCinfo();
    	static const Finfo* synGbarFinfo = synchanCinfo->findFinfo( "Gbar" );
	static const Finfo* synEkFinfo = synchanCinfo->findFinfo( "Ek" );
	static const Finfo* synTau1Finfo = synchanCinfo->findFinfo( "tau1" );
	static const Finfo* synTau2Finfo = synchanCinfo->findFinfo( "tau2" );
	map< string,vector<string> >::iterator cmap_iter;
    	map< string,vector<string> >::iterator smap_iter; 
	for( int i = 1; i <= numsynchans; i ++ )
	{
		SynChannel* synchl;	
		Id loc( "/library" );
		synchl = ncl_->getSynChannel(i);
		string name = synchl->getSynType();
 	 	synchannel_ = Neutral::create( "SynChan",name,loc,Id::scratchId() );
		double gmax = synchl->getMax_Conductance();
		double tau1 = synchl->getRise_Time();
		set< double >( synchannel_, synTau1Finfo, tau1 );
		double tau2 = synchl->getDecay_Time();
		set< double >( synchannel_, synTau2Finfo, tau2 );
		double ek = synchl->getReversal_Potential();
		set< double >( synchannel_, synEkFinfo, ek );
		string group = synchl->getGroup();
		vector< string > cId ;
	 	vector< string > sId;
	 	cmap_iter = groupcableMap.find( group );
	 	if ( cmap_iter != groupcableMap.end() )
	    		cId = cmap_iter->second;
	 	else {
	   		cout << "Error: CableId is empty" << endl;
	   		return;
		 }
	 	for( int i = 0; i < cId.size(); i++ )
	 	{
	  	     smap_iter = cablesegMap.find( cId[i] );
	    	     if ( smap_iter != cablesegMap.end() )
			sId = smap_iter->second;
	       	     else{
			cout << "Error:SegmentId is empty" << endl;
			return;
	    	     }
	    	     for( int j = 0; j < sId.size(); j++ ){
			Id comptEl = segMap_[sId[j]];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double sa = calcSurfaceArea( len,dia );
			cout << " len "<< len << " dia " << dia << " sa " << sa << endl; 
			double gbar = gmax * sa;
			set< double >( synchannel_,synGbarFinfo,gbar );
			Element* copyEl = synchannel_->copy(comptEl(),synchannel_->name());
			Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default ); 
		     }
		}
	}
	
}

void NeuromlReader::pushtoVector(vector< double > &result,string expr_form,double r,double s,double m)
{
	double A,B,C,D,F;
	if( expr_form == "exp_linear" ){
		A = r*m/s;
		B = -r/s;
		C = -1;
		D = -m;
		F = -s;	
	}
	if( expr_form == "sigmoid" ){
		A = r;
		B = 0;
		C = 1;
		D = -m;
		F = s;	
	}
	if( expr_form == "exponential" ){
		A = r;
		B = 0;
		C = 0;
		D = -m;
		F = -s;
	}
	result.push_back(A);
	result.push_back(B);
	result.push_back(C);
	result.push_back(D);
	result.push_back(F);	
}



  

  

