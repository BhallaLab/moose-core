#include "moose.h"
#include "utility/PathUtility.h"
#include "element/Neutral.h"
#include <iostream>
#include <set>
#include <neuroML/NCable.h>
#include <neuroML/NCell.h>
#include <neuroML/NBase.h>
#include <neuroML/SynChannel.h>
#include "Cable.h"
#include "NeuromlReader.h"
using namespace std;
const double NeuromlReader::PI = M_PI;
//void setupSegments();
//void setupCables();
/*  */
void NeuromlReader::readModel( string filename,Id location )
{
   #ifdef USE_NEUROML
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
   static const Cinfo* cableCinfo = initCableCinfo();
   PathUtility pathUtil( Property::getProperty( Property::SIMPATH ) );
   vector< string > paths;
   for ( unsigned int i = 0; i < pathUtil.size(); ++i )
   {
	paths.push_back( pathUtil.getPath( i ) );
	//cout << "path is : " << pathUtil.getPath( i ) << endl;
   }
   //cout << "path size : " << pathUtil.size() << endl;
   NBase::setPaths( paths );
   NBase nb;   
   ncl_= nb.readNeuroML (filename);
   ncl_->register_namespaces();
   string lengthunits = ncl_->getLengthUnits();
   double x,y,z,x1,y1,z1,diam1,diam2,diameter,length,initVm,r,ca,Cm,Ra;
   initVm = ncl_->getInit_memb_potential();
   ca = ncl_->getSpec_capacitance();
   r = ncl_->getSpec_axial_resistance();
   string biophysicsunit = ncl_->getBiophysicsUnits();
   if ( biophysicsunit == "Physiological Units" ){
   	ca /= 100; 
  	r *= 10; 
   	initVm /= 1000; 
   }
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
   map< string,string >NMcabMap;
   for( unsigned int cb = 1; cb <= num_cables; cb++ )
   {
	const NCable * cab;
	cab =ncl_->getCable(cb);
	std::string cid = "", cname = "", name = "";
	cid = cab->getId();
	cname = cab->getName(); 
	name = cname + "_" + cid;
	NMcabMap[cid] = name; //Neuroml Moose Map
	cable_ = Neutral::create( "Cable",name,loc,Id::scratchId() ); 
	cabMap[name] = cable_->id();
	::set< string >( cable_,nameFinfo,name );
	std::vector< std::string > groups;
	groups = cab->getGroups();
	::set< vector< string > >( cable_,groupsFinfo,groups );
	for (unsigned int itr = 0; itr < groups.size(); itr++ )
	     groupcableMap[groups[itr]].push_back(name);
   }
   map< string,vector< string > > cablesegMap;
   for(unsigned int i = 1; i<= num_segments; i ++ )
   {
		
	const Segment * seg ;
	seg =ncl_->getSegment(i);
	std::string id = "", sname = "", cable = "", sparent = "", name = "", parent =""; 
	id = seg->getId();
	sname = seg->getName();   
	cable = seg->getCable();
	sparent = seg->getParent();
	if (sparent != "")
	    parent = NMsegMap_[sparent];
	name = sname + "_" + id ;
	NMsegMap_[id] = name; //Neuroml Moose Map
	string mooseCable = NMcabMap[cable];
	compt_ = Neutral::create( "Compartment",name,location,Id::scratchId() );
	segMap_[name] = compt_->id(); 
	if (parent != "")
		Eref(compt_).add("raxial",segMap_[parent](),"axial",ConnTainer::Default ); 
	Eref(cabMap[mooseCable]()).add("compartment",Eref(compt_),"cable",ConnTainer::Default ); 
	cablesegMap[mooseCable].push_back(name);
	bool proxm = seg->isSetProximal();
        /* if proximal is not given then proximal is the  distal of parent compt */
	if ( proxm ){
		x = seg->proximal.getX();
		y = seg->proximal.getY();
		z = seg->proximal.getZ();
		x *= 1e-6;
		y *= 1e-6;
		z *= 1e-6;
		
	}
	else {
		get< double >( segMap_[parent].eref(), "x",x );
		get< double >( segMap_[parent].eref(), "y",y );
		get< double >( segMap_[parent].eref(), "z",z );
	}
	x1 = seg->distal.getX();
	y1 = seg->distal.getY();
	z1 = seg->distal.getZ();
	x1 *= 1e-6;
	y1 *= 1e-6;
	z1 *= 1e-6;
	
	length = sqrt(pow((x1-x),2) + pow((y1-y),2) + pow((z1-z),2));
	if ( proxm )
		diam1 = seg->proximal.getDiameter();
	else 
		diam1 = seg->distal.getDiameter();
	diam2 = seg->distal.getDiameter();
	diameter = (diam1 + diam2 )/2;
	diameter *= 1e-6; 
	::set< double >( compt_, x0Finfo, x );
	::set< double >( compt_, y0Finfo, y );
 	::set< double >( compt_, z0Finfo, z );
 	::set< double >( compt_, xFinfo, x1  );
	::set< double >( compt_, yFinfo, y1 );
	::set< double >( compt_, zFinfo, z1 );
	::set< double >( compt_, diameterFinfo, diameter );
	::set< double >( compt_, lengthFinfo, length );
	if ( length == 0 ){
		Cm = ca * PI * diameter * diameter;
		//Ra = 13.50 * r /( diameter * PI );
		Ra = r * 8.0 / ( diameter * PI );
	}
	else{
        	Cm = ca * length * PI * diameter;
        	//Ra = (r * length )/(PI * diameter * diameter/4);
		Ra = r * length * 4.0 / ( PI * diameter * diameter );
	}

	::set< double >( compt_, CmFinfo, Cm );
        ::set< double >( compt_, RaFinfo, Ra ); 
	::set< double >( compt_, initVmFinfo, initVm );
        
	
    }
   
   setupChannels(groupcableMap,cablesegMap,biophysicsunit);
   setupPools(groupcableMap,cablesegMap,biophysicsunit);
   unsigned int numsynchans =ncl_->getNumSynChannels();
   if ( numsynchans != 0 )
 	setupSynChannels(groupcableMap,cablesegMap,numsynchans);
 #else
	cout << "This version does not have NEUROML support." << endl; 
#endif 
}
 #ifdef USE_NEUROML
/* Returns the surface area of the compartment */
double NeuromlReader::calcSurfaceArea(double length,double diameter)
{
	double sa ;
	if( length == 0 )
		sa = PI * diameter * diameter;
	else
		sa = PI * diameter * length;
	return sa;
}
/* Returns the volume of the compartment */
double NeuromlReader::calcVolume(double length,double diameter)
{
	double v ;
	double r = diameter / 2 ;
	if( length == 0 )
		
		v = 4 / 3 * PI * r * r * r;
	else
		v = PI * r * r * length;
	return v;
}
void NeuromlReader::setupPools(map< string,vector<string> > &groupcableMap,map< string,vector< string > > &cablesegMap,string biophysicsunit)
{
	static const Cinfo* CaConcCinfo = initCaConcCinfo();
	static const Finfo* Ca_baseFinfo = CaConcCinfo->findFinfo( "Ca_base" );
	static const Finfo* tauFinfo = CaConcCinfo->findFinfo( "tau" );
	static const Finfo* BFinfo = CaConcCinfo->findFinfo( "B" );
	static const Finfo* thickFinfo = CaConcCinfo->findFinfo( "thick" );
	unsigned int num_pools = ncl_->getNumPools(); 
	if ( num_pools != 0 ) {  		
	map< string,vector<string> >::iterator cmap_iter;
    	map< string,vector<string> >::iterator smap_iter; 
	vector< string > pool_vec;
	pool_vec.clear();
	vector< string >::iterator it;
	for( unsigned int i = 0; i < num_pools; i++ )
    	{
    		IonPool* pool;
		pool = ncl_->getPool( i );
		Id loc( "/library" );
		string name = pool->getName();
		it = find(pool_vec.begin(), pool_vec.end(), name);
                if( it == pool_vec.end() ){
		    pool_vec.push_back(name);
		    ionPool_ = Neutral::create( "CaConc",name,loc,Id::scratchId() );
		}
	}

    	for( unsigned int i = 0; i < num_pools; i++ )
    	{
    		IonPool* pool;
		pool = ncl_->getPool( i );
		string name = "", group = "",scaling;
		double Ca_base, tau, B, thick;
		name = pool->getName();
		//cout<< "pool name " << name << endl;
		string path = "/library/"+name;
		Id poolId(path);
		Ca_base = pool->getResting_conc();
		tau = pool->getDecay_constant();
		thick = pool->getShell_thickness();
		scaling = pool->getScaling();	
		double B_ = pool->getScalingFactor();
		if ( biophysicsunit == "Physiological Units" ){
			Ca_base *= 1e6;
			thick *= 1e6;
			tau *= 1e-3;	
			B_ *= 1e-8;	
		}
		::set< double >( Eref(poolId()), Ca_baseFinfo, Ca_base );
		::set< double >( Eref(poolId()), tauFinfo, tau );
		::set< double >( Eref(poolId()), thickFinfo, thick );
		std::vector< std::string > groups;
		groups.clear();
		groups = pool->getGroups();
		std::vector< std::string > segs;
		segs.clear();
		segs = pool->getSegGroups();
		vector< Eref > channels;
		for(unsigned int i = 0; i < segs.size(); i++ )
		{
			string segName = NMsegMap_[segs[i]];
			Id comptEl = segMap_[segName];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double V = calcVolume( len,dia );
			//cout << "volume : " << V << "  of  " << comptEl()->name()<< endl;
			if( scaling == "off" )
				B = B_;
			else if( scaling == "shell" ){
				if ( thick <= 0 )
				    B = B_ / V ; //use compt vol or use absolute value
				else B = B_ / thick;
			}
			else if( scaling == "volume" )
				B = B_ / V;
			::set< double >( Eref(poolId()), BFinfo, B );
		 	Element* copyEl = Eref(poolId())->copy(comptEl(),ionPool_->name());
			//cout << "in pool compt " << comptEl()->name() << endl;
			channels.clear();
			targets( comptEl(),"channel",channels );
			unsigned int numchls = channels.size();
			for ( unsigned int i=0; i<numchls; i++ ){
				string name = channels[ i ].name();
				//if ( name == "CaConductance" )
				 if ( ionchlMap_[name] == "ca" )
				    channels[ i ].add("IkSrc",Eref(copyEl),"current",ConnTainer::Default ); 
				int use_conc;
				get< int >( channels[ i ], "useConcentration", use_conc );
				if( use_conc == 1 )
					Eref(copyEl).add("concSrc",channels[ i ],"concen",ConnTainer::Default ); 
			}
		  }
		vector< string > cId ;
		vector< string > sId;
		std::set< string > cableId;
		for(unsigned int gr = 0; gr < groups.size(); gr ++ )
		{
			//cout << "groups in vector from pool : " << groups[gr] << endl;				
			cmap_iter = groupcableMap.find( groups[gr] );
			if ( cmap_iter != groupcableMap.end() ){
			   cId = cmap_iter->second;
			   cableId.insert( cId.begin(),cId.end() );
			}
		}
		std::set< string >::iterator itr;
		for( itr = cableId.begin(); itr != cableId.end(); itr++ )
		{
		    smap_iter = cablesegMap.find( (*itr) );
		    if ( smap_iter != cablesegMap.end() ){
			sId = smap_iter->second;
		    }
		    for(unsigned int j = 0; j < sId.size(); j++ ){
			//cout << " seg Id:  " << sId[j] << endl; 
			Id comptEl = segMap_[sId[j]];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double V = calcVolume( len,dia );
			if( scaling == "fixed_current_scaling_factor" )
				B = B_;
			else if( scaling == "shell" ){
				if ( thick <= 0 )
				    B = B_ / V ; //use compt vol or use absolute value
				else B = B_ / thick;
			}
			else if( scaling == "specific_current_scaling_factor" )
				B = B_ / V;
			::set< double >( Eref(poolId()), BFinfo, B );
		 	Element* copyEl = Eref(poolId())->copy(comptEl(),ionPool_->name());
			channels.clear();
			targets( comptEl(),"channel",channels );
			unsigned int numchls = channels.size();
			for ( unsigned int i=0; i<numchls; i++ ){
				string name = channels[ i ].name();
				//if ( name == "CaConductance" )
				if ( ionchlMap_[name] == "ca" )
				    channels[ i ].add("IkSrc",Eref(copyEl),"current",ConnTainer::Default ); 
				int use_conc;
				get< int >( channels[ i ], "useConcentration", use_conc );
				if( use_conc == 1 )
					Eref(copyEl).add("concSrc",channels[ i ],"concen",ConnTainer::Default ); 
			}
		    }
           	}
	 
     	    }
	}
}

void NeuromlReader::setupChannels(map< string,vector<string> > &groupcableMap,map< string,vector< string > > &cablesegMap,string biophysicsunit)
{
	static const Cinfo* compartmentCinfo = initCompartmentCinfo();
	static const Finfo* EmFinfo = compartmentCinfo->findFinfo( "Em" );
        static const Finfo* RmFinfo = compartmentCinfo->findFinfo( "Rm" );	
	static const Cinfo* channelCinfo = initHHChannelCinfo();
	static const Finfo* gbarFinfo = channelCinfo->findFinfo( "Gbar" );
	static const Finfo* ekFinfo = channelCinfo->findFinfo( "Ek" );
	static const Finfo* xpowerFinfo = channelCinfo->findFinfo( "Xpower" );
	static const Finfo* ypowerFinfo = channelCinfo->findFinfo( "Ypower" );
	static const Finfo* zpowerFinfo = channelCinfo->findFinfo( "Zpower" );
	static const Finfo* useConcFinfo = channelCinfo->findFinfo( "useConcentration" );
	static const Cinfo* gateCinfo = initHHGateCinfo();
	static const Cinfo* leakCinfo = initLeakageCinfo();
	static const Finfo* leakekFinfo = leakCinfo->findFinfo( "Ek" );
	static const Finfo* setupAlphaFinfo = gateCinfo->findFinfo( "setupAlpha" );
	static const Cinfo* interpolCinfo = initInterpolCinfo();
	static const Finfo* tableVectorFinfo = interpolCinfo->findFinfo( "tableVector" );
	unsigned int num_channels = 0;
	num_channels = ncl_->getNumChannels();
	if ( num_channels != 0 ) {  		
	map< string,vector<string> >::iterator cmap_iter;
    	map< string,vector<string> >::iterator smap_iter; 
	vector< string > channel_vec;
	channel_vec.clear();
	vector< string >::iterator it;
	for(unsigned int ch = 0; ch < num_channels; ch++ )
	{
		Channel * chl ;	        
		chl = ncl_->getChannel( ch );
		double ek,gmax;
		Id loc( "/library" );
		string name = chl->getName();
		it = find(channel_vec.begin(), channel_vec.end(), name);
                if( it == channel_vec.end() ){
		    channel_vec.push_back(name);
		    //bool is2DChannel = chl->isSetConc_dependence();
		    string cion = chl->getConc_ion();
		    if ( cion != "" )
                        ionchlMap_[name] = cion;
		    string ion = chl->getIon();
		    if ( ion != "" )
                        ionchlMap_[name] = ion;
		    bool is2DChannel  = false;
		    bool passive = chl->getPassivecond();
		    ek = chl->getDefault_erev();
		    gmax = chl->getDefault_gmax();
		    if ( biophysicsunit == "Physiological Units" ){
		    	ek /= 1000; 
		    	gmax *= 10; 
		    }
		    if ( passive ){
			leak_ = Neutral::create( "Leakage",name,loc,Id::scratchId() ); 
			::set< double >( leak_, leakekFinfo, ek );
		    }
		    else if ( is2DChannel ){
			channel_ = Neutral::create( "HHChannel2D",name,loc,Id::scratchId() );	
			::set< double >( channel_, ekFinfo, ek );
			 ::set< double > ( channel_, gbarFinfo, gmax );
		   }
		   else{	
 	 		channel_ = Neutral::create( "HHChannel",name,loc,Id::scratchId() );
			::set< double >( channel_, ekFinfo, ek );
			 ::set< double > ( channel_, gbarFinfo, gmax );
		   }
	      }
        }
    	for(unsigned int ch = 0; ch < num_channels; ch++ )
    	{
    		Channel * chl ;
		chl =ncl_->getChannel( ch );
		string name = "", group = "";
		double gmax,ek;
		name = chl->getName();
		//cout << "channel is : "<< name << endl;
		string path = "/library/"+name;
		Id chlId(path);
		bool passive = chl->getPassivecond();
		ek = chl->getDefault_erev();
		bool use_conc = false;
		gmax = chl->getGmax();
                if ( biophysicsunit == "Physiological Units" ){
			gmax *= 10; 
	 	        ek /= 1000; 
	 	}
		if ( passive )
			::set< double >( Eref(chlId()), leakekFinfo, ek );
		else ::set< double >( Eref(chlId()), ekFinfo, ek );
		if ( !passive ){
			use_conc = chl->isSetConc_dependence();
			if( use_conc )
				::set< int >( Eref(chlId()), useConcFinfo, 1 );	
			else 
				::set< int >( Eref(chlId()), useConcFinfo, 0 );		
		}
	 	
	 	
	 	double xmin,xmax;
	 	int xdivs;
		double min,max;
	 	xmin = -0.1;
	 	xmax = 0.05;
		xdivs = 3000;
		vector< string > gatename;
		gatename.push_back("/xGate");
		gatename.push_back("/yGate");
		gatename.push_back("/zGate");
		string gatepath, tablepath;
		unsigned int  num = chl->getNumGates();
		if ( num > 3 ) 
		    	cout << "Error: More than 3 gates is not alllowed" << endl;
		int counter = 0;
	 	for( unsigned int i=1;i<=num;i++ )
	 	{	
			Gate* gat ;	
			bool flag = false;	
			gat = chl->getGate(i);
			string name = gat->getGateName();
			double power = gat->getInstances();
			string x_variable = gat->getX_variable();
			if( (x_variable == "concentration") || (use_conc == true && num == 1 ) ) {
			    if( flag == true )
				cout << "Error: already encountered Z gate" << endl;
			    else{
			        ::set< double >( Eref(chlId()), zpowerFinfo, power );
			        gatepath = path + gatename[2];
			        flag = true;
			    }
			}
			string y_variable = gat->getY_variable();
			if ( flag == false ){
			   counter++;
			   if ( counter == 1 )
			       ::set< double >( Eref(chlId()), xpowerFinfo, power );
			   else if ( counter == 2 )
				::set< double >( Eref(chlId()), ypowerFinfo, power );
			   else if ( counter == 3 )
				::set< double >( Eref(chlId()), zpowerFinfo, power );
			   gatepath = path + gatename[counter-1];
			}
			Id gateid(gatepath);
			vector< double > result;
			vector< double > alphaTable;
			vector< double > betaTable;
			vector< double > tableEntry;
			result.clear();
			alphaTable.clear();
			betaTable.clear();
			tableEntry.clear();
			string Aexpr_form = gat->alpha.expr_form; 
			if( Aexpr_form == "tabulated" ){
				tablepath = gatepath + "/A";
				Id tableid(tablepath);
				min = gat->alpha.xmin;
				max = gat->alpha.xmax;
				alphaTable = gat->alpha.tableEntry;
				::set< double > ( Eref(tableid()),"xmin",min );
				::set< double > ( Eref(tableid()),"xmax",max );
				::set< std::vector< double > >( Eref(tableid()),tableVectorFinfo,alphaTable );
			}
			else{
				double Arate = gat->alpha.rate;
				double Ascale = gat->alpha.scale;
				double Amidpoint = gat->alpha.midpoint;
				if ( biophysicsunit == "Physiological Units" ){
					Arate *= 1000; 
			 	        Ascale /= 1000; 
					Amidpoint /= 1000; 
	 			}
				pushtoVector(result,Aexpr_form,Arate,Ascale,Amidpoint);
				
			}
			string Bexpr_form = gat->beta.expr_form;
			if( Bexpr_form == "tabulated" ){
				tablepath = gatepath + "/B";
				Id tableid(tablepath);
				min = gat->beta.xmin;
				max = gat->beta.xmax;
				alphaTable = gat->alpha.tableEntry;
				tableEntry = gat->beta.tableEntry;
				int asize = alphaTable.size();
				int bsize = tableEntry.size();
				if ( asize == bsize ){
				   for(int i = 0; i < asize; i++ )
					betaTable.push_back(alphaTable[i]+tableEntry[i]);
				   ::set< double > ( Eref(tableid()),"xmin",min );
				   ::set< double > ( Eref(tableid()),"xmax",max );
				   ::set< std::vector< double > >( Eref(tableid()),tableVectorFinfo,betaTable );						
				}
				else cout << "Error: two table values should be of same size " << endl; 
			}
			else{
				double Brate = gat->beta.rate;
				double Bscale = gat->beta.scale;
				double Bmidpoint = gat->beta.midpoint;
				if ( biophysicsunit == "Physiological Units" ){
					Brate *= 1000; 
			 	        Bscale /= 1000; 
					Bmidpoint /= 1000; 
	 			}
				pushtoVector(result,Bexpr_form,Brate,Bscale,Bmidpoint);
				result.push_back(xdivs);
				result.push_back(xmin);
				result.push_back(xmax);
				::set< std::vector< double > >( Eref(gateid()),setupAlphaFinfo,result );
			}
			
	 	}
		
		std::vector< std::string > groups;
		groups = chl->getGroups();
		std::vector< std::string > segs;
		segs = chl->getSegGroups();
		for(unsigned int i = 0; i < segs.size(); i++ )
		{
			string segName = NMsegMap_[segs[i]];
			Id comptEl = segMap_[segName];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double sa = calcSurfaceArea( len,dia );
			double gbar = gmax * sa;
		 	
			if (passive){
				double Rm = 1/(gmax*sa);
		   		::set< double >( comptEl(), RmFinfo, Rm );
				::set< double >( comptEl(), EmFinfo, ek );
		 	}
			else{
				::set< double >( Eref(chlId()), gbarFinfo, gbar );  
				/*if (chlId()->name() == "K_CConductance"){
				   cout << comptEl()->name() << " " << gbar << " " << ek << " " << endl;
				   showmsg chlId()->path() ;
				}*/
				Element* copyEl = Eref(chlId())->copy(comptEl(),Eref(chlId())->name());
				Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default ); 
			}
		  }
		  vector< string > cId ;
		  vector< string > sId;
		  std::set< string > cableId;
		  for(unsigned int gr = 0; gr < groups.size(); gr ++ )
		  {
			//cout << "groups in vector : " << groups[gr] << endl;				
			cmap_iter = groupcableMap.find( groups[gr] );
			if ( cmap_iter != groupcableMap.end() ){
			   cId = cmap_iter->second;
			   cableId.insert( cId.begin(),cId.end() );
			}
		  }
		  std::set< string >::iterator itr;
		  for( itr = cableId.begin(); itr != cableId.end(); itr++ )
		  {
		    smap_iter = cablesegMap.find( (*itr) );
		    if ( smap_iter != cablesegMap.end() ){
			sId = smap_iter->second;
		    }
		    for( unsigned int j = 0; j < sId.size(); j++ ){
			//cout << " seg Id:  " << sId[j] << endl; 
			Id comptEl = segMap_[sId[j]];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double sa = calcSurfaceArea( len,dia );
			double gbar = gmax * sa;
		 	
			if (passive){
				double Rm = 1/(gmax*sa);
		   		::set< double >( comptEl(), RmFinfo, Rm );
				::set< double >( comptEl(), EmFinfo, ek );
		 	}
			else{
				::set< double >( Eref(chlId()), gbarFinfo, gbar );  
				
				Element* copyEl = Eref(chlId())->copy(comptEl(),Eref(chlId())->name());
				Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default ); 
			}
           	    }
	          }
	  }  
	}
}

void NeuromlReader::setupSynChannels(map< string,vector<string> > &groupcableMap,map< string,vector< string > > &cablesegMap,unsigned int numsynchans)
{
	//unsigned int numsynchans =ncl_->getNumSynChannels(); 
	static const Cinfo* synchanCinfo = initSynChanCinfo();
    	static const Finfo* synGbarFinfo = synchanCinfo->findFinfo( "Gbar" );
	static const Finfo* synEkFinfo = synchanCinfo->findFinfo( "Ek" );
	static const Finfo* synTau1Finfo = synchanCinfo->findFinfo( "tau1" );
	static const Finfo* synTau2Finfo = synchanCinfo->findFinfo( "tau2" );
	static const Cinfo* mgblockCinfo = initMg_blockCinfo();
	static const Finfo* CMgFinfo = mgblockCinfo->findFinfo( "CMg" );
	static const Finfo* KMg_AFinfo = mgblockCinfo->findFinfo( "KMg_A" );
	static const Finfo* KMg_BFinfo = mgblockCinfo->findFinfo( "KMg_B" );
	map< string,vector<string> >::iterator cmap_iter;
    	map< string,vector<string> >::iterator smap_iter; 
	if ( numsynchans != 0 ){
	for( unsigned int i = 1; i <= numsynchans; i ++ )
	{
		SynChannel* synchl;	
		Id loc( "/library" );
		synchl = ncl_->getSynChannel(i);
		string name = synchl->getSynType();
 	 	synchannel_ = Neutral::create( "SynChan",name,loc,Id::scratchId() );
		double gmax = synchl->getMax_Conductance();
		double tau1 = synchl->getRise_Time();
		::set< double >( synchannel_, synTau1Finfo, tau1 );
		double tau2 = synchl->getDecay_Time();
		::set< double >( synchannel_, synTau2Finfo, tau2 );
		double ek = synchl->getReversal_Potential();
		::set< double >( synchannel_, synEkFinfo, ek );
		bool isblock = synchl->isMgblock();
		if ( isblock == true ){
			string blockpath = "/library/"+name;
			Id blockid(blockpath);
			mgblock_ = Neutral::create( "Mg_block","block",blockid,Id::scratchId() );
			double mgconc = synchl->getMgConc();
			double eta = synchl->getEta();
			double gamma = synchl->getGamma();
			double KMg_A = 1.0/eta;
			double KMg_B = 1.0/gamma;
			::set< double >( mgblock_, CMgFinfo, mgconc );
			::set< double >( mgblock_, KMg_AFinfo, KMg_A );
			::set< double >( mgblock_, KMg_BFinfo, KMg_B );
			Eref(synchannel_).add("origChannel",mgblock_,"origChannel",ConnTainer::Default );
			
		}
		std::vector< std::string > groups;
		groups = synchl->getGroups();
		vector< string > cId ;
	 	vector< string > sId;
		std::set< string > cableId;
		for( unsigned int gr = 0; gr < groups.size(); gr ++ )
		{
			cmap_iter = groupcableMap.find( groups[gr] );
			if ( cmap_iter != groupcableMap.end() ){
			   cId = cmap_iter->second;
			   cableId.insert( cId.begin(),cId.end() );
			}
		}
		std::set< string >::iterator itr;
		for( itr = cableId.begin(); itr != cableId.end(); itr++ )
		{
		    smap_iter = cablesegMap.find( (*itr) );
		    if ( smap_iter != cablesegMap.end() ){
			sId = smap_iter->second;
		    }
	    	    for( unsigned int j = 0; j < sId.size(); j++ ){
			Id comptEl = segMap_[sId[j]];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double sa = calcSurfaceArea( len,dia );
			double gbar = gmax * sa;
			::set< double >( synchannel_,synGbarFinfo,gbar );
			Element* copyEl = synchannel_->copy(comptEl(),synchannel_->name());
			if ( isblock == true ){
				Id child = Neutral::getChildByName(Eref(copyEl),"block");
				Eref(comptEl()).add("channel",child(),"channel",ConnTainer::Default );
			}
			else {
				Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default );
			}
			
		   }
		}
	}
	}
}
/*function which insert elements into the vector for setupAlpha */
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
int NeuromlReader::targets( Eref object,const string& msg,vector< Eref >& target,const string& type )
{
	unsigned int oldSize = target.size();
	Eref found;
	Conn* i = object->targets( msg, 0 );
	for ( ; i->good(); i->increment() ) {
		found = i->target();
		if ( type != "" && !isType( found, type ) )	// speed this up
			continue;
		
		target.push_back( found );		
	}
	delete i;
	
	return target.size() - oldSize;
}

bool NeuromlReader::isType( Eref object, const string& type )
{
	return object->cinfo()->isA( Cinfo::find( type ) );
}
#endif // USE_NEUROML

  

  

