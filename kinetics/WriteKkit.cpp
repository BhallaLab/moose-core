/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <time.h>
#include <iostream>
#include <fstream>
#include "header.h"
#include "PoolBase.h"
#include "../shell/Wildcard.h"

void writeHeader( ofstream& fout, 
		double simdt, double plotdt, double maxtime, double defaultVol)
{
	time_t rawtime;
	time( &rawtime );

	fout << 
	"//genesis\n"
	"// kkit Version 11 flat dumpfile\n\n";
	fout << "// Saved on " << ctime( &rawtime ) << endl;
	fout << "include kkit {argv 1}\n";
	fout << "FASTDT = " << simdt << endl;
	fout << "SIMDT = " << simdt << endl;
	fout << "CONTROLDT = " << plotdt << endl;
	fout << "PLOTDT = " << plotdt << endl;
	fout << "MAXTIME = " << maxtime << endl;
	fout << "TRANSIENT_TIME = 2\n"
	"VARIABLE_DT_FLAG = 0\n";
	fout << "DEFAULT_VOL = " << defaultVol << endl;
	fout << "VERSION = 11.0\n"
	"setfield /file/modpath value ~/scripts/modules\n"
	"kparms\n\n";

	fout << 
	"initdump -version 3 -ignoreorphans 1\n"
	"simobjdump table input output alloced step_mode stepsize x y z\n"
	"simobjdump xtree path script namemode sizescale\n"
	"simobjdump xcoredraw xmin xmax ymin ymax\n"
	"simobjdump xtext editable\n"
	"simobjdump xgraph xmin xmax ymin ymax overlay\n"
	"simobjdump xplot pixflags script fg ysquish do_slope wy\n"
	"simobjdump group xtree_fg_req xtree_textfg_req plotfield expanded movealone \\\n"
  	"  link savename file version md5sum mod_save_flag x y z\n"
	"simobjdump geometry size dim shape outside xtree_fg_req xtree_textfg_req x y z\n"
	"simobjdump kpool DiffConst CoInit Co n nInit mwt nMin vol slave_enable \\\n"
  	"  geomname xtree_fg_req xtree_textfg_req x y z\n"
	"simobjdump kreac kf kb notes xtree_fg_req xtree_textfg_req x y z\n"
	"simobjdump kenz CoComplexInit CoComplex nComplexInit nComplex vol k1 k2 k3 \\\n"
  	"  keepconc usecomplex notes xtree_fg_req xtree_textfg_req link x y z\n"
	"simobjdump stim level1 width1 delay1 level2 width2 delay2 baselevel trig_time \\\n"
  	"  trig_mode notes xtree_fg_req xtree_textfg_req is_running x y z\n"
	"simobjdump xtab input output alloced step_mode stepsize notes editfunc \\\n"
  	"  xtree_fg_req xtree_textfg_req baselevel last_x last_y is_running x y z\n"
	"simobjdump kchan perm gmax Vm is_active use_nernst notes xtree_fg_req \\\n"
  	"  xtree_textfg_req x y z\n"
	"simobjdump transport input output alloced step_mode stepsize dt delay clock \\\n"
  	"  kf xtree_fg_req xtree_textfg_req x y z\n"
	"simobjdump proto x y z\n"
	"simundump geometry /kinetics/geometry 0 1.6667e-19 3 sphere "" white black 0 0 0\n\n";
}


void writeReac( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/kinetics" );
	path = path.substr( pos );
	double kf = Field< double >::get( id, "kf" );
	double kb = Field< double >::get( id, "kb" );

	fout << "simundump kreac " << path << " 0 " << 
			kf << " " << kb << " \"\" " << 
			colour << " " << textcolour << " " << x << " " << y << " 0\n";
}

void writePool( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/kinetics" );
	path = path.substr( pos );
	double diffConst = Field< double >::get( id, "diffConst" );
	double concInit = Field< double >::get( id, "concInit" );
	double conc = Field< double >::get( id, "conc" );
	double nInit = Field< double >::get( id, "nInit" );
	double n = Field< double >::get( id, "n" );
	double size = Field< double >::get( id, "size" );

	fout << "simundump kpool " << path << " 0 " << 
			diffConst << " " <<
			concInit << " " << 
			conc << " " <<
			n << " " <<
			nInit << " " <<
			0 << " " << 0 << " " << // mwt, nMin
			size * NA * 1e-3  << " " << // volscale
			"0 /kinetics/geometry " << 
			colour << " " << textcolour << " " << x << " " << y << " 0\n";
}

Id getEnzMol( Id id )
{
	return Id();
}

Id getEnzCplx( Id id )
{
	return Id();
}

void writeEnz( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/kinetics" );
	path = path.substr( pos );
	double k1 = 0;
	double k2 = 0;
	double k3 = 0;
	double nInit = 0;
	double concInit = 0;
	double n = 0;
	double conc = 0;
	Id enzMol = getEnzMol( id );
	assert( enzMol != Id() );
	double vol = Field< double >::get( enzMol, "size" ) * NA * 1e-3; 
	unsigned int isMassAction = 0;
	if ( id.element()->cinfo()->isA( "CplxEnzBase" ) ) {
		k1 = Field< double >::get( id, "k1" );
		k2 = Field< double >::get( id, "k2" );
		k3 = Field< double >::get( id, "k3" );
		Id cplx = getEnzCplx( id );
		assert( cplx != Id() );
		nInit = Field< double >::get( cplx, "nInit" );
		n = Field< double >::get( cplx, "n" );
		concInit = Field< double >::get( cplx, "concInit" );
		conc = Field< double >::get( cplx, "conc" );
		isMassAction = 1;
	} else {
		k1 = Field< double >::get( id, "numKm" );
		k3 = Field< double >::get( id, "kcat" );
		k2 = 4.0 * k3;
		k1 = (k2 + k3) / k1;
	}

	fout << "simundump kenz " << path << " 0 " << 
			concInit << " " <<
			conc << " " << 
			nInit << " " <<
			n << " " <<
			vol << " " <<
			k1 << " " <<
			k2 << " " <<
			k3 << " " <<
			0 << " " <<
			isMassAction << " " <<
			"\"\"" << " " << 
			colour << " " << textcolour << "\"\"" << 
			" " << x << " " << y << " 0\n";
}

void writeGroup()
{
}

void writePlot()
{
		/*
simundump xplot /graphs/conc1/Sub.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/Prd.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 60 0 0 1
  */
}

void writeGui( ofstream& fout )
{
	fout << "simundump xgraph /graphs/conc1 0 0 99 0.001 0.999 0\n"
	"simundump xgraph /graphs/conc2 0 0 100 0 1 0\n"
	"simundump xgraph /moregraphs/conc3 0 0 100 0 1 0\n"
	"simundump xgraph /moregraphs/conc4 0 0 100 0 1 0\n"
	"simundump xcoredraw /edit/draw 0 -6 4 -2 6\n"
	"simundump xtree /edit/draw/tree 0 /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z> auto 0.6\n"
	"simundump xtext /file/notes 0 1\n";
}

void writeMsgs()
{
		/*
addmsg /kinetics/Sub /kinetics/kreac SUBSTRATE n 
addmsg /kinetics/Prd /kinetics/kreac PRODUCT n 
addmsg /kinetics/kreac /kinetics/Sub REAC A B 
addmsg /kinetics/kreac /kinetics/Prd REAC B A 
addmsg /kinetics/Sub /graphs/conc1/Sub.Co PLOT Co *Sub.Co *blue 
addmsg /kinetics/Prd /graphs/conc1/Prd.Co PLOT Co *Prd.Co *60 
*/
}

void writeFooter( ofstream& fout )
{
	fout << "\nenddump\n";
	fout << "complete_loading\n";
}

Id findInfo( Id id )
{
		return Id();
}

void writeKkit( Id model, const string& fname )
{
		vector< Id > ids;
		unsigned int num = simpleWildcardFind( model.path() + "/##", ids );
		if ( num == 0 ) {
			cout << "Warning: writeKkit:: No model found on " << model << 
					endl;
			return;
		}
		ofstream fout( fname.c_str(), ios::out );
		writeHeader( fout, 1, 1, 1, 1 );
		writeGui( fout );

		string bg = "cyan";
		string fg = "black";
		double x = 0;
		double y = 0;
		double side = floor( 1.0 + sqrt( static_cast< double >( num ) ) );
		double dx = side / num;
		for( vector< Id >::iterator i = ids.begin(); i != ids.end(); ++i ) {
			Id info = findInfo( *i );
			if ( info != Id() ) {
				bg = Field< string >::get( info, "color" );
				fg = Field< string >::get( info, "textColor" );
				x = Field< double >::get( info, "x" );
				y = Field< double >::get( info, "y" );
			} else {
				x += dx;
				if ( x > side ) {
						x = 0;
						y += dx;
				}
			}
			if ( i->element()->cinfo()->isA( "PoolBase" ) ) {
				// Check that we don't have an enz cplx.
				writePool( fout, *i, fg, bg, x, y );
			} else if ( i->element()->cinfo()->isA( "ReacBase" ) ) {
				writeReac( fout, *i, fg, bg, x, y );
			} else if ( i->element()->cinfo()->isA( "EnzBase" ) ) {
				writeEnz( fout, *i, fg, bg, x, y );
			} else if ( i->element()->cinfo()->name() == "Neutral" ) {
			}
		}
		writePlot();
		writeMsgs();
		writeFooter( fout );
}
