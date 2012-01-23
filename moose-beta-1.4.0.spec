Summary:Multiscale Object-Oriented Simulation Environment
Name: moose
Version: 1.4
Release: 0
License: LGPL
Group: Applications/Science
URL: http://moose.ncbs.res.in/
Source0: %{name}-%{version}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root
Requires: libstdc++ >= 4.1.2,readline >= 6.1,gnuplot >= 4.4.0,PyQt4 >= 4.8.3,PyQt4-devel >= 4.8.3,numpy >= 1.4.1,PyQwt >= 5.2.0,python-matplotlib >= 1.0.1,PyOpenGL >= 3.0.1,libxml2 >= 2.7.7
BuildRequires: gcc >= 4.5.1,gcc-c++ >= 4.5.1,libstdc++ >= 4.1.2,readline-devel >= 6.1,gsl >= 1.14,gsl-devel >= 1.14,python-devel >= 2.7,swig >= 2.0.1,bison >= 2.4.3,flex >= 2.5.35,sed >= 4.2.1
%description
MOOSE is the Multiscale Object-Oriented Simulation Environment. 
 It is the base and numerical core for large, detailed simulations including 
 Computational Neuroscience and Systems Biology. 
 . 
 MOOSE spans the range from single molecules to subcellular networks, from 
 single cells to neuronal networks, and to still larger systems. It is 
 backwards-compatible with GENESIS, and forward compatible with Python 
 and XML-based model definition standards like SBML and MorphML. It supports
 transparent parallelization, that is, it can run models on a cluster with no
 user effort required to parallelize them.
%prep
%setup -q

%build
make clean
make pymoose
strip python/moose/_moose.so
mv python/moose/_moose.so _moose.so.tmp
make clean
make moose
strip moose
g++ TESTS/regression/neardiff.cpp -o TESTS/regression/neardiff
%install
rm -rf $RPM_BUILD_ROOT
mkdir -p "$RPM_BUILD_ROOT/usr/bin"
mkdir -p "$RPM_BUILD_ROOT/usr/share/doc/"
mkdir -p "$RPM_BUILD_ROOT/usr/share/man/man1"
mkdir -p "$RPM_BUILD_ROOT/usr/share/info"
mkdir -p "$RPM_BUILD_ROOT/usr/share/moose1.4/lib"
mkdir -p "$RPM_BUILD_ROOT/usr/share/moose1.4/py_stage"
cp README.txt "$RPM_BUILD_ROOT/usr/share/moose1.4"
cp find_python_path.py "$RPM_BUILD_ROOT/usr/share/moose1.4"
cp moose-script "$RPM_BUILD_ROOT/usr/bin"
mv "$RPM_BUILD_ROOT/usr/bin/moose-script" "$RPM_BUILD_ROOT/usr/bin/moose"
svn export TESTS/ "$RPM_BUILD_ROOT/usr/share/moose1.4/TESTS"
svn export DOCS/ "$RPM_BUILD_ROOT/usr/share/doc/moose1.4/"
cp INSTALL COPYING.LIB "$RPM_BUILD_ROOT/usr/share/doc/moose1.4/"
svn export gui/ "$RPM_BUILD_ROOT/usr/share/moose1.4/gui"
svn export DEMOS/ "$RPM_BUILD_ROOT/usr/share/moose1.4/DEMOS"
svn export python/moose "$RPM_BUILD_ROOT/usr/share/moose1.4/py_stage/moose"
install -s -m 755 moose "$RPM_BUILD_ROOT/usr/bin/moose-bin"
install -s -m 644 _moose.so.tmp "$RPM_BUILD_ROOT/usr/share/moose1.4/py_stage/moose/_moose.so"
install -m 644 "DOCS/moose.1" "$RPM_BUILD_ROOT/usr/share/man/man1/"
install -m 644 DOCS/pymoose/pymoose.info "$RPM_BUILD_ROOT/usr/share/info/pymoose.info"
install -s -m 755 TESTS/regression/neardiff  "$RPM_BUILD_ROOT/usr/share/moose1.4/TESTS/regression/"
install -m 755 TESTS/regression/do_regression.bat "$RPM_BUILD_ROOT/usr/share/moose1.4/TESTS/regression/"
chmod 755 "$RPM_BUILD_ROOT/usr/share/moose1.4/gui/moosestart.py"
chmod 755 "$RPM_BUILD_ROOT/usr/share/moose1.4/gui/MooseGUI.desktop"

# Copy the libraries to lib directory in moose-shared deirectory
#( ldd _moose.so )    | while read line; do 
( ldd _moose.so.tmp)    | while read line; do 
    a=`echo $line |sed -e 's/(.*$//'`; 
    b=`echo $a | cut -s -d '>' -f2`
    if [ ! -z `echo $b | tr -d [:space:]` ]; then
	for libname in libsbml ; do
	    echo $b $libname 
	    if [[ $b =~ $libname ]] ; then
		echo "Copying $b to lib/"  
		install -m 644 $b "$RPM_BUILD_ROOT/usr/share/moose1.4/lib"
		break
	     fi
	done
    fi
done

%post
PPATH=`python /usr/share/moose1.4/find_python_path.py`
mv /usr/share/moose1.4/py_stage/* $PPATH
ln -s "$RPM_BUILD_ROOT/usr/share/moose1.4/gui/moosestart.py" "$RPM_BUILD_ROOT/usr/bin/moosegui"
echo "$RPM_BUILD_ROOT/usr/share/moose1.4/lib/" > /etc/ld.so.conf.d/moose.conf
cp /usr/share/moose1.4/gui/MooseGUI.desktop /home/$SUDO_USER/Desktop
chmod 755  /home/$SUDO_USER/Desktop/MooseGUI.desktop
chown $SUDO_USER /home/$SUDO_USER/Desktop/MooseGUI.desktop
install-info --infodir="$RPM_BUILD_ROOT/usr/share/info" "$RPM_BUILD_ROOT/usr/share/info/pymoose.info"


%clean
rm -rf $RPM_BUILD_ROOT

%preun
if [ -d /usr/share/moose1.4 ];
then
	rm -r /usr/share/moose1.4
fi
if [ -d /usr/share/doc/moose1.4 ];
then
	rm -r /usr/share/doc/moose1.4
fi
rm /usr/bin/moosegui
rm /home/$SUDO_USER/Desktop/MooseGUI.desktop
if [ -d /usr/lib/python2.7/site-packages/moose ];
then
	rm -r /usr/lib/python2.7/site-packages/moose
fi
if [ -d /usr/lib/python2.6/dist-packages/moose ];
then
	rm -r /usr/lib/python2.6/dist-packages/moose
fi
if [ -d /usr/lib/python2.5/site-packages/moose ];
then
	rm -r /usr/lib/python2.5/site-packages/moose
fi
if [ -f /usr/lib/python2.6/dist-packages/_moose.so ];
then 
	rm /usr/lib/python2.6/dist-packages/_moose.so
fi
if [ -f /usr/lib/python2.6/dist-packages/moose.py ];
then 
	rm /usr/lib/python2.6/dist-packages/moose.py
fi
if [ -f /usr/lib/python2.6/dist-packages/moose.pyc ];
then
	rm /usr/lib/python2.6/dist-packages/moose.pyc
fi
if [ -f /usr/lib/python2.6/dist-packages/pymoose.py ];
then 
	rm /usr/lib/python2.6/dist-packages/pymoose.py
fi
if [ -f /usr/lib/python2.6/dist-packages/pymoose.pyc ];
then
	rm /usr/lib/python2.6/dist-packages/pymoose.pyc
fi
if [ -f /usr/lib/python2.5/site-packages/_moose.so ];
then 
	rm /usr/lib/python2.5/site-packages/_moose.so
fi
if [ -f /usr/lib/python2.5/site-packages/moose.py ];
then 
	rm /usr/lib/python2.5/site-packages/moose.py
fi
if [ -f /usr/lib/python2.5/site-packages/moose.pyc ];
then
	rm /usr/lib/python2.5/site-packages/moose.pyc
fi
if [ -f /usr/lib/python2.5/site-packages/pymoose.py ];
then 
	rm /usr/lib/python2.5/site-packages/pymoose.py
fi
if [ -f /usr/lib/python2.5/site-packages/pymoose.pyc ];
then
	rm /usr/lib/python2.5/site-packages/pymoose.pyc
fi
if [ -f /etc/ld.so.conf.d/moose.conf ];
then
	rm /etc/ld.so.conf.d/moose.conf
fi
if [ -f /usr/share/info/pymoose.info ];
then
	sudo install-info --remove --info-dir=/usr/share/info /usr/share/info/pymoose.info
fi

%postun
ldconfig

%files
%defattr(-,root,root,-)
/usr/bin/moose-bin
/usr/share/moose1.4/DEMOS/axon/axon0.genesis.plot
/usr/share/moose1.4/DEMOS/axon/axon100.genesis.plot
/usr/share/moose1.4/DEMOS/axon/axon200.genesis.plot
/usr/share/moose1.4/DEMOS/axon/axon300.genesis.plot
/usr/share/moose1.4/DEMOS/axon/axon400.genesis.plot
/usr/share/moose1.4/DEMOS/axon/Axon.g
/usr/share/moose1.4/DEMOS/axon/axon.p
/usr/share/moose1.4/DEMOS/axon/axon.png
/usr/share/moose1.4/DEMOS/axon/bulbchan.g
/usr/share/moose1.4/DEMOS/axon/compatibility.g
/usr/share/moose1.4/DEMOS/axon/plot.gnuplot
/usr/share/moose1.4/DEMOS/axon-passive/Axon.g
/usr/share/moose1.4/DEMOS/axon-passive/axon.p
/usr/share/moose1.4/DEMOS/axon-passive/axon-passive.p
/usr/share/moose1.4/DEMOS/axon-passive/Axon.py
/usr/share/moose1.4/DEMOS/axon-passive/bulbchan.g
/usr/share/moose1.4/DEMOS/axon-passive/compatibility.g
/usr/share/moose1.4/DEMOS/axon-passive/plot.gnuplot
/usr/share/moose1.4/DEMOS/channels/bulbchan.g
/usr/share/moose1.4/DEMOS/channels/Ca_hip_traub91.png
/usr/share/moose1.4/DEMOS/channels/Channels.g
/usr/share/moose1.4/DEMOS/channels/compatibility.g
/usr/share/moose1.4/DEMOS/channels/hh_tchan.g
/usr/share/moose1.4/DEMOS/channels/K2_mit_usb.png
/usr/share/moose1.4/DEMOS/channels/KA_bsg_yka.png
/usr/share/moose1.4/DEMOS/channels/Ka_hip_traub91.png
/usr/share/moose1.4/DEMOS/channels/Kahp_hip_traub91.png
/usr/share/moose1.4/DEMOS/channels/K_bsg_yka.png
/usr/share/moose1.4/DEMOS/channels/Kdr_hip_traub91.png
/usr/share/moose1.4/DEMOS/channels/K_hh_tchan.png
/usr/share/moose1.4/DEMOS/channels/KM_bsg_yka.png
/usr/share/moose1.4/DEMOS/channels/K_mit_usb.png
/usr/share/moose1.4/DEMOS/channels/LCa3_mit_usb.png
/usr/share/moose1.4/DEMOS/channels/Na_bsg_yka.png
/usr/share/moose1.4/DEMOS/channels/Na_hh_tchan.png
/usr/share/moose1.4/DEMOS/channels/Na_hip_traub91.png
/usr/share/moose1.4/DEMOS/channels/Na_mit_usb.png
/usr/share/moose1.4/DEMOS/channels/Na_rat_smsnn.png
/usr/share/moose1.4/DEMOS/channels/plot.gnuplot
/usr/share/moose1.4/DEMOS/channels/SMSNNchan.g
/usr/share/moose1.4/DEMOS/channels/traub91chan.g
/usr/share/moose1.4/DEMOS/channels/traubchan.g
/usr/share/moose1.4/DEMOS/channels/yamadachan.g
/usr/share/moose1.4/DEMOS/channels/reference_plots/Ca_bsg_yka.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Ca_hip_traub91.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Ca_hip_traub.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/K2_mit_usb.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/KA_bsg_yka.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Ka_hip_traub91.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Kahp_hip_traub91.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/K_bsg_yka.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Kca_hip_traub.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Kca_mit_usb.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Kc_hip_traub91.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Kdr_hip_traub91.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/K_hh_tchan.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/K_hip_traub.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/KM_bsg_yka.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/K_mit_usb.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/LCa3_mit_usb.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Na_bsg_yka.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Na_hh_tchan.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Na_hip_traub91.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Na_mit_usb.genesis.plot
/usr/share/moose1.4/DEMOS/channels/reference_plots/Na_rat_smsnn.genesis.plot
/usr/share/moose1.4/DEMOS/fields/chan.g
/usr/share/moose1.4/DEMOS/fields/compatibility.g
/usr/share/moose1.4/DEMOS/fields/myelin2.p
/usr/share/moose1.4/DEMOS/fields/Myelin.g
/usr/share/moose1.4/DEMOS/fields/nonmyelin2.p
/usr/share/moose1.4/DEMOS/gbar/chan.g
/usr/share/moose1.4/DEMOS/gbar/cMyelin.g
/usr/share/moose1.4/DEMOS/gbar/compatibility.g
/usr/share/moose1.4/DEMOS/gbar/myelin2.p
/usr/share/moose1.4/DEMOS/gbar/Myelin.g
/usr/share/moose1.4/DEMOS/gbar/myelin.png
/usr/share/moose1.4/DEMOS/gbar/nonmyelin2.p
/usr/share/moose1.4/DEMOS/gbar/vMyelin.g
/usr/share/moose1.4/DEMOS/integrate_fire/intfire.g
/usr/share/moose1.4/DEMOS/integrate_fire/plot.gp
/usr/share/moose1.4/DEMOS/integrate_fire/README
/usr/share/moose1.4/DEMOS/kholodenko/Kholodenko.g
/usr/share/moose1.4/DEMOS/kholodenko/kkit.g
/usr/share/moose1.4/DEMOS/kholodenko/plot.gnuplot
/usr/share/moose1.4/DEMOS/kholodenko/reference.plot
/usr/share/moose1.4/DEMOS/mitral-ee/bulbchan.g
/usr/share/moose1.4/DEMOS/mitral-ee/compatibility.g
/usr/share/moose1.4/DEMOS/mitral-ee/mit.p
/usr/share/moose1.4/DEMOS/mitral-ee/Mitral.g
/usr/share/moose1.4/DEMOS/mitral-ee/mitral.genesis.plot
/usr/share/moose1.4/DEMOS/mitral-ee/mitral.png
/usr/share/moose1.4/DEMOS/mitral-ee/plot.gnuplot
/usr/share/moose1.4/DEMOS/network-ee/compatibility.g
/usr/share/moose1.4/DEMOS/network-ee/hh_tchan.g
/usr/share/moose1.4/DEMOS/network-ee/I0.png
/usr/share/moose1.4/DEMOS/network-ee/Network.g
/usr/share/moose1.4/DEMOS/network-ee/network.genesis.plot
/usr/share/moose1.4/DEMOS/network-ee/plot.gnuplot
/usr/share/moose1.4/DEMOS/network-ee/V0.png
/usr/share/moose1.4/DEMOS/network-ee/V9.png
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/defaultNewProject/Generated.net.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/defaultNewProject/KConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/defaultNewProject/LeakConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/defaultNewProject/NaConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/defaultNewProject/README
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/defaultNewProject/SampleCell.morph.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Generated.net.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Gran_CaHVA_98.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Gran_CaPool_98.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Gran_H_98.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Gran_KA_98.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Gran_KCa_98.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Gran_KDr_98.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Gran_NaF_98.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/GranPassiveCond.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/Granule_98.morph.xml
/usr/share/moose1.4/DEMOS/NeuroML_neuroConstruct/GranuleCell/README
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/Ca1.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/CaConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/Ca.moose.plot
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/Ca_NMDA.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/CaPool.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/glu.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/K_AConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/K_AHPConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/K_CConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/K_DRConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/LeakConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/NaConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/NMDA_Ca_conc.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/NMDA.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/plot.gnuplot
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/plotUtil.g
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/Reader.g
/usr/share/moose1.4/DEMOS/NeuroML_Reader/CA1/Vm.moose.plot
/usr/share/moose1.4/DEMOS/NeuroML_Reader/GranuleCell.morph.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/KConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/moose.plot
/usr/share/moose1.4/DEMOS/NeuroML_Reader/NaConductance.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/PassiveCond.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/plot.gnuplot
/usr/share/moose1.4/DEMOS/NeuroML_Reader/plotUtil.g
/usr/share/moose1.4/DEMOS/NeuroML_Reader/PurkinjeCell.morph.xml
/usr/share/moose1.4/DEMOS/NeuroML_Reader/Reader.g
/usr/share/moose1.4/DEMOS/NeuroML_Reader/README
/usr/share/moose1.4/DEMOS/NeuroML_Reader/Vm.genesis.plot
/usr/share/moose1.4/DEMOS/NeuroML_Reader/Vm.moose.plot
/usr/share/moose1.4/DEMOS/nmda/blocked_fraction/blocked_fraction.png
/usr/share/moose1.4/DEMOS/nmda/blocked_fraction/blocked_fraction.py
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.1.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.1.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.2.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.2.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.3.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.3.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.4.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.4.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.5.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.5.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.6.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.6.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.7.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.7.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/c2.Vm.png
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.1.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.1.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.2.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.2.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.3.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.3.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.4.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.4.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.5.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.5.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.6.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.6.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.7.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.7.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/mgblock.Gk.png
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.1.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.1.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.2.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.2.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.3.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.3.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.4.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.4.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.5.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.5.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.6.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.6.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.7.genesis.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.7.moose.plot
/usr/share/moose1.4/DEMOS/nmda/output/syn.Gk.png
/usr/share/moose1.4/DEMOS/nmda/blocked_fraction
/usr/share/moose1.4/DEMOS/nmda/compatibility.g
/usr/share/moose1.4/DEMOS/nmda/NMDA.g
/usr/share/moose1.4/DEMOS/nmda/output
/usr/share/moose1.4/DEMOS/nmda/plot.gnuplot
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/compatibility.g
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/output
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/plot.gnuplot
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/PulseAsc.g
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/PulseAsc.py
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/output/a1.genesis.plot
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/output/a1.png
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/output/a2.genesis.plot
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/output/a2.png
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/output/diff1.png
/usr/share/moose1.4/DEMOS/pulsegen-ascfile/output/diff2.png
/usr/share/moose1.4/DEMOS/rallpack2/branch-0.analytical.plot
/usr/share/moose1.4/DEMOS/rallpack2/branch-0.genesis.plot
/usr/share/moose1.4/DEMOS/rallpack2/branch-0.png
/usr/share/moose1.4/DEMOS/rallpack2/branch-x.analytical.plot
/usr/share/moose1.4/DEMOS/rallpack2/branch-x.genesis.plot
/usr/share/moose1.4/DEMOS/rallpack2/branch-x.png
/usr/share/moose1.4/DEMOS/rallpack2/compatibility.g
/usr/share/moose1.4/DEMOS/rallpack2/plot.gnuplot
/usr/share/moose1.4/DEMOS/rallpack2/Rallpack2.g
/usr/share/moose1.4/DEMOS/rallpack2/util.g
/usr/share/moose1.4/DEMOS/rallpack3/axon-0.genesis.plot
/usr/share/moose1.4/DEMOS/rallpack3/axon-0.png
/usr/share/moose1.4/DEMOS/rallpack3/axon-x.genesis.plot
/usr/share/moose1.4/DEMOS/rallpack3/axon-x.png
/usr/share/moose1.4/DEMOS/rallpack3/compatibility.g
/usr/share/moose1.4/DEMOS/rallpack3/plot.gnuplot
/usr/share/moose1.4/DEMOS/rallpack3/Rallpack3.g
/usr/share/moose1.4/DEMOS/rallpack3/squidchan.g
/usr/share/moose1.4/DEMOS/rallpack3/util.g
/usr/share/moose1.4/DEMOS/sbml_Reader/acc88.xml
/usr/share/moose1.4/DEMOS/sbml_Reader/copasi.plot
/usr/share/moose1.4/DEMOS/sbml_Reader/copasi.xsd
/usr/share/moose1.4/DEMOS/sbml_Reader/plot.gnuplot
/usr/share/moose1.4/DEMOS/sbml_Reader/plotUtil.g
/usr/share/moose1.4/DEMOS/sbml_Reader/Reader.g
/usr/share/moose1.4/DEMOS/sbml_Reader/README
/usr/share/moose1.4/DEMOS/sbml_roundtrip/acc88.xml
/usr/share/moose1.4/DEMOS/sbml_roundtrip/copasi.plot
/usr/share/moose1.4/DEMOS/sbml_roundtrip/copasi.xsd
/usr/share/moose1.4/DEMOS/sbml_roundtrip/plot.gnuplot
/usr/share/moose1.4/DEMOS/sbml_roundtrip/plotUtil.g
/usr/share/moose1.4/DEMOS/sbml_roundtrip/README
/usr/share/moose1.4/DEMOS/sbml_roundtrip/RoundTrip.g
/usr/share/moose1.4/DEMOS/sbml_Writer/acc88.g
/usr/share/moose1.4/DEMOS/sbml_Writer/copasi.plot
/usr/share/moose1.4/DEMOS/sbml_Writer/copasi.xsd
/usr/share/moose1.4/DEMOS/sbml_Writer/kkit.g
/usr/share/moose1.4/DEMOS/sbml_Writer/plot.gnuplot
/usr/share/moose1.4/DEMOS/sbml_Writer/plotUtil.g
/usr/share/moose1.4/DEMOS/sbml_Writer/README
/usr/share/moose1.4/DEMOS/sbml_Writer/Writer.g
/usr/share/moose1.4/DEMOS/squid/plot.gp
/usr/share/moose1.4/DEMOS/squid/plot.py
/usr/share/moose1.4/DEMOS/squid/squid.g
/usr/share/moose1.4/DEMOS/squid-ee/Squid.g
/usr/share/moose1.4/DEMOS/synapse/cable.p
/usr/share/moose1.4/DEMOS/synapse/compatibility.g
/usr/share/moose1.4/DEMOS/synapse/Gk.genesis.plot
/usr/share/moose1.4/DEMOS/synapse/Gk.png
/usr/share/moose1.4/DEMOS/synapse/myelin2.p
/usr/share/moose1.4/DEMOS/synapse/NMDA.g
/usr/share/moose1.4/DEMOS/synapse/plot.gnuplot
/usr/share/moose1.4/DEMOS/synapse/simplechan.g
/usr/share/moose1.4/DEMOS/synapse/Synapse.g
/usr/share/moose1.4/DEMOS/synapse/Vm.genesis.plot
/usr/share/moose1.4/DEMOS/synapse/Vm.png
/usr/share/moose1.4/DEMOS/tab2Dchannel/compatibility.g
/usr/share/moose1.4/DEMOS/tab2Dchannel/loadMoc.g
/usr/share/moose1.4/DEMOS/tab2Dchannel/MoczydKC.g
/usr/share/moose1.4/DEMOS/tab2Dchannel/printtab.g
/usr/share/moose1.4/DEMOS/tab2Dchannel/testmoc.g
/usr/share/moose1.4/DEMOS/tab2Dchannel/XA.txt
/usr/share/moose1.4/DEMOS/tab2Dchannel/XB.txt
/usr/share/moose1.4/DEMOS/timetable/compatibility.g
/usr/share/moose1.4/DEMOS/timetable/plot.gnuplot
/usr/share/moose1.4/DEMOS/timetable/spikes.txt
/usr/share/moose1.4/DEMOS/timetable/TimeTable.g
/usr/share/moose1.4/DEMOS/timetable/TimeTable.py
/usr/share/moose1.4/DEMOS/timetable/output/Gk.genesis.plot
/usr/share/moose1.4/DEMOS/timetable/output/Gk.png
/usr/share/moose1.4/DEMOS/timetable/output/Vm.genesis.plot
/usr/share/moose1.4/DEMOS/timetable/output/Vm.png
/usr/share/moose1.4/DEMOS/traub91/acute.p
/usr/share/moose1.4/DEMOS/traub91/CA1.p
/usr/share/moose1.4/DEMOS/traub91/CA3.p
/usr/share/moose1.4/DEMOS/traub91/Ca.genesis.plot
/usr/share/moose1.4/DEMOS/traub91/Ca.png
/usr/share/moose1.4/DEMOS/traub91/compatibility.g
/usr/share/moose1.4/DEMOS/traub91/plot.gnuplot
/usr/share/moose1.4/DEMOS/traub91/Traub91.g
/usr/share/moose1.4/DEMOS/traub91/traub91proto.g
/usr/share/moose1.4/DEMOS/traub91/Vm.genesis.plot
/usr/share/moose1.4/DEMOS/traub91/Vm.png
/usr/share/moose1.4/DEMOS/pymoose/axon/axon.p
/usr/share/moose1.4/DEMOS/pymoose/axon/axon.py
/usr/share/moose1.4/DEMOS/pymoose/channels/bulbchan.g
/usr/share/moose1.4/DEMOS/pymoose/channels/bulbchan.py
/usr/share/moose1.4/DEMOS/pymoose/channels/compatibility.g
/usr/share/moose1.4/DEMOS/pymoose/channels/test_bulbchan.g
/usr/share/moose1.4/DEMOS/pymoose/channels/test_bulbchan.py
/usr/share/moose1.4/DEMOS/pymoose/device/efield.py
/usr/share/moose1.4/DEMOS/pymoose/device/pulsegengui.py
/usr/share/moose1.4/DEMOS/pymoose/device/pulsegen.py
/usr/share/moose1.4/DEMOS/pymoose/izhikevich/demogui_qt.py
/usr/share/moose1.4/DEMOS/pymoose/izhikevich/demogui_tk.py
/usr/share/moose1.4/DEMOS/pymoose/izhikevich/Izhikevich.py
/usr/share/moose1.4/DEMOS/pymoose/michaelis_menten/guimichaelis.py
/usr/share/moose1.4/DEMOS/pymoose/michaelis_menten/Help.py
/usr/share/moose1.4/DEMOS/pymoose/michaelis_menten/Michaelis.py
/usr/share/moose1.4/DEMOS/pymoose/michaelis_menten/simple.py
/usr/share/moose1.4/DEMOS/pymoose/spikegen/spikegen.py
/usr/share/moose1.4/DEMOS/pymoose/squid/qtSquid.py
/usr/share/moose1.4/DEMOS/pymoose/squid/squidModel.py
/usr/share/moose1.4/DEMOS/pymoose/squid/squid.py
/usr/share/moose1.4/DEMOS/pymoose/traub2005/comps.cvs
/usr/share/moose1.4/DEMOS/pymoose/traub2005/nrn
/usr/share/moose1.4/DEMOS/pymoose/traub2005/py
/usr/share/moose1.4/DEMOS/pymoose/traub2005/README.txt
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/acc79.g
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/borgka.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/borgkm.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/ca3a.geo
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/ca3_db.hoc
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/cadiv.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/cagk.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/cal2.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/can2.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/cat.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/kahp.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/kdr.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/kkit.g
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/moosenrn.py
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/mosinit.hoc
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/nahh.mod
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/README.txt
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/mus_nrn/test_a.hoc
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/squid_fft/model.py
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/squid_fft/README.txt
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/squid_fft/runtest.py
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/squid_fft/squid.py
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/squid/qtSquid.py
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/squid/README.txt
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/squid/squidModel.py
/usr/share/moose1.4/DEMOS/pymoose/ray_bhalla_2008/squid/squid.py
/usr/share/doc/moose1.4/DesignDocument.rtf
/usr/share/doc/moose1.4/hsolve.ppt
/usr/share/doc/moose1.4/HSOLVE.txt
/usr/share/doc/moose1.4/IDS.txt
/usr/share/doc/moose1.4/KinSynChan.tex
/usr/share/doc/moose1.4/MESSAGING.txt
/usr/share/doc/moose1.4/moose.1
/usr/share/doc/moose1.4/MOOSE2007_INTRO.txt
/usr/share/doc/moose1.4/moose_logo.png
/usr/share/doc/moose1.4/mooseLogo.ppt
/usr/share/doc/moose1.4/moose_thumbnail2.png
/usr/share/doc/moose1.4/moose_thumbnail.png
/usr/share/doc/moose1.4/ParallelNeuronSimulation.doc
/usr/share/doc/moose1.4/PARALLEL.txt
/usr/share/doc/moose1.4/PyMOOSE_HowTo.doc
/usr/share/doc/moose1.4/README.parallel
/usr/share/doc/moose1.4/README.txt
/usr/share/doc/moose1.4/INSTALL 
/usr/share/doc/moose1.4/COPYING.LIB
/usr/share/doc/moose1.4/RELEASE_NOTES_BETA_1.4.0.txt
/usr/share/doc/moose1.4/RELEASE_NOTES_BETA_1.4.rtf
/usr/share/doc/moose1.4/SOLVERS.txt
/usr/share/doc/moose1.4/benchmark/acc4_1.6e-21.g
/usr/share/doc/moose1.4/benchmark/acc.plot
/usr/share/doc/moose1.4/benchmark/kholodenko.g
/usr/share/doc/moose1.4/benchmark/kh.plot
/usr/share/doc/moose1.4/benchmark/kkit.g
/usr/share/doc/moose1.4/benchmark/mega_1.6e-21.g
/usr/share/doc/moose1.4/benchmark/mega.plot
/usr/share/doc/moose1.4/benchmark/nonscaf6_1.6e-21b.g
/usr/share/doc/moose1.4/benchmark/nonscaf.plot
/usr/share/doc/moose1.4/benchmark/RESULTS
/usr/share/doc/moose1.4/gui/afterDrag.png
/usr/share/doc/moose1.4/gui/afterRun.png
/usr/share/doc/moose1.4/gui/clickedSoma.png
/usr/share/doc/moose1.4/gui/doc.html
/usr/share/doc/moose1.4/gui/doc.org
/usr/share/doc/moose1.4/gui/doc.pdf
/usr/share/doc/moose1.4/gui/doc.tex
/usr/share/doc/moose1.4/gui/firsttime1.png
/usr/share/doc/moose1.4/gui/firsttime2.png
/usr/share/doc/moose1.4/gui/firsttime3.png
/usr/share/doc/moose1.4/gui/firsttime4.png
/usr/share/doc/moose1.4/gui/kinetikit.png
/usr/share/doc/moose1.4/gui/layout.png
/usr/share/doc/moose1.4/gui/loadmodel.png
/usr/share/doc/moose1.4/gui/missfont.log
/usr/share/doc/moose1.4/gui/newGL.png
/usr/share/doc/moose1.4/gui/plotCombo.png
/usr/share/doc/moose1.4/gui/plottingTimeStep.png
/usr/share/doc/moose1.4/pymoose/backcompat.texi
/usr/share/doc/moose1.4/pymoose/copying.texi
/usr/share/doc/moose1.4/pymoose/detailed.texi
/usr/share/doc/moose1.4/pymoose/faq.texi
/usr/share/doc/moose1.4/pymoose/fdl.texi
/usr/share/doc/moose1.4/pymoose/index.texi
/usr/share/doc/moose1.4/pymoose/install.texi
/usr/share/doc/moose1.4/pymoose/intro.texi
/usr/share/doc/moose1.4/pymoose/Makefile
/usr/share/doc/moose1.4/pymoose/pymoose
/usr/share/doc/moose1.4/pymoose/pymoose.html
/usr/share/doc/moose1.4/pymoose/pymoose.info
/usr/share/doc/moose1.4/pymoose/pymoose.pdf
/usr/share/doc/moose1.4/pymoose/pymoose.texi
/usr/share/doc/moose1.4/pymoose/quickstart.texi
/usr/share/moose1.4/TESTS/GSSA/acc4_1.6e-18.g
/usr/share/moose1.4/TESTS/GSSA/acc4_1.6e-21.g
/usr/share/moose1.4/TESTS/GSSA/acc4_1e-18.plot
/usr/share/moose1.4/TESTS/GSSA/acc4_moose_rk5.plot
/usr/share/moose1.4/TESTS/GSSA/assignReac.g
/usr/share/moose1.4/TESTS/GSSA/enz.g
/usr/share/moose1.4/TESTS/GSSA/enztot.g
/usr/share/moose1.4/TESTS/GSSA/kkit.g
/usr/share/moose1.4/TESTS/GSSA/manyvolsAssignReac.g
/usr/share/moose1.4/TESTS/GSSA/manyvolsenz.g
/usr/share/moose1.4/TESTS/GSSA/manyvolsenztot.g
/usr/share/moose1.4/TESTS/GSSA/manyvols.g
/usr/share/moose1.4/TESTS/GSSA/manyvolsMMenz.g
/usr/share/moose1.4/TESTS/GSSA/manyvolsrMMenz.g
/usr/share/moose1.4/TESTS/GSSA/manyvolstotenz.g
/usr/share/moose1.4/TESTS/GSSA/MMenz.g
/usr/share/moose1.4/TESTS/GSSA/reac2.g
/usr/share/moose1.4/TESTS/GSSA/reac.g
/usr/share/moose1.4/TESTS/GSSA/rMMenz.g
/usr/share/moose1.4/TESTS/GSSA/testDynamicBuffering.g
/usr/share/moose1.4/TESTS/GSSA/test_volscale.g
/usr/share/moose1.4/TESTS/GSSA/totenz.g
/usr/share/moose1.4/TESTS/parallel/arrayElement.g
/usr/share/moose1.4/TESTS/parallel/ce.g
/usr/share/moose1.4/TESTS/parallel/do_regression.bat
/usr/share/moose1.4/TESTS/parallel/do_regression_win.bat
/usr/share/moose1.4/TESTS/parallel/element_manipulation.g
/usr/share/moose1.4/TESTS/parallel/hh_tchan.g
/usr/share/moose1.4/TESTS/parallel/msg.g
/usr/share/moose1.4/TESTS/parallel/network.g
/usr/share/moose1.4/TESTS/parallel/network.plot
/usr/share/moose1.4/TESTS/parallel/showField.g
/usr/share/moose1.4/TESTS/parallel/synchan.g
/usr/share/moose1.4/TESTS/pymoose/neuron_nmda.dat
/usr/share/moose1.4/TESTS/pymoose/test_nmda.py
/usr/share/moose1.4/TESTS/pymoose/test_stpnmdachan.py
/usr/share/moose1.4/TESTS/pymoose/test_stpsynchan.py
/usr/share/moose1.4/TESTS/pymoose/test_synchan.py
/usr/share/moose1.4/TESTS/regression/acc88_copasi.plot
/usr/share/moose1.4/TESTS/regression/axon
/usr/share/moose1.4/TESTS/regression/bulbchan.g
/usr/share/moose1.4/TESTS/regression/cable.p
/usr/share/moose1.4/TESTS/regression/channelplots
/usr/share/moose1.4/TESTS/regression/do_regression.bat
/usr/share/moose1.4/TESTS/regression/do_regression_win.bat
/usr/share/moose1.4/TESTS/regression/globalParms.p
/usr/share/moose1.4/TESTS/regression/hh_tchan.g
/usr/share/moose1.4/TESTS/regression/mit.p
/usr/share/moose1.4/TESTS/regression/mitral-ee
/usr/share/moose1.4/TESTS/regression/moose_axon.g
/usr/share/moose1.4/TESTS/regression/moose_axon.plot
/usr/share/moose1.4/TESTS/regression/moose_channels.g
/usr/share/moose1.4/TESTS/regression/moose_file2tab2file.g
/usr/share/moose1.4/TESTS/regression/moose_file2tab.plot
/usr/share/moose1.4/TESTS/regression/moose_kholodenko.g
/usr/share/moose1.4/TESTS/regression/moose_kholodenko.plot
/usr/share/moose1.4/TESTS/regression/moose_mitral.g
/usr/share/moose1.4/TESTS/regression/moose_mitral.plot
/usr/share/moose1.4/TESTS/regression/moose_network.g
/usr/share/moose1.4/TESTS/regression/moose_network.plot
/usr/share/moose1.4/TESTS/regression/moose_NeuroML_reader.g
/usr/share/moose1.4/TESTS/regression/moose_NeuroMLReader.plot
/usr/share/moose1.4/TESTS/regression/moose_rallpack1.g
/usr/share/moose1.4/TESTS/regression/moose_rallpack1.plot
/usr/share/moose1.4/TESTS/regression/moose_rallpack2.g
/usr/share/moose1.4/TESTS/regression/moose_rallpack2.plot
/usr/share/moose1.4/TESTS/regression/moose_rallpack3.g
/usr/share/moose1.4/TESTS/regression/moose_rallpack3.plot
/usr/share/moose1.4/TESTS/regression/moose_readcell.g
/usr/share/moose1.4/TESTS/regression/moose_readcell_global_parms.g
/usr/share/moose1.4/TESTS/regression/moose_readcell_global_parms.plot
/usr/share/moose1.4/TESTS/regression/moose_readcell.plot
/usr/share/moose1.4/TESTS/regression/moose_sbml_reader.g
/usr/share/moose1.4/TESTS/regression/moose_sbml_read_write.g
/usr/share/moose1.4/TESTS/regression/moose_squid.g
/usr/share/moose1.4/TESTS/regression/moose_squid.plot
/usr/share/moose1.4/TESTS/regression/moose_synapse.g
/usr/share/moose1.4/TESTS/regression/moose_synapse.plot
/usr/share/moose1.4/TESTS/regression/moose_synchan.g
/usr/share/moose1.4/TESTS/regression/moose_synchan.plot
/usr/share/moose1.4/TESTS/regression/moose_traub91.g
/usr/share/moose1.4/TESTS/regression/moose_traub91.plot
/usr/share/moose1.4/TESTS/regression/myelin2.p
/usr/share/moose1.4/TESTS/regression/neardiff
/usr/share/moose1.4/TESTS/regression/neardiff.cpp
/usr/share/moose1.4/TESTS/regression/network-ee
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader
/usr/share/moose1.4/TESTS/regression/rallpack1
/usr/share/moose1.4/TESTS/regression/rallpack2
/usr/share/moose1.4/TESTS/regression/rallpack3
/usr/share/moose1.4/TESTS/regression/README
/usr/share/moose1.4/TESTS/regression/sbml_Reader
/usr/share/moose1.4/TESTS/regression/sbml_Read_Write
/usr/share/moose1.4/TESTS/regression/simplechan.g
/usr/share/moose1.4/TESTS/regression/SMSNNchan.g
/usr/share/moose1.4/TESTS/regression/soma.p
/usr/share/moose1.4/TESTS/regression/synapse
/usr/share/moose1.4/TESTS/regression/test_nmda_mgblock.g
/usr/share/moose1.4/TESTS/regression/traub91
/usr/share/moose1.4/TESTS/regression/traub91chan.g
/usr/share/moose1.4/TESTS/regression/traubchan.g
/usr/share/moose1.4/TESTS/regression/v20_copasi.plot
/usr/share/moose1.4/TESTS/regression/v20_moose.plot
/usr/share/moose1.4/TESTS/regression/yamadachan.g
/usr/share/moose1.4/TESTS/regression/axon/axon.p
/usr/share/moose1.4/TESTS/regression/axon/bulbchan.g
/usr/share/moose1.4/TESTS/regression/axon/compatibility.g
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Ca_bsg_yka.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Ca_hip_traub91.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Ca_hip_traub.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_K2_mit_usb.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_KA_bsg_yka.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Ka_hip_traub91.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Kahp_hip_traub91.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_K_bsg_yka.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Kca_hip_traub.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Kc_hip_traub91.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Kdr_hip_traub91.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_K_hh_tchan.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_K_hip_traub.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_KM_bsg_yka.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_K_mit_usb.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_LCa3_mit_usb.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Na_bsg_yka.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Na_hh_tchan.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Na_hip_traub91.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Na_mit_usb.plot
/usr/share/moose1.4/TESTS/regression/channelplots/moose_Na_rat_smsnn.plot
/usr/share/moose1.4/TESTS/regression/mitral-ee/bulbchan.g
/usr/share/moose1.4/TESTS/regression/mitral-ee/compatibility.g
/usr/share/moose1.4/TESTS/regression/mitral-ee/mit.p
/usr/share/moose1.4/TESTS/regression/network-ee/compatibility.g
/usr/share/moose1.4/TESTS/regression/network-ee/hh_tchan.g
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader/GranuleCell.morph.xml
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader/KConductance.xml
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader/moose_NeuroML_reader.g
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader/NaConductance.xml
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader/PassiveCond.xml
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader/plotUtil.g
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader/PurkinjeCell.morph.xml
/usr/share/moose1.4/TESTS/regression/NeuroML_Reader/README
/usr/share/moose1.4/TESTS/regression/rallpack1/compatibility.g
/usr/share/moose1.4/TESTS/regression/rallpack1/util.g
/usr/share/moose1.4/TESTS/regression/rallpack2/compatibility.g
/usr/share/moose1.4/TESTS/regression/rallpack2/util.g
/usr/share/moose1.4/TESTS/regression/rallpack3/compatibility.g
/usr/share/moose1.4/TESTS/regression/rallpack3/squidchan.g
/usr/share/moose1.4/TESTS/regression/rallpack3/util.g
/usr/share/moose1.4/TESTS/regression/sbml_Reader/acc88_moose.plot
/usr/share/moose1.4/TESTS/regression/sbml_Reader/acc88.xml
/usr/share/moose1.4/TESTS/regression/sbml_Reader/long_copasi.plot
/usr/share/moose1.4/TESTS/regression/sbml_Reader/plotUtil.g
/usr/share/moose1.4/TESTS/regression/sbml_Reader/README
/usr/share/moose1.4/TESTS/regression/sbml_Reader/v20_copasi.plot
/usr/share/moose1.4/TESTS/regression/sbml_Reader/v20_long.plot
/usr/share/moose1.4/TESTS/regression/sbml_Reader/v20_moose.plot
/usr/share/moose1.4/TESTS/regression/sbml_Reader/v20.xml
/usr/share/moose1.4/TESTS/regression/sbml_Read_Write/acc88.xml
/usr/share/moose1.4/TESTS/regression/sbml_Read_Write/plotUtil.g
/usr/share/moose1.4/TESTS/regression/sbml_Read_Write/README
/usr/share/moose1.4/TESTS/regression/synapse/cable.p
/usr/share/moose1.4/TESTS/regression/synapse/compatibility.g
/usr/share/moose1.4/TESTS/regression/synapse/myelin2.p
/usr/share/moose1.4/TESTS/regression/synapse/simplechan.g
/usr/share/moose1.4/TESTS/regression/traub91/CA3.p
/usr/share/moose1.4/TESTS/regression/traub91/compatibility.g
/usr/share/moose1.4/TESTS/regression/traub91/traub91proto.g
/usr/share/moose1.4/TESTS/robot/acc79.g
/usr/share/moose1.4/TESTS/robot/cell16.g
/usr/share/moose1.4/TESTS/robot/cell16.p
/usr/share/moose1.4/TESTS/robot/cyl50.plot
/usr/share/moose1.4/TESTS/robot/cylinder.g
/usr/share/moose1.4/TESTS/robot/fix79.g
/usr/share/moose1.4/TESTS/robot/full.g
/usr/share/moose1.4/TESTS/robot/kkit.g
/usr/share/moose1.4/TESTS/robot/ksolveArrayDanglingOff2.g
/usr/share/moose1.4/TESTS/robot/ksolveArrayDanglingOn2.g
/usr/share/moose1.4/TESTS/robot/ksolveArrayOff.g
/usr/share/moose1.4/TESTS/robot/ksolveArrayOn.g
/usr/share/moose1.4/TESTS/robot/NOTES
/usr/share/moose1.4/TESTS/robot/proto16.g
/usr/share/moose1.4/TESTS/robot/run16.g
/usr/share/moose1.4/TESTS/robot/sigNeur1.g
/usr/share/moose1.4/TESTS/robot/sigNeur2.g
/usr/share/moose1.4/TESTS/robot/sigNeur3.g
/usr/share/moose1.4/TESTS/robot/sigNeur5.g
/usr/share/moose1.4/TESTS/robot/sigNeurTiny.g
/usr/share/moose1.4/TESTS/robot/soma.g
/usr/share/moose1.4/TESTS/robot/soma.p
/usr/share/moose1.4/TESTS/robot/soma_sig.g
/usr/share/info/pymoose.info.gz
/usr/share/man/man1/moose.1.gz
/usr/share/moose1.4/lib/libsbml.so
#/usr/share/moose1.4/lib/libxml2.so.2
/usr/share/moose1.4/gui/biomodelsclient.py
/usr/share/moose1.4/gui/config.py
/usr/share/moose1.4/gui/documentation.pdf
/usr/share/moose1.4/gui/firsttime.py
/usr/share/moose1.4/gui/icons
/usr/share/moose1.4/gui/layout.py
/usr/share/moose1.4/gui/mooseclasses.py
/usr/share/moose1.4/gui/mooseglobals.py
/usr/share/moose1.4/gui/MooseGUI.desktop
/usr/share/moose1.4/gui/moosegui.py
/usr/share/moose1.4/gui/moosehandler.py
/usr/share/moose1.4/gui/mooseplot.py
/usr/share/moose1.4/gui/mooseshell.py
/usr/share/moose1.4/gui/moosestart.py
/usr/share/moose1.4/gui/moosetree.py
/usr/share/moose1.4/gui/objectedit.py
/usr/share/moose1.4/gui/oglfunc
/usr/share/moose1.4/gui/plotconfig.py
/usr/share/moose1.4/gui/PyCute.py
/usr/share/moose1.4/gui/PyGLWidget.py
/usr/share/moose1.4/gui/test_soap.py
/usr/share/moose1.4/gui/test_urldownload.py
/usr/share/moose1.4/gui/updatepaintGL.py
/usr/share/moose1.4/gui/vizParasDialogue.py
/usr/share/moose1.4/gui/vizParasDialogue.ui
/usr/share/moose1.4/gui/icons/continue.png
/usr/share/moose1.4/gui/icons/help.png
/usr/share/moose1.4/gui/icons/moose_icon.png
/usr/share/moose1.4/gui/icons/QMdiBackground.png
/usr/share/moose1.4/gui/icons/reset.png
/usr/share/moose1.4/gui/icons/run.png
/usr/share/moose1.4/gui/oglfunc/colors
/usr/share/moose1.4/gui/oglfunc/group.py
/usr/share/moose1.4/gui/oglfunc/__init__.py
/usr/share/moose1.4/gui/oglfunc/objects.py
/usr/share/moose1.4/gui/oglfunc/colors/fire
/usr/share/moose1.4/gui/oglfunc/colors/greenfire
/usr/share/moose1.4/gui/oglfunc/colors/grey
/usr/share/moose1.4/gui/oglfunc/colors/heat
/usr/share/moose1.4/gui/oglfunc/colors/jet
/usr/share/moose1.4/gui/oglfunc/colors/redhot
/usr/share/moose1.4/py_stage/moose/neuroml/ChannelML.py
/usr/share/moose1.4/py_stage/moose/neuroml/__init__.py
/usr/share/moose1.4/py_stage/moose/neuroml/MorphML.py
/usr/share/moose1.4/py_stage/moose/neuroml/NetworkML.py
/usr/share/moose1.4/py_stage/moose/neuroml/NeuroML.py
/usr/share/moose1.4/py_stage/moose/neuroml/neuroml_utils.py
/usr/share/moose1.4/py_stage/moose/__init__.py
/usr/share/moose1.4/py_stage/moose/mooseConstants.py
/usr/share/moose1.4/py_stage/moose/moose.py
/usr/share/moose1.4/py_stage/moose/_moose.so
/usr/share/moose1.4/py_stage/moose/neuroml
/usr/share/moose1.4/py_stage/moose/utils.py
/usr/share/moose1.4/README.txt
/usr/share/moose1.4/find_python_path.py
/usr/bin/moose


#%doc README.txt
#%doc INSTALL 
#%doc COPYING.LIB 


%changelog
* Thu Nov 11 2010  <subhasis@ncbs.res.in> - beta-1.4.0
- Initial RPM build.
