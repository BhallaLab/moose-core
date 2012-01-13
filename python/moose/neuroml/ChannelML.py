from xml.etree import ElementTree as ET
import string
import os, sys
from math import *

import moose
from moose.neuroml import utils

class ChannelML():

    def __init__(self,nml_params):
        self.cml='http://morphml.org/channelml/schema'
        self.nml_params = nml_params
        self.temperature = nml_params['temperature']

    def readChannelMLFromFile(self,filename,params={}):
        """ specify params as a dict: e.g. temperature that you need to pass to channels """
        tree = ET.parse(filename)
        channelml_element = tree.getroot()
        for channel in channelml_element.findall('.//{'+self.cml+'}channel_type'):
            ## ideally I should read in extra params from within the channel_type element
            ## and put those in also. Global params should override local ones.
            self.readChannelML(channel,params,channelml_element.attrib['units'])
        for synapse in channelml_element.findall('.//{'+self.cml+'}synapse_type'):
            self.readSynapseML(synapse,channelml_element.attrib['units'])
        for ionConc in channelml_element.findall('.//{'+self.cml+'}ion_concentration'):
            self.readIonConcML(ionConc,channelml_element.attrib['units'])

    def readSynapseML(self,synapseElement,units="SI units"):
        if 'Physiological Units' in units: # see pg 219 (sec 13.2) of Book of Genesis
            Vfactor = 1e-3 # V from mV
            Tfactor = 1e-3 # s from ms
            Gfactor = 1e-3 # S from mS       
        elif 'SI Units' in units:
            Vfactor = 1.0
            Tfactor = 1.0
            Gfactor = 1.0
        else:
            print "wrong units", units,": exiting ..."
            sys.exit(1)
        print "loading synapse :",synapseElement.attrib['name'],"into /library ."
        moosesynapse = moose.SynChan('/library/'+synapseElement.attrib['name'])
        doub_exp_syn = synapseElement.find('./{'+self.cml+'}doub_exp_syn')
        moosesynapse.Ek = float(doub_exp_syn.attrib['reversal_potential'])*Vfactor
        moosesynapse.Gbar = float(doub_exp_syn.attrib['max_conductance'])*Gfactor
        moosesynapse.tau1 = float(doub_exp_syn.attrib['rise_time'])*Tfactor # seconds
        moosesynapse.tau2 = float(doub_exp_syn.attrib['decay_time'])*Tfactor # seconds
        ### The delay and weight can be set only after connecting a spike event generator.
        ### delay and weight are arrays: multiple event messages can be connected to a single synapse
        moosesynapse.addField('graded')
        moosesynapse.setField('graded','False')
        moosesynapse.addField('mgblock')
        moosesynapse.setField('mgblock','False')
      
    def readChannelML(self,channelElement,params={},units="SI units"):
        ## I first calculate all functions assuming a consistent system of units.
        ## While filling in the A and B tables, I just convert to SI.
        ## Also convert gmax and Erev.
        if 'Physiological Units' in units: # see pg 219 (sec 13.2) of Book of Genesis
            Vfactor = 1e-3 # V from mV
            Tfactor = 1e-3 # s from ms
            Gfactor = 1e1 # S/m^2 from mS/cm^2  
            concfactor = 1e6 # Mol = mol/m^-3 from mol/cm^-3      
        elif 'SI Units' in units:
            Vfactor = 1.0
            Tfactor = 1.0
            Gfactor = 1.0
            concfactor = 1.0
        else:
            print "wrong units", units,": exiting ..."
            sys.exit(1)
        channel_name = channelElement.attrib['name']
        print "loading channel :", channel_name,"into /library ."
        IVrelation = channelElement.find('./{'+self.cml+'}current_voltage_relation')
        concdep = IVrelation.find('./{'+self.cml+'}conc_dependence')
        if concdep is None:
            moosechannel = moose.HHChannel('/library/'+channel_name)
        else:
            moosechannel = moose.HHChannel2D('/library/'+channel_name)
        if IVrelation.attrib['cond_law']=="ohmic":
            moosechannel.Gbar = float(IVrelation.attrib['default_gmax']) * Gfactor
            moosechannel.Ek = float(IVrelation.attrib['default_erev']) * Vfactor
            moosechannel.addField('ion')
            moosechannel.setField('ion',IVrelation.attrib['ion'])
            if concdep is not None:
                moosechannel.addField('ionDependency')
                moosechannel.setField('ionDependency',concdep.attrib['ion'])
                
            
        gates = IVrelation.findall('./{'+self.cml+'}gate')
        if len(gates)>3:
            print "Sorry! Maximum x, y, and z (three) gates are possible in MOOSE/Genesis"
            sys.exit()
        moosegates = [['Xpower','xGate'],['Ypower','yGate'],['Zpower','zGate']]
        ## if impl_prefs tag is present change VMIN, VMAX and NDIVS
        impl_prefs = channelElement.find('./{'+self.cml+'}impl_prefs')
        if impl_prefs is not None:
            table_settings = impl_prefs.find('./{'+self.cml+'}table_settings')
            ## some problem here... disable
            VMIN_here = float(table_settings.attrib['min_v'])
            VMAX_here = float(table_settings.attrib['max_v'])
            NDIVS_here = int(table_settings.attrib['table_divisions'])
            dv_here = (VMAX_here - VMIN_here) / NDIVS_here
        else:
            ## default VMIN, VMAX and dv are in SI
            ## convert them to current calculation units used by channel definition
            ## while loading into tables, convert them back to SI
            VMIN_here = VMIN/Vfactor
            VMAX_here = VMAX/Vfactor
            NDIVS_here = NDIVS
            dv_here = dv/Vfactor
        offset = IVrelation.find('./{'+self.cml+'}offset')
        if offset is None: vNegOffset = 0.0
        else: vNegOffset = float(offset.attrib['value'])
        self.parameters = []
        for parameter in channelElement.findall('.//{'+self.cml+'}parameter'):
            self.parameters.append( (parameter.attrib['name'],float(parameter.attrib['value'])) )
        
        for num,gate in enumerate(gates):
            # if no q10settings tag, the q10factor remains 1.0
            # if present but no gate attribute, then set q10factor
            # if there is a gate attribute, then set it only if gate attrib matches gate name
            self.q10factor = 1.0
            self.gate_name = gate.attrib['name']
            for q10settings in IVrelation.findall('./{'+self.cml+'}q10_settings'):
                ## self.temperature from neuro_utils.py
                if 'gate' in q10settings.attrib.keys():
                    if q10settings.attrib['gate'] == self.gate_name:
                        self.setQ10(q10settings)
                        break
                else:
                    self.setQ10(q10settings)

            ## only if you create moosechannel.Xpower will the xGate be created, so do that first below
            ## I cannot use the below single-line list-based way of setting Xpower, etc.
            ## because the getters and setters of swig don't get called this way!!!
            ## complicated way to do moosechannel.Xpower = 1.0 ! but doesn't work as written above
            #vars(moosechannel)[moosegates[num][0]] = float(gate.attrib['instances'])  
            if num==0:
                moosechannel.Xpower = float(gate.attrib['instances'])
                if concdep is not None: moosechannel.Xindex = "VOLT_C1_INDEX"
            elif num==1:
                moosechannel.Ypower = float(gate.attrib['instances'])
                if concdep is not None: moosechannel.Yindex = "VOLT_C1_INDEX"
            elif num==2:
                moosechannel.Zpower = float(gate.attrib['instances'])
                if concdep is not None: moosechannel.Zindex = "VOLT_C1_INDEX"
            ## wrap the xGate, yGate or zGate
            if concdep is None:
                moosegate = moose.HHGate(moosechannel.path+'/'+moosegates[num][1])
            else:
                moosegate = moose.HHGate2D(moosechannel.path+'/'+moosegates[num][1])
            ## set SI values inside MOOSE
            moosegate.A.xmin = VMIN_here*Vfactor
            moosegate.A.xmax = VMAX_here*Vfactor
            moosegate.A.xdivs = NDIVS_here
            moosegate.B.xmin = VMIN_here*Vfactor
            moosegate.B.xmax = VMAX_here*Vfactor
            moosegate.B.xdivs = NDIVS_here
          
            for transition in gate.findall('./{'+self.cml+'}transition'):
                ## make python functions with names of transitions...
                fn_name = transition.attrib['name']
                ## I assume that transitions if present are called alpha and beta
                ## for forward and backward transitions...
                if fn_name in ['alpha','beta']:
                    self.make_cml_function(transition, fn_name, concdep)
                else:
                    print "Unsupported transition ", name
                    sys.exit()

            ## non Ca dependent channel
            if concdep is None:
                time_course = gate.find('./{'+self.cml+'}time_course')
                ## tau is divided by self.q10factor in make_function()
                ## thus, it gets divided irrespective of <time_course> tag present or not.
                if time_course is None:     # if time_course element is not present,
                                            # there have to be alpha and beta transition elements
                    self.make_function('tau','generic',\
                        expr_string="1/(self.alpha(v)+self.beta(v))")
                else:
                    self.make_cml_function(time_course, 'tau')
                steady_state = gate.find('./{'+self.cml+'}steady_state')
                if steady_state is None:    # if steady_state element is not present,
                                            # there have to be alpha and beta transition elements
                    self.make_function('inf','generic',\
                        expr_string="self.alpha(v)/(self.alpha(v)+self.beta(v))")
                else:
                    self.make_cml_function(steady_state, 'inf')

                ## while calculating, use the units used in xml defn,
                ## while filling in table, I convert to SI units.
                v = VMIN_here - vNegOffset
                for i in range(NDIVS_here+1):
                    inf = self.inf(v)
                    tau = self.tau(v)
                    ## convert to SI before writing to table
                    moosegate.A[i] = inf/tau / Tfactor
                    moosegate.B[i] = 1.0/tau / Tfactor
                    v += dv_here
            
            ## Ca dependent channel
            else:
                ## HHGate2D is not wrapped properly in pyMOOSE.
                ## ymin, ymax and ydivs are not exposed.
                ## Setting them creates new and useless attributes within python HHGate2D without warning!
                ## Hence use runG to set these via Genesis command
                ## UNITS: while calculating, use the units used in xml defn,
                ##        while filling in table, I convert to SI units.
                #~ self.context.runG("setfield "+moosegate.path+"/A"+\
                    #~ #" ydivs "+str(CaNDIVS)+\ # these get overridden by the number of values in the table
                    #~ " ymin "+str(float(concdep.attrib['min_conc'])*concfactor)+\
                    #~ " ymax "+str(float(concdep.attrib['max_conc'])*concfactor))
                #~ self.context.runG("setfield "+moosegate.path+"/B"+\
                    #~ #" ydivs "+str(CaNDIVS)+\ # these get overridden by the number of values in the table
                    #~ " ymin "+str(float(concdep.attrib['min_conc'])*concfactor)+\
                    #~ " ymax "+str(float(concdep.attrib['max_conc'])*concfactor))
                ## for Ca dep channel, I expect only generic alpha and beta functions
                ## these have already been made above
                ftableA = open("CaDepA.dat","w")
                ftableB = open("CaDepB.dat","w")
                v = VMIN_here - vNegOffset
                CaMIN = float(concdep.attrib['min_conc'])
                CaMAX = float(concdep.attrib['max_conc'])
                CaNDIVS = 100
                dCa = (CaMAX-CaMIN)/CaNDIVS
                for i in range(NDIVS_here+1):
                    Ca = CaMIN
                    for j in range(CaNDIVS+1):
                        ## convert to SI before writing to table
                        ## in non-Ca channels, I put in q10factor into tau,
                        ## which percolated to A and B
                        ## Here, I do not calculate tau, so put q10factor directly into A and B.
                        alpha = self.alpha(v,Ca)*self.q10factor/Tfactor
                        ftableA.write(str(alpha)+" ")
                        ftableB.write(str(alpha+self.beta(v,Ca)*self.q10factor/Tfactor)+" ")
                        Ca += dCa
                    ftableA.write("\n")
                    ftableB.write("\n")
                    v += dv_here
                ftableA.close()
                ftableB.close()

                ### PRESENTLY, Interpol2D.cpp in MOOSE only allows loading via a data file,
                ### one cannot set individual entries A[0][0] etc.
                ### Thus pyMOOSE also has not wrapped Interpol2D
                #~ self.context.runG("call "+moosegate.path+"/A load CaDepA.dat 0")
                #~ self.context.runG("call "+moosegate.path+"/B load CaDepB.dat 0")
                os.remove('CaDepA.dat')
                os.remove('CaDepB.dat')

    def setQ10(self,q10settings):
        if 'q10_factor' in q10settings.attrib.keys():
            self.q10factor = float(q10settings.attrib['q10_factor'])\
                **((self.temperature-float(q10settings.attrib['experimental_temp']))/10.0)
        elif 'fixed_q10' in q10settings.attrib.keys():
            self.q10factor = float(q10settings.attrib['fixed_q10'])


    def readIonConcML(self, ionConcElement, units="SI units"):
        if units == 'Physiological Units': # see pg 219 (sec 13.2) of Book of Genesis
            Vfactor = 1e-3 # V from mV
            Tfactor = 1e-3 # s from ms
            Gfactor = 1e1 # S/m^2 from mS/cm^2
            concfactor = 1e6 # mol/m^3 from mol/cm^3
            Lfactor = 1e-2 # m from cm
        else:
            Vfactor = 1.0
            Tfactor = 1.0
            Gfactor = 1.0
            concfactor = 1.0
            Lfactor = 1.0
        ionSpecies = ionConcElement.find('./{'+self.cml+'}ion_species')
        if ionSpecies is not None:
            if not 'ca' in ionSpecies.attrib['name']:
                print "Sorry, I cannot handle non-Ca-ion pools. Exiting ..."
                sys.exit(1)
        capoolName = ionConcElement.attrib['name']
        print "loading Ca pool :",capoolName,"into /library ."
        caPool = moose.CaConc('/library/'+capoolName)
        poolModel = ionConcElement.find('./{'+self.cml+'}decaying_pool_model')
        caPool.CaBasal = float(poolModel.attrib['resting_conc']) * concfactor
        caPool.Ca_base = float(poolModel.attrib['resting_conc']) * concfactor
        caPool.tau = float(poolModel.attrib['decay_constant']) * Tfactor
        volInfo = poolModel.find('./{'+self.cml+'}pool_volume_info')
        caPool.thick = float(volInfo.attrib['shell_thickness']) * Lfactor
        ## Put a high ceiling and floor for the Ca conc
        #~ self.context.runG('setfield ' + caPool.path + ' ceiling 1e6')
        #~ self.context.runG('setfield ' + caPool.path + ' floor 0.0')

    def make_cml_function(self, element, fn_name, concdep=None):
        fn_type = element.attrib['expr_form']
        if fn_type in ['exponential','sigmoid','exp_linear']:
            fn = self.make_function( fn_name, fn_type, rate=float(element.attrib['rate']),\
                midpoint=float(element.attrib['midpoint']), scale=float(element.attrib['scale'] ) )
        elif fn_type == 'generic':
            ## OOPS! These expressions should be in SI units, since I converted to SI
            ## Ideally I should not convert to SI and have the user consistently use one or the other
            ## Or convert constants in these expressions to SI, only voltage and ca_conc appear!
            expr_string = element.attrib['expr']
            if concdep is None: ca_name = ''                        # no Ca dependence
            else: ca_name = ','+concdep.attrib['variable_name']     # Ca dependence
            expr_string = self.replace(expr_string, 'alpha', 'self.alpha(v'+ca_name+')')
            expr_string = self.replace(expr_string, 'beta', 'self.beta(v'+ca_name+')')                
            fn = self.make_function( fn_name, fn_type, expr_string=expr_string, concdep=concdep )
        else:
            print "Unsupported function type ", fn_type
            sys.exit()
                
    def make_function(self, fn_name, fn_type, **kwargs):
        """ This dynamically creates a function called fn_name
        If fn_type is exponential, sigmoid or exp_linear,
            **kwargs is a dict having keys rate, midpoint and scale.
        If fin_type is generic, **kwargs is a dict having key expr_string """
        if fn_type == 'exponential':
            def fn(self,v):
                return kwargs['rate']*exp((v-kwargs['midpoint'])/kwargs['scale'])
        elif fn_type == 'sigmoid':
            def fn(self,v):
                return kwargs['rate'] / ( 1 + exp((v-kwargs['midpoint'])/kwargs['scale']) )
        elif fn_type == 'exp_linear':
            def fn(self,v):
                if v-kwargs['midpoint'] == 0.0: return kwargs['rate']
                else:
                    return kwargs['rate'] * ((v-kwargs['midpoint'])/kwargs['scale']) \
                        / ( 1 - exp((kwargs['midpoint']-v)/kwargs['scale']) )
        elif fn_type == 'generic':
            ## python cannot evaluate the ternary operator ?:, so evaluate explicitly
            ## for security purposes eval() is not allowed any __builtins__
            ## but only safe functions in safe_dict (from neuroml_utils.py) and
            ## only the local variables/functions v, self
            allowed_locals = {'self':self}
            allowed_locals.update(safe_dict)
            def fn(self,v,ca=None):
                expr_str = kwargs['expr_string']
                allowed_locals['v'] = v
                allowed_locals['celsius'] = self.temperature
                allowed_locals['temp_adj_'+self.gate_name] = self.q10factor
                for i,parameter in enumerate(self.parameters):
                    allowed_locals[parameter[0]] = self.parameters[i][1]
                if kwargs.has_key('concdep'):
                    concdep = kwargs['concdep']
                    ## ca should appear as neuroML defined 'variable_name' to eval()
                    if concdep is not None:
                        allowed_locals[concdep.attrib['variable_name']] = ca
                if '?' in expr_str:
                    condition, alternatives = expr_str.split('?',1)
                    alternativeTrue, alternativeFalse = alternatives.split(':',1)
                    if eval(condition,{"__builtins__":None},allowed_locals):
                        val = eval(alternativeTrue,{"__builtins__":None},allowed_locals)
                    else:
                        val = eval(alternativeFalse,{"__builtins__":None},allowed_locals)
                else:
                    val = eval(expr_str,{"__builtins__":None},allowed_locals)
                if fn_name == 'tau': return val/self.q10factor
                else: return val

        fn.__name__ = fn_name
        setattr(self.__class__, fn.__name__, fn)

    def replace(self, text, findstr, replacestr):
        return string.join(string.split(text,findstr),replacestr)
