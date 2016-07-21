"""params.py: 

Parameters used in this model

These parameters are from paper Miller et. al. "The stability of CaMKII
switch"

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

run_time = 1 * 24 * 60 * 60

N_CaMK = 20  # range 4 - 20 (this is ring) a holoenzyme = 2 rings
N_PP1 =  20  # N_CaMK

#volume = 5e-19  # m^3
volume = 1e-20  # m^3

conc_i1_free = 0.1e-3

act_CaN = 1.0
act_PKA = 1.0
K_M = 0.5e-3     # Michaelis constant of protein phosphatase
K_H1 = 0.7e-3    # Hill coefficientfor Ca++ activation of CaMKII
K_H2 = 0.3e-3
n_H1 = 3         # 
k_1 = 1.5
k_2 = 10.0
k_3 = 100e3
k_4 = 0.1
K_I = 1e-6

turnover_rate_holoenzyme = 1/(1200)

# calcium
resting_ca_conc = 0.1e-3
ca_pulse = 0.18e-3 #resting_ca_conc # 1e-3

v_1 = 1.268e-5
v_2 = 4.36e-3

conc_i1p_free = 2.8e-3
phosphatase_inhibit = 280.0 
vi = phosphatase_inhibit
