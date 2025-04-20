# Sample code for threshold estimation

import numpy as np

from qecsim.models.generic import BiasedDepolarizingErrorModel, PhaseFlipErrorModel
from qecsim.app import run, run_once
from qsdxzzx.planarxz import PlanarXZCode, PlanarXZMPSDecoder

from joblib import Parallel, delayed, dump

# generate code with the desired sizes
sizes_1000 = [(n, 157*n) for n in range(3,8,2)]
codes_1000 = [PlanarXZCode(*size) for size in sizes_1000][::-1]
# Initialise an error model 
error_model_1000 = BiasedDepolarizingErrorModel(1000,"Z")
# Initialise a decoder
decoder = PlanarXZMPSDecoder(2, mode = 'c')
# Choosing a range of error probabilities
error_probability_min, error_probability_max = 0.43, 0.49
error_probabilities = np.linspace(error_probability_min, error_probability_max, 7)[1:]

# Number of runs per configuration, the larger the better
max_runs = 10
# print run parameters
print('Codes:', [code.label for code in codes_1000])
print('Error model:', error_model_1000.label)
print('Decoder:', decoder.label)
print('Error probabilities:', error_probabilities)
print('Maximum runs:', max_runs)
def run_print(code, error_model_1000,  decoder, error_probability, max_runs):
    a = run(code, error_model_1000,  decoder, error_probability, max_runs)
    with open('bb_1000.csv', 'a') as file:
        v_sv = [str(value) for key,value in a.items()]
        tne = ','.join(v_sv)
        file.write(tne+'\n')
    return a    

# Simulating the decoding process and collect data about physical and logical error rate
data = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
delayed(run_print)(code, error_model_1000,  decoder, error_probability, max_runs=max_runs)
        for code in codes_1000 for error_probability in error_probabilities)
# csv_try = [[value  for dat in data] for key,value in dat.items()]
# titles=[str(key) for key,value in data[0].items()]
# print(', '.join(titles))
dump(data,'hpc.joblib')

for dat in data:
    v_sv = [str(value) for key,value in dat.items()]
    print(', '.join(v_sv))

