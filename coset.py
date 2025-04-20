# Sample code for comparing coset probabilities under exact and approximate contractions

import numpy as np
from qecsim import paulitools as pt
from qecsim.models.generic import BiasedDepolarizingErrorModel, PhaseFlipErrorModel, BitFlipErrorModel, BitPhaseFlipErrorModel,DepolarizingErrorModel
# from qsu import qsu
from qsdxzzx.planarxz import PlanarXZCode, PlanarXZMPSDecoder
from joblib import Parallel, delayed, dump
import pandas as pd

def coset_compare(error_probability,my_error_model = PhaseFlipErrorModel()):
    my_code = PlanarXZCode(5,5)
    
    # my_error_model = PhaseFlipErrorModel()
    # Initialise decoders
    decode_1 = PlanarXZMPSDecoder() #Exact
    decode_2 = PlanarXZMPSDecoder(chi=2, mode='c') #chi=2 by column
    decode_3 = PlanarXZMPSDecoder(chi=2, mode='r') #chi=2 by row

    # generate errors on the code
    error = my_error_model.generate(my_code, error_probability)

    # Finding sample recovery procedures
    syndrome = pt.bsp(error, my_code.stabilizers.T)
    any_recovery = decode_1.sample_recovery(my_code, syndrome)

    # Calculate coset probabilities with the above decoders
    prob_dist = my_error_model.probability_distribution(error_probability)
    c_prob=np.array([*decode_1._coset_probabilities(prob_dist, any_recovery)[0]])
    c_prob2=np.array([*decode_2._coset_probabilities(prob_dist, any_recovery)[0]])
    c_prob3=np.array([*decode_3._coset_probabilities(prob_dist, any_recovery)[0]])
    # print(error_probability)
    # print(c_prob)
    # print(c_prob2)
    # print(c_prob3)
    max_exact=np.max(c_prob)
    max_chi2=np.max(c_prob2)
    max_chi2r=np.max(c_prob3)
    exact_index = c_prob==max_exact
    chi2_index = c_prob2 == max_chi2
    chi2r_index = c_prob3 == max_chi2r

    return {'Error_Probability':error_probability, 
            'Same_max_c':sum(exact_index*chi2_index),  # Proportion of identifying the same recovery procedure
            'Percentage_difference_c':float(sum(np.abs(c_prob-c_prob2))/sum(c_prob)), # Normalized L1
            'Same_max_r':sum(exact_index*chi2r_index), 
            'Percentage_difference_r':float(sum(np.abs(c_prob-c_prob3))/sum(c_prob))}

data = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
delayed(coset_compare)(error,BiasedDepolarizingErrorModel(0.5,'Z')) for error in [0.2,0.25,0.3,0.35,0.4,0.45,0.49] for i in range(1000))
df = pd.DataFrame(data)
grouped_data = df.groupby('Error_Probability')
# print(df.head())
dump(data,'cosetd_5x5.joblib')
averages = grouped_data.mean()

print(averages)