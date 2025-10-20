import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_list):
        self.a = 1
        self.file_list = file_list
        self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str, 'Patient_subID': str})

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch'])
        patient_id_list = np.asarray(c['Patient_ID'])
        patient_subid_list = np.asarray(c['Patient_subID'])
        random_num_list = np.asarray(c['random_num'])
        noise_file_list = np.asarray(c['noise_file'])
        ground_truth_file_list = np.asarray(c['ground_truth_file']) 
        
        return batch_list, patient_id_list, patient_subid_list, random_num_list, noise_file_list, ground_truth_file_list
