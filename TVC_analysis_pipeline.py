#This script contains all TVC analyses prior to computing participation coefficient and within-module degree z-score
#That is parcellation, denoising, and deriving TVC estimates
#Note that the code below is tailored only to patients. 
#However, the same steps were carried out for controls. 

import teneto
from teneto import TenetoBIDS

datdir = '/data/silfan/RA/pre/bids/' 

patients = ['1253','1281','1324','1550','1551','1567','1659','2056',
            '2223','2350','2585','2613','2876','3006',
           '306','3578','3917','4716','4903','5174','5291',
           '5447','5749','6230','874','7712','8021','8712'] #tot 28

bids_filter = {'subject': patients,
               'run': [1,2],
               'task': ['joint','thumb']}

# Defining the TenetoBIDS object
tnet=TenetoBIDS(datdir, selected_pipeline='fMRIPrep', bids_filter=bids_filter, exist_ok=True)


# Setting parameters for parcellation
parcellation_params = {'atlas': 'Schaefer2018',
                       'atlas_desc': '400Parcels7Networks',
                       'parc_params': {'detrend': True}}

tnet.run('make_parcellation', parcellation_params, exist_ok=True)

# Setting parameters confound regression
remove_params = {'confound_selection': ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                                        'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1', 'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
                                        'a_comp_cor_04', 'a_comp_cor_05', 'framewise_displacement', 'white_matter', 'csf'] }

tnet.run('remove_confounds', remove_params, exist_ok=True)

# Setting parameters for additional data cleaning: removing runs with FD>0.5
exclude_params = {'confound_name': 'framewise_displacement',
                   'exclusion_criteria': '>0.5'}

tnet.run('exclude_runs', exclude_params, exist_ok=True) 

# Setting parameters to derive TVC estimates via the application of the jackknife correlation method
derive_params = {'params': {'method': 'jackknife',
                            'postpro': 'standardize'}}

tnet.run('derive_temporalnetwork', derive_params, exist_ok=True)