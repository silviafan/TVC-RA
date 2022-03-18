# Computing participation coeffieicient (PC) and within-module degree z-score (z).
# Note that the code below is tailored only to patients. 
# However, the same steps were carried out for controls. 

import teneto
from teneto import TenetoBIDS
import pandas as pd

datdir = '/data/silfan/RA/pre/bids/' 

patients = ['1253','1281','1324','1550','1551','1567','1659','2056',
            '2223','2350','2585','2613','2876','3006',
           '306','3578','3917','4716','4903','5174','5291',
           '5447','5749','6230','874','7712','8021','8712'] #tot 28

bids_filter = {'subject': patients,
               'run': [1,2],
               'task': ['joint','thumb']}

# Defining the TenetoBIDS object
tnet = teneto.TenetoBIDS(datdir, selected_pipeline='teneto-derive-temporalnetwork', bids_filter=bids_filter)


#factorizing the communities by assigning a 0–6 value to 400 nodes based on the community they belong to. 
#obtaining an array of all nodes with unique numbers (400,) to feed into teneto PC and z

uniquenet = pd.read_csv('/data/silfan/RA/pre/bids/derivatives/teneto-make-parcellation/sub-306/func/sub-306_run-1_task-joint_timeseries.tsv', sep='\t')

uniquenet.rename(columns={'Unnamed: 0':'networks'}, inplace=True)
uniquenet['networks'] = uniquenet['networks'].str.replace('\d+', '') 
uniquenet['networks'] = uniquenet['networks'].str.rstrip('_') 
uniquenet['networks'] = uniquenet['networks'].str.replace('_LH','') 
uniquenet['networks'] = uniquenet['networks'].str.replace('_RH','') 
uniquenet['networks'] = uniquenet['networks'].str.replace('Networks_','') 
uniquenet['networks'] = uniquenet['networks'].str.split('_').str[0] 

uniquenet = uniquenet[['networks']] 

codes = pd.factorize(uniquenet['networks']) #assigne a unique int to networks with the same name

##calculating PC and z

#setting parameters to compute PC
PC_params = {'communities': codes,
                    'removeneg': True} #only positive edges were included

tnet.run('temporal_participation_coeff', PC_params, exist_ok=True)

#setting parameters to compute z 
z_score_params = {'communities': codes,
                     'calc':'module_degree_zscore',
                    'ignorediagonal': True} 

tnet.run('temporal_degree_centrality', z_score_params, exist_ok=True)