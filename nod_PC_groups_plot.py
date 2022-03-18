## Plotting the degree of change in PC (integration) over time at the nodal level across groups for painful stimulation at the joint
## Note that code for plotting the degree of change in within-module degree z-score under the same conditions required lines 37,40 to be changed into  
## tnet_controls = TenetoBIDS(datdir, selected_pipeline='teneto-temporal-degree-centrality', bids_filter=bids_filter_controls)
## tnet_patients = TenetoBIDS(datdir, selected_pipeline='teneto-temporal-degree-centrality', bids_filter=bids_filter_patients)
## Also, different ylim and plot colors were chosen. 

import teneto
from teneto import TenetoBIDS
import numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

datdir = '/data/silfan/RA/pre/bids/' 

controls = ['1279','1616','2047','2187','2298','2565','2612','351',
            '354','4872','5158','5360','5446','5633',
           '5796','5797','5801','5819','5820','6772','7386',
           '7433'] #tot 22

bids_filter_controls = {'subject': controls,
               'run': [1,2],
               'task': ['joint']}

patients = ['1253','1281','1324','1550','1551','1567','1659','2056',
            '2223','2350','2585','2613','2876','3006',
           '306','3578','3917','4716','4903','5174','5291',
           '5447','5749','6230','874','7712','8021','8712'] #tot 28

bids_filter_patients = {'subject': patients,
               'run': [1,2],
               'task': ['joint']}

# Defining the TenetoBIDS object for controls
tnet_controls = TenetoBIDS(datdir, selected_pipeline='teneto-temporal-participation-coeff', bids_filter=bids_filter_controls)

# Defining the TenetoBIDS object for patients
tnet_patients = TenetoBIDS(datdir, selected_pipeline='teneto-temporal-participation-coeff', bids_filter=bids_filter_patients)

data_controls = tnet_controls.load_data()
data_patients = tnet_patients.load_data()

## Data preparation for plotting PC of every selected node over time (-2TR,-1TR,onset,+1TR,+2TR,+3TR)

# Painful stimulation to joint of HC
nodesHC = []

for key in data_controls.keys():
    file_ent = tnet_controls.BIDSLayout.parse_file_entities(str(key))
    rootdir = "/data/silfan/RA/pre/bids/sub-" + str(key.split('_')[0].split('-')[1]) + "/func/sub-" + str(key.split('_')[0].split('-')[1]) + "_task-" + file_ent['task'] + "_run-" + str(file_ent['run']) + "_events.tsv"
    if "task-joint" in rootdir:
        events = pd.read_csv(rootdir, sep=';')
        events = events.loc[:, ~events.columns.str.contains('^Unnamed')]

        events['onsetTR'] = events['onset'].div(3).round(0) 
        events['onsetTR'] = list(events['onsetTR'].astype('int64'))
        events['onsetTR'] = events['onsetTR'].astype(str)

        pain_df = events[events['trial_type'] == 'pain'] 

        PC_df = data_controls[key]
        PC_df = pd.DataFrame(PC_df)

        PC_df_ROIs = PC_df.iloc[[97,307,33,235,176,383],:]
        PC_df_ROIs = PC_df_ROIs.rename(index={97: "L AIns", 307: "R AIns", 33: "L PIns", 235: "R PIns", 176: "L ACgG", 383: "R ACgG"}) 
        PC_df_ROIs.index.name = 'ROIs'

        data_pre2TR = []
        data_pre1TR = []
        data_onset = []
        data_post1TR = []
        data_post2TR = []
        data_post3TR = []

        for column in PC_df_ROIs:
            if column in pain_df['onsetTR'].values:

                    pre2TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)-2:PC_df_ROIs.columns.get_loc(column)-1]], axis=1) #select 2TRs pre pain onset
                    pre1TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)-1:PC_df_ROIs.columns.get_loc(column)]], axis=1) #select 2TR pre pain onset
                    onset = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column):PC_df_ROIs.columns.get_loc(column)+1]], axis=1) #select TR of pain onset
                    post1TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)+1:PC_df_ROIs.columns.get_loc(column)+2]], axis=1) #select 1TR afer pain onset
                    post2TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)+2:PC_df_ROIs.columns.get_loc(column)+3]], axis=1) #select 2TR afer pain onset
                    post3TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)+3:PC_df_ROIs.columns.get_loc(column)+4]], axis=1) #select 3TR afer pain onset

                    data_pre2TR.append(pre2TR)
                    data_pre1TR.append(pre1TR)
                    data_onset.append(onset)
                    data_post1TR.append(post1TR)
                    data_post2TR.append(post2TR)
                    data_post3TR.append(post3TR)

        data_pre2TR = pd.concat(data_pre2TR, axis=1) 
        data_pre1TR = pd.concat(data_pre1TR, axis=1) 
        data_onset = pd.concat(data_onset, axis=1) 
        data_post1TR = pd.concat(data_post1TR, axis=1) 
        data_post2TR = pd.concat(data_post2TR, axis=1) 
        data_post3TR = pd.concat(data_post3TR, axis=1) 

        #mean for time frame
        data_pre2TR_mean = data_pre2TR.T.mean()
        data_pre1TR_mean = data_pre1TR.T.mean()
        data_onset_mean = data_onset.T.mean()
        data_post1TR_mean = data_post1TR.T.mean()
        data_post2TR_mean = data_post2TR.T.mean()
        data_post3TR_mean = data_post3TR.T.mean()

        frames = [data_pre2TR_mean, data_pre1TR_mean, data_onset_mean, data_post1TR_mean, data_post2TR_mean, data_post3TR_mean]
        nodes = pd.concat(frames, axis=1)

        nodes.columns = ['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR'] 

        nodesHC.append(nodes)

nodesHC_col = pd.concat(nodesHC, axis=1)
nodesHC_mean = nodesHC_col.groupby(by=nodesHC_col.columns, axis=1).mean()
nodesHC_mean = nodesHC_mean[['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']] 

# Painful stimulation to joint of RA patients
nodesRA = []

for key in data_patients.keys():
    file_ent = tnet_patients.BIDSLayout.parse_file_entities(str(key))
    rootdir = "/data/silfan/RA/pre/bids/sub-" + str(key.split('_')[0].split('-')[1]) + "/func/sub-" + str(key.split('_')[0].split('-')[1]) + "_task-" + file_ent['task'] + "_run-" + str(file_ent['run']) + "_events.tsv"
    if "task-joint" in rootdir:
        events = pd.read_csv(rootdir, sep=';')
        events = events.loc[:, ~events.columns.str.contains('^Unnamed')]

        events['onsetTR'] = events['onset'].div(3).round(0) 
        events['onsetTR'] = list(events['onsetTR'].astype('int64'))
        events['onsetTR'] = events['onsetTR'].astype(str)

        pain_df = events[events['trial_type'] == 'pain'] 

        PC_df = data_patients[key]
        PC_df = pd.DataFrame(PC_df)

        PC_df_ROIs = PC_df.iloc[[97,307,33,235,176,383],:]
        PC_df_ROIs = PC_df_ROIs.rename(index={97: "L AIns", 307: "R AIns", 33: "L PIns", 235: "R PIns", 176: "L ACgG", 383: "R ACgG"}) 
        PC_df_ROIs.index.name = 'ROIs' 

        data_pre2TR = []
        data_pre1TR = []
        data_onset = []
        data_post1TR = []
        data_post2TR = []
        data_post3TR = []

        for column in PC_df_ROIs:
            if column in pain_df['onsetTR'].values:

                    pre2TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)-2:PC_df_ROIs.columns.get_loc(column)-1]], axis=1) #select 2TRs pre pain onset
                    pre1TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)-1:PC_df_ROIs.columns.get_loc(column)]], axis=1) #select 2TR pre pain onset
                    onset = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column):PC_df_ROIs.columns.get_loc(column)+1]], axis=1) #select TR of pain onset
                    post1TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)+1:PC_df_ROIs.columns.get_loc(column)+2]], axis=1) #select 1TR afer pain onset
                    post2TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)+2:PC_df_ROIs.columns.get_loc(column)+3]], axis=1) #select 2TR afer pain onset
                    post3TR = pd.concat([PC_df_ROIs.iloc[:, PC_df_ROIs.columns.get_loc(column)+3:PC_df_ROIs.columns.get_loc(column)+4]], axis=1) #select 3TR afer pain onset

                    data_pre2TR.append(pre2TR)
                    data_pre1TR.append(pre1TR)
                    data_onset.append(onset)
                    data_post1TR.append(post1TR)
                    data_post2TR.append(post2TR)
                    data_post3TR.append(post3TR)

        data_pre2TR = pd.concat(data_pre2TR, axis=1) 
        data_pre1TR = pd.concat(data_pre1TR, axis=1) 
        data_onset = pd.concat(data_onset, axis=1) 
        data_post1TR = pd.concat(data_post1TR, axis=1) 
        data_post2TR = pd.concat(data_post2TR, axis=1) 
        data_post3TR = pd.concat(data_post3TR, axis=1) 

        #mean for time frame
        data_pre2TR_mean = data_pre2TR.T.mean()
        data_pre1TR_mean = data_pre1TR.T.mean()
        data_onset_mean = data_onset.T.mean()
        data_post1TR_mean = data_post1TR.T.mean()
        data_post2TR_mean = data_post2TR.T.mean()
        data_post3TR_mean = data_post3TR.T.mean()

        frames = [data_pre2TR_mean, data_pre1TR_mean, data_onset_mean, data_post1TR_mean, data_post2TR_mean, data_post3TR_mean]
        nodes = pd.concat(frames, axis=1)

        nodes.columns = ['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR'] 

        nodesRA.append(nodes)

nodesRA_col = pd.concat(nodesRA, axis=1)
nodesRA_mean = nodesRA_col.groupby(by=nodesRA_col.columns, axis=1).mean()
nodesRA_mean = nodesRA_mean[['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']] 

## Computing standard error of the mean (sem) 
HC = numpy.concatenate([([i]*2) for i in [351,354,1279,1616,2047,2187,2298,2565,2612,4872,5158,5360,5446,5633,5796,5797,5801,5819,5820,6772,7386,7433]], axis=0) #tot=22
HC = HC.tolist()

RA = numpy.concatenate([([i]*2) for i in [306,874,1253,1281,1324,1550,1551,1567,1659,2056,2223,2350,2585,2613,2876,3006,3578,3917,4716,4903,5174,5291,5447,5749,6230,7712,8021,8712]], axis=0) #tot=28
RA = RA.tolist() 

ROIS = ['L AIns', 'R AIns', 'L PIns', 'R PIns', 'L ACgG', 'R ACgG']
timepoints = ['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']
HC_sem_dict = {}
PAT_sem_dict = {}


for ROI in ROIS:
    for time in timepoints:
        # HC
        H = nodesHC_col.loc[[ROI], [time]].T

        H = H.reset_index() 
        H = H[ROI] 
        H = H.to_frame() 
        H = H.assign(controls=HC) 
        H = H.rename(columns = {ROI:'JOINTHC'})
        colswap = ['controls', 'JOINTHC']
        H = H.reindex(columns=colswap) 

        contr = H.groupby('controls')['JOINTHC'].agg(averageJOINTHC='mean').reset_index() # mean between runs 1 and 2 of the same subject

        # RA patients
        P = nodesRA_col.loc[[ROI], [time]] 
        P = P.T 
        P = P.reset_index() 
        P = P[ROI] 
        P = P.to_frame()
        P = P.assign(patients=RA)
        P = P.rename(columns = {ROI:'JOINTPAT'})
        colswap = ['patients', 'JOINTPAT']
        P = P.reindex(columns=colswap) 

        pat = P.groupby('patients')['JOINTPAT'].agg(averageJOINTPAT='mean').reset_index() #mean between runs 1 and 2 of the same subject

        HCtime = contr.assign(time=time)
        HCtimenet = HCtime.assign(network=ROI)

        PATtime = pat.assign(time=time)
        PATtimenet = PATtime.assign(network=ROI)

        sem_HC = sem(HCtimenet['averageJOINTHC'])
        sem_PAT = sem(PATtimenet['averageJOINTPAT'])

        HC_sem_dict['{}_{}'.format(ROI,time)] = sem_HC
        PAT_sem_dict['{}_{}'.format(ROI,time)] = sem_PAT


HC_sem_df = pd.DataFrame()
HC_sem_df = HC_sem_df.append(HC_sem_dict, ignore_index=True) 

HC_sem_df = HC_sem_df.T.reset_index() 
HC_sem_df = HC_sem_df.rename(columns = {0:'sem', 'index':'nodetime'})

HC_sem_df['node'],HC_sem_df['time'] = HC_sem_df['nodetime'].str.split('_',1).str 
HC_sem_df = HC_sem_df.drop('nodetime', 1) 

nodesHC_sem = HC_sem_df.groupby("node")["sem"].apply(lambda x: pd.Series(x.values)).unstack().add_prefix('time')
nodesHC_sem = nodesHC_sem.rename(columns={'time0': 'pre2TR', 'time1': 'pre1TR', 'time2': 'onset', 'time3': 'post1TR', 'time4': 'post2TR', 'time5': 'post3TR'})
nodesHC_sem.index.names = ['ROIs']
nodesHC_sem = nodesHC_sem.reindex(["L AIns", "R AIns", "L PIns", "R PIns", "L ACgG", "R ACgG"]) 

PAT_sem_df = pd.DataFrame()
PAT_sem_df = PAT_sem_df.append(PAT_sem_dict, ignore_index=True) 

PAT_sem_df = PAT_sem_df.T.reset_index()
PAT_sem_df = PAT_sem_df.rename(columns = {0:'sem', 'index':'nodetime'})

PAT_sem_df['node'],PAT_sem_df['time'] = PAT_sem_df['nodetime'].str.split('_',1).str 
PAT_sem_df = PAT_sem_df.drop('nodetime', 1) 

nodesRA_sem = PAT_sem_df.groupby("node")["sem"].apply(lambda x: pd.Series(x.values)).unstack().add_prefix('time')
nodesRA_sem = nodesRA_sem.rename(columns={'time0': 'pre2TR', 'time1': 'pre1TR', 'time2': 'onset', 'time3': 'post1TR', 'time4': 'post2TR', 'time5': 'post3TR'})
nodesRA_sem.index.names = ['ROIs']
nodesRA_sem = nodesRA_sem.reindex(["L AIns", "R AIns", "L PIns", "R PIns", "L ACgG", "R ACgG"]) 


## Plotting all 6 nodes in the same panel

LAIns_HC = nodesHC_mean.iloc[[0]] 
LAIns_HC = LAIns_HC.rename(index={'L AIns': 'L AIns Joint HC'}) 
LAIns_PAT = nodesRA_mean.iloc[[0]] 
LAIns_PAT = LAIns_PAT.rename(index={'L AIns': 'L AIns Joint RA patients'}) 
LAIns_HCPAT_mean = pd.concat([LAIns_HC, LAIns_PAT]) 

LAIns_HC_sem = nodesHC_sem.iloc[[0]] 
LAIns_HC_sem = LAIns_HC_sem.rename(index={'L AIns': 'L AIns Joint HC'}) 
LAIns_PAT_sem = nodesRA_sem.iloc[[0]] 
LAIns_PAT_sem = LAIns_PAT_sem.rename(index={'L AIns': 'L AIns Joint RA patients'}) 

RAIns_HC = nodesHC_mean.iloc[[1]] 
RAIns_HC = RAIns_HC.rename(index={'R AIns': 'R AIns Joint HC'}) 
RAIns_PAT = nodesRA_mean.iloc[[1]] 
RAIns_PAT = RAIns_PAT.rename(index={'R AIns': 'R AIns Joint RA patients'}) 
RAIns_HCPAT_mean = pd.concat([RAIns_HC, RAIns_PAT]) 

RAIns_HC_sem = nodesHC_sem.iloc[[1]] 
RAIns_HC_sem = RAIns_HC_sem.rename(index={'R AIns': 'R AIns Joint HC'}) 
RAIns_PAT_sem = nodesRA_sem.iloc[[1]] 
RAIns_PAT_sem = RAIns_PAT_sem.rename(index={'R AIns': 'R AIns Joint RA patients'}) 

LPIns_HC = nodesHC_mean.iloc[[2]] 
LPIns_HC = LPIns_HC.rename(index={'L PIns': 'L PIns Joint HC'}) 
LPIns_PAT = nodesRA_mean.iloc[[2]] 
LPIns_PAT = LPIns_PAT.rename(index={'L PIns': 'L PIns Joint RA patients'}) 
LPIns_HCPAT_mean = pd.concat([LPIns_HC, LPIns_PAT]) 

LPIns_HC_sem = nodesHC_sem.iloc[[2]] 
LPIns_HC_sem = LPIns_HC_sem.rename(index={'L PIns': 'L PIns Joint HC'}) 
LPIns_PAT_sem = nodesRA_sem.iloc[[2]] 
LPIns_PAT_sem = LPIns_PAT_sem.rename(index={'L PIns': 'L PIns Joint RA patients'}) 

RPIns_HC = nodesHC_mean.iloc[[3]] 
RPIns_HC = RPIns_HC.rename(index={'R PIns': 'R PIns Joint HC'}) 
RPIns_PAT = nodesRA_mean.iloc[[3]] 
RPIns_PAT = RPIns_PAT.rename(index={'R PIns': 'R PIns Joint RA patients'}) 
RPIns_HCPAT_mean = pd.concat([RPIns_HC, RPIns_PAT]) 

RPIns_HC_sem = nodesHC_sem.iloc[[3]] 
RPIns_HC_sem = RPIns_HC_sem.rename(index={'R PIns': 'R PIns Joint HC'}) 
RPIns_PAT_sem = nodesRA_sem.iloc[[3]] 
RPIns_PAT_sem = RPIns_PAT_sem.rename(index={'R PIns': 'R PIns Joint RA patients'}) 

LACgG_HC = nodesHC_mean.iloc[[4]] 
LACgG_HC = LACgG_HC.rename(index={'L ACgG': 'L ACgG Joint HC'}) 
LACgG_PAT = nodesRA_mean.iloc[[4]] 
LACgG_PAT = LACgG_PAT.rename(index={'L ACgG': 'L ACgG Joint RA patients'}) 
LACgG_HCPAT_mean = pd.concat([LACgG_HC, LACgG_PAT])

LACgG_HC_sem = nodesHC_sem.iloc[[4]] 
LACgG_HC_sem = LACgG_HC_sem.rename(index={'L ACgG': 'L ACgG Joint HC'})
LACgG_PAT_sem = nodesRA_sem.iloc[[4]] 
LACgG_PAT_sem = LACgG_PAT_sem.rename(index={'L ACgG': 'L ACgG Joint RA patients'}) 

RACgG_HC = nodesHC_mean.iloc[[5]] 
RACgG_HC = RACgG_HC.rename(index={'R ACgG': 'R ACgG Joint HC'}) 
RACgG_PAT = nodesRA_mean.iloc[[5]] 
RACgG_PAT = RACgG_PAT.rename(index={'R ACgG': 'R ACgG Joint RA patients'})
RACgG_HCPAT_mean = pd.concat([RACgG_HC, RACgG_PAT]) 

RACgG_HC_sem = nodesHC_sem.iloc[[5]] 
RACgG_HC_sem = RACgG_HC_sem.rename(index={'R ACgG': 'R ACgG Joint HC'})
RACgG_PAT_sem = nodesRA_sem.iloc[[5]] 
RACgG_PAT_sem = RACgG_PAT_sem.rename(index={'R ACgG': 'R ACgG Joint RA patients'}) 

fig = plt.figure(figsize=(5,655))

ax1 = fig.add_subplot(611, ylim=(0.76, 0.80))
ax1.set_title('L AIns', size=20)
ax1.set_xticklabels([])
ax1.tick_params(axis='both', labelsize=20)
ax1.plot(LAIns_HCPAT_mean.T['L AIns Joint HC'], color='lightgreen', linewidth=4)
ax1.fill_between(LAIns_HCPAT_mean.T.index, LAIns_HCPAT_mean.T['L AIns Joint HC']-LAIns_HC_sem.T['L AIns Joint HC'], LAIns_HCPAT_mean.T['L AIns Joint HC']+LAIns_HC_sem.T['L AIns Joint HC'], alpha=0.20, color='lightgreen')
ax1.plot(LAIns_HCPAT_mean.T['L AIns Joint RA patients'], color='lightgreen', linestyle='dashed', linewidth=4)
ax1.fill_between(LAIns_HCPAT_mean.T.index, LAIns_HCPAT_mean.T['L AIns Joint RA patients']-LAIns_PAT_sem.T['L AIns Joint RA patients'], LAIns_HCPAT_mean.T['L AIns Joint RA patients']+LAIns_PAT_sem.T['L AIns Joint RA patients'], alpha=0.20, color='lightgreen')
ax1.yaxis.set_major_locator(MaxNLocator(3))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2 = fig.add_subplot(612, ylim=(0.76, 0.80))
ax2.set_xticklabels([])
ax2.tick_params(axis='both', labelsize=20)
ax2.set_title('R AIns', size=20)
ax2.plot(RAIns_HCPAT_mean.T['R AIns Joint HC'], color='rebeccapurple', linewidth=4)
ax2.fill_between(RAIns_HCPAT_mean.T.index, RAIns_HCPAT_mean.T['R AIns Joint HC']-RAIns_HC_sem.T['R AIns Joint HC'], RAIns_HCPAT_mean.T['R AIns Joint HC']+RAIns_HC_sem.T['R AIns Joint HC'], alpha=0.20, color='rebeccapurple')
ax2.plot(RAIns_HCPAT_mean.T['R AIns Joint RA patients'], color='rebeccapurple', linestyle='dashed', linewidth=4)
ax2.fill_between(RAIns_HCPAT_mean.T.index, RAIns_HCPAT_mean.T['R AIns Joint RA patients']-RAIns_PAT_sem.T['R AIns Joint RA patients'], RAIns_HCPAT_mean.T['R AIns Joint RA patients']+RAIns_PAT_sem.T['R AIns Joint RA patients'], alpha=0.20, color='rebeccapurple')
ax2.yaxis.set_major_locator(MaxNLocator(3))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax3 = fig.add_subplot(613, ylim=(0.76, 0.80))
ax3.set_title('L PIns', size=20)
ax3.set_xticklabels([])
ax3.tick_params(axis='both', labelsize=20)
ax3.plot(LPIns_HCPAT_mean.T['L PIns Joint HC'], color='skyblue', linewidth=4)
ax3.fill_between(LPIns_HCPAT_mean.T.index, LPIns_HCPAT_mean.T['L PIns Joint HC']-LPIns_HC_sem.T['L PIns Joint HC'], LPIns_HCPAT_mean.T['L PIns Joint HC']+LPIns_HC_sem.T['L PIns Joint HC'], alpha=0.20, color='skyblue')
ax3.plot(LPIns_HCPAT_mean.T['L PIns Joint RA patients'], color='skyblue', linestyle='dashed', linewidth=4)
ax3.fill_between(LPIns_HCPAT_mean.T.index, LPIns_HCPAT_mean.T['L PIns Joint RA patients']-LPIns_PAT_sem.T['L PIns Joint RA patients'], LPIns_HCPAT_mean.T['L PIns Joint RA patients']+LPIns_PAT_sem.T['L PIns Joint RA patients'], alpha=0.20, color='skyblue')
ax3.yaxis.set_major_locator(MaxNLocator(3))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax4 = fig.add_subplot(614, ylim=(0.76, 0.80))
ax4.set_xticklabels([])
ax4.tick_params(axis='both', labelsize=20)
ax4.set_title('R PIns', size=20)
ax4.plot(RPIns_HCPAT_mean.T['R PIns Joint HC'], color='darkorange', linewidth=4)
ax4.fill_between(RPIns_HCPAT_mean.T.index, RPIns_HCPAT_mean.T['R PIns Joint HC']-RPIns_HC_sem.T['R PIns Joint HC'], RPIns_HCPAT_mean.T['R PIns Joint HC']+RPIns_HC_sem.T['R PIns Joint HC'], alpha=0.20, color='darkorange')
ax4.plot(RPIns_HCPAT_mean.T['R PIns Joint RA patients'], color='darkorange', linestyle='dashed', linewidth=4)
ax4.fill_between(RPIns_HCPAT_mean.T.index, RPIns_HCPAT_mean.T['R PIns Joint RA patients']-RPIns_PAT_sem.T['R PIns Joint RA patients'], RPIns_HCPAT_mean.T['R PIns Joint RA patients']+RPIns_PAT_sem.T['R PIns Joint RA patients'], alpha=0.20, color='darkorange')
ax4.yaxis.set_major_locator(MaxNLocator(3))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax5 = fig.add_subplot(615, ylim=(0.76, 0.80))
ax5.set_title('L ACgG', size=20)
ax5.set_xticklabels([])
ax5.tick_params(axis='both', labelsize=20)
ax5.plot(LACgG_HCPAT_mean.T['L ACgG Joint HC'], color='red', linewidth=4)
ax5.fill_between(LACgG_HCPAT_mean.T.index, LACgG_HCPAT_mean.T['L ACgG Joint HC']-LACgG_HC_sem.T['L ACgG Joint HC'], LACgG_HCPAT_mean.T['L ACgG Joint HC']+LACgG_HC_sem.T['L ACgG Joint HC'], alpha=0.20, color='red')
ax5.plot(LACgG_HCPAT_mean.T['L ACgG Joint RA patients'], color='red', linestyle='dashed', linewidth=4)
ax5.fill_between(LACgG_HCPAT_mean.T.index, LACgG_HCPAT_mean.T['L ACgG Joint RA patients']-LACgG_PAT_sem.T['L ACgG Joint RA patients'], LACgG_HCPAT_mean.T['L ACgG Joint RA patients']+LACgG_PAT_sem.T['L ACgG Joint RA patients'], alpha=0.20, color='red')
ax5.yaxis.set_major_locator(MaxNLocator(3))
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax6 = fig.add_subplot(616, ylim=(0.76, 0.80))
ax6.set_title('R ACgG', size=20)
timepoints = ['-2','-1','0','1', '2', '3']
ax6.set_xticklabels(timepoints)
ax6.tick_params(axis='both', labelsize=20)
ax6.plot(RACgG_HCPAT_mean.T['R ACgG Joint HC'], color='chocolate', linewidth=4)
ax6.fill_between(RACgG_HCPAT_mean.T.index, RACgG_HCPAT_mean.T['R ACgG Joint HC']-RACgG_HC_sem.T['R ACgG Joint HC'], RACgG_HCPAT_mean.T['R ACgG Joint HC']+RACgG_HC_sem.T['R ACgG Joint HC'], alpha=0.20, color='chocolate')
ax6.plot(RACgG_HCPAT_mean.T['R ACgG Joint RA patients'], color='chocolate', linestyle='dashed', linewidth=4)
ax6.fill_between(RACgG_HCPAT_mean.T.index, RACgG_HCPAT_mean.T['R ACgG Joint RA patients']-RACgG_PAT_sem.T['R ACgG Joint RA patients'], RACgG_HCPAT_mean.T['R ACgG Joint RA patients']+RACgG_PAT_sem.T['R ACgG Joint RA patients'], alpha=0.20, color='chocolate')
ax6.yaxis.set_major_locator(MaxNLocator(3))
ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout(pad=2, h_pad=250)