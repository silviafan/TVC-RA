## Plotting the degree of change in PC (integration) over time at the community level across groups for painful stimulation at the joint
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

uniquenet = pd.read_csv('/data/silfan/RA/pre/bids/derivatives/teneto-make-parcellation/sub-306/func/sub-306_run-1_task-joint_timeseries.tsv', sep='\t')

uniquenet.rename(columns={'Unnamed: 0':'networks'}, inplace=True) 
uniquenet['networks'] = uniquenet['networks'].str.replace('\d+', '') 
uniquenet['networks'] = uniquenet['networks'].str.rstrip('_') 
uniquenet['networks'] = uniquenet['networks'].str.replace('_LH','') 
uniquenet['networks'] = uniquenet['networks'].str.replace('_RH','') 
uniquenet['networks'] = uniquenet['networks'].str.replace('Networks_','') 
uniquenet['networks'] = uniquenet['networks'].str.split('_').str[0] 

## Data preparation for plotting median PC of all nodes within each community over time (-2TR,-1TR,onset,+1TR,+2TR,+3TR)

# Painful stimulation to joint of HC
netHC = []

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
        PC_df.columns = PC_df.columns.map(str)
        PC_df = PC_df.join(uniquenet[['networks']]) 

        PC_df = PC_df.groupby('networks').median()

        data_pre2TR = []
        data_pre1TR = []
        data_onset = []
        data_post1TR = []
        data_post2TR = []
        data_post3TR = []

        for column in PC_df:
            if column in pain_df['onsetTR'].values:

                    pre2TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)-2:PC_df.columns.get_loc(column)-1]], axis=1) #select 2TRs pre pain onset
                    pre1TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)-1:PC_df.columns.get_loc(column)]], axis=1) #select 2TR pre pain onset
                    onset = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column):PC_df.columns.get_loc(column)+1]], axis=1) #select TR of pain onset
                    post1TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)+1:PC_df.columns.get_loc(column)+2]], axis=1) #select 1TR afer pain onset
                    post2TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)+2:PC_df.columns.get_loc(column)+3]], axis=1) #select 2TR afer pain onset
                    post3TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)+3:PC_df.columns.get_loc(column)+4]], axis=1) #select 3TR afer pain onset

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
        net = pd.concat(frames, axis=1)

        net.columns = ['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR'] 

        netHC.append(net)

netHC_col = pd.concat(netHC, axis=1)
netHC_mean = netHC_col.groupby(by=netHC_col.columns, axis=1).mean()
netHC_mean = netHC_mean[['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']] 

# Painful stimulation to joint of RA patients
netRA = []

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
        PC_df.columns = PC_df.columns.map(str)
        PC_df = PC_df.join(uniquenet[['networks']]) 

        PC_df = PC_df.groupby('networks').median()

        data_pre2TR = []
        data_pre1TR = []
        data_onset = []
        data_post1TR = []
        data_post2TR = []
        data_post3TR = []

        for column in PC_df:
            if column in pain_df['onsetTR'].values:

                    pre2TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)-2:PC_df.columns.get_loc(column)-1]], axis=1) #select 2TRs pre pain onset
                    pre1TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)-1:PC_df.columns.get_loc(column)]], axis=1) #select 2TR pre pain onset
                    onset = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column):PC_df.columns.get_loc(column)+1]], axis=1) #select TR of pain onset
                    post1TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)+1:PC_df.columns.get_loc(column)+2]], axis=1) #select 1TR afer pain onset
                    post2TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)+2:PC_df.columns.get_loc(column)+3]], axis=1) #select 2TR afer pain onset
                    post3TR = pd.concat([PC_df.iloc[:, PC_df.columns.get_loc(column)+3:PC_df.columns.get_loc(column)+4]], axis=1) #select 3TR afer pain onset

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
        net = pd.concat(frames, axis=1)

        net.columns = ['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR'] 

        netRA.append(net)

netRA_col = pd.concat(netRA, axis=1)
netRA_mean = netRA_col.groupby(by=netRA_col.columns, axis=1).mean()
netRA_mean = netRA_mean[['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']] 

## Computing standard error of the mean (sem) 
HC = numpy.concatenate([([i]*2) for i in [351,354,1279,1616,2047,2187,2298,2565,2612,4872,5158,5360,5446,5633,5796,5797,5801,5819,5820,6772,7386,7433]], axis=0) #tot=22
HC = HC.tolist() 

RA = numpy.concatenate([([i]*2) for i in [306,874,1253,1281,1324,1550,1551,1567,1659,2056,2223,2350,2585,2613,2876,3006,3578,3917,4716,4903,5174,5291,5447,5749,6230,7712,8021,8712]], axis=0) #tot=28
RA = RA.tolist() 

networks = ['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']
timepoints = ['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']
HC_sem_dict = {}
PAT_sem_dict = {}

for network in networks:
    for time in timepoints:
        #HC
        H = netHC_col.loc[[network], [time]].T

        H = H.reset_index() 
        H = H[network] 
        H = H.to_frame() 
        H = H.assign(controls=HC) 
        H = H.rename(columns = {network:'JOINTHC'})
        colswap = ['controls', 'JOINTHC']
        H = H.reindex(columns=colswap) 

        contr = H.groupby('controls')['JOINTHC'].agg(averageJOINTHC='mean').reset_index() # mean between runs 1 and 2 of the same subject

        #RA patients
        P = netRA_col.loc[[network], [time]] 
        P = P.T 
        P = P.reset_index() 
        P = P[network] 
        P = P.to_frame()
        P = P.assign(patients=RA)
        P = P.rename(columns = {network:'JOINTPAT'})
        colswap = ['patients', 'JOINTPAT']
        P = P.reindex(columns=colswap) 

        pat = P.groupby('patients')['JOINTPAT'].agg(averageJOINTPAT='mean').reset_index() # mean between runs 1 and 2 of the same subject

        HCtime = contr.assign(time=time)
        HCtimenet = HCtime.assign(network=network)

        PATtime = pat.assign(time=time)
        PATtimenet = PATtime.assign(network=network)

        sem_HC = sem(HCtimenet['averageJOINTHC'])
        sem_PAT = sem(PATtimenet['averageJOINTPAT'])

        HC_sem_dict['{}_{}'.format(network,time)] = sem_HC
        PAT_sem_dict['{}_{}'.format(network,time)] = sem_PAT

HC_sem_df = pd.DataFrame()
HC_sem_df = HC_sem_df.append(HC_sem_dict, ignore_index=True) 

HC_sem_df = HC_sem_df.T.reset_index() 
HC_sem_df = HC_sem_df.rename(columns = {0:'sem', 'index':'networktime'})

HC_sem_df['network'],HC_sem_df['time'] = HC_sem_df['networktime'].str.split('_',1).str 
HC_sem_df = HC_sem_df.drop('networktime', 1) 

netHC_sem = HC_sem_df.groupby("network")["sem"].apply(lambda x: pd.Series(x.values)).unstack().add_prefix('time')
netHC_sem = netHC_sem.rename(columns={'time0': 'pre2TR', 'time1': 'pre1TR', 'time2': 'onset', 'time3': 'post1TR', 'time4': 'post2TR', 'time5': 'post3TR'})
netHC_sem.index.names = ['networks']
netHC_sem = netHC_sem.reindex(['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']) 

PAT_sem_df = pd.DataFrame()
PAT_sem_df = PAT_sem_df.append(PAT_sem_dict, ignore_index=True) 

PAT_sem_df = PAT_sem_df.T.reset_index() 
PAT_sem_df = PAT_sem_df.rename(columns = {0:'sem', 'index':'networktime'})

PAT_sem_df['network'],PAT_sem_df['time'] = PAT_sem_df['networktime'].str.split('_',1).str 
PAT_sem_df = PAT_sem_df.drop('networktime', 1) 

netRA_sem = PAT_sem_df.groupby("network")["sem"].apply(lambda x: pd.Series(x.values)).unstack().add_prefix('time')
netRA_sem = netRA_sem.rename(columns={'time0': 'pre2TR', 'time1': 'pre1TR', 'time2': 'onset', 'time3': 'post1TR', 'time4': 'post2TR', 'time5': 'post3TR'})
netRA_sem.index.names = ['networks']
netRA_sem = netRA_sem.reindex(['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']) 

## Plotting all 7 networks in the same panel

Cont_HC = netHC_mean.iloc[[0]] 
Cont_HC = Cont_HC.rename(index={'Cont': 'Cont Joint HC'}) 
Cont_PAT = netRA_mean.iloc[[0]] 
Cont_PAT = Cont_PAT.rename(index={'Cont': 'Cont Joint RA patients'}) 
Cont_HCPAT_mean = pd.concat([Cont_HC, Cont_PAT]) 

Cont_HC_sem = netHC_sem.iloc[[0]] 
Cont_HC_sem = Cont_HC_sem.rename(index={'Cont': 'Cont Joint HC'}) 
Cont_PAT_sem = netRA_sem.iloc[[0]] 
Cont_PAT_sem = Cont_PAT_sem.rename(index={'Cont': 'Cont Joint RA patients'}) 

Default_HC = netHC_mean.iloc[[1]] 
Default_HC = Default_HC.rename(index={'Default': 'Default Joint HC'}) 
Default_PAT = netRA_mean.iloc[[1]] 
Default_PAT = Default_PAT.rename(index={'Default': 'Default Joint RA patients'}) 
Default_HCPAT_mean = pd.concat([Default_HC, Default_PAT]) 

Default_HC_sem = netHC_sem.iloc[[1]] 
Default_HC_sem = Default_HC_sem.rename(index={'Default': 'Default Joint HC'}) 
Default_PAT_sem = netRA_sem.iloc[[1]] 
Default_PAT_sem = Default_PAT_sem.rename(index={'Default': 'Default Joint RA patients'}) 

DorsAttn_HC = netHC_mean.iloc[[2]] 
DorsAttn_HC = DorsAttn_HC.rename(index={'DorsAttn': 'DorsAttn Joint HC'}) 
DorsAttn_PAT = netRA_mean.iloc[[2]] 
DorsAttn_PAT = DorsAttn_PAT.rename(index={'DorsAttn': 'DorsAttn Joint RA patients'}) 
DorsAttn_HCPAT_mean = pd.concat([DorsAttn_HC, DorsAttn_PAT]) 

DorsAttn_HC_sem = netHC_sem.iloc[[2]] 
DorsAttn_HC_sem = DorsAttn_HC_sem.rename(index={'DorsAttn': 'DorsAttn Joint HC'}) 
DorsAttn_PAT_sem = netRA_sem.iloc[[2]] 
DorsAttn_PAT_sem = DorsAttn_PAT_sem.rename(index={'DorsAttn': 'DorsAttn Joint RA patients'}) 

Limbic_HC = netHC_mean.iloc[[3]] 
Limbic_HC = Limbic_HC.rename(index={'Limbic': 'Limbic Joint HC'}) 
Limbic_PAT = netRA_mean.iloc[[3]] 
Limbic_PAT = Limbic_PAT.rename(index={'Limbic': 'Limbic Joint RA patients'}) 
Limbic_HCPAT_mean = pd.concat([Limbic_HC, Limbic_PAT]) 

Limbic_HC_sem = netHC_sem.iloc[[3]] 
Limbic_HC_sem = Limbic_HC_sem.rename(index={'Limbic': 'Limbic Joint HC'}) 
Limbic_PAT_sem = netRA_sem.iloc[[3]] 
Limbic_PAT_sem = Limbic_PAT_sem.rename(index={'Limbic': 'Limbic Joint RA patients'}) 

SalVentAttn_HC = netHC_mean.iloc[[4]] 
SalVentAttn_HC = SalVentAttn_HC.rename(index={'SalVentAttn': 'SalVentAttn Joint HC'}) 
SalVentAttn_PAT = netRA_mean.iloc[[4]] 
SalVentAttn_PAT = SalVentAttn_PAT.rename(index={'SalVentAttn': 'SalVentAttn Joint RA patients'}) 
SalVentAttn_HCPAT_mean = pd.concat([SalVentAttn_HC, SalVentAttn_PAT]) 

SalVentAttn_HC_sem = netHC_sem.iloc[[4]] 
SalVentAttn_HC_sem = SalVentAttn_HC_sem.rename(index={'SalVentAttn': 'SalVentAttn Joint HC'}) #
SalVentAttn_PAT_sem = netRA_sem.iloc[[4]] 
SalVentAttn_PAT_sem = SalVentAttn_PAT_sem.rename(index={'SalVentAttn': 'SalVentAttn Joint RA patients'}) 

SomMot_HC = netHC_mean.iloc[[5]] 
SomMot_HC = SomMot_HC.rename(index={'SomMot': 'SomMot Joint HC'}) 
SomMot_PAT = netRA_mean.iloc[[5]] 
SomMot_PAT = SomMot_PAT.rename(index={'SomMot': 'SomMot Joint RA patients'}) 
SomMot_HCPAT_mean = pd.concat([SomMot_HC, SomMot_PAT]) 

SomMot_HC_sem = netHC_sem.iloc[[5]] 
SomMot_HC_sem = SomMot_HC_sem.rename(index={'SomMot': 'SomMot Joint HC'}) 
SomMot_PAT_sem = netRA_sem.iloc[[5]] 
SomMot_PAT_sem = SomMot_PAT_sem.rename(index={'SomMot': 'SomMot Joint RA patients'}) 

Vis_HC = netHC_mean.iloc[[6]] 
Vis_HC = Vis_HC.rename(index={'Vis': 'Vis Joint HC'}) 
Vis_PAT = netRA_mean.iloc[[6]] 
Vis_PAT = Vis_PAT.rename(index={'Vis': 'Vis Joint RA patients'}) 
Vis_HCPAT_mean = pd.concat([Vis_HC, Vis_PAT]) 

Vis_HC_sem = netHC_sem.iloc[[6]] 
Vis_HC_sem = Vis_HC_sem.rename(index={'Vis': 'Vis Joint HC'}) 
Vis_PAT_sem = netRA_sem.iloc[[6]] 
Vis_PAT_sem = Vis_PAT_sem.rename(index={'Vis': 'Vis Joint RA patients'}) 


fig = plt.figure(figsize=(5,655))

ax1 = fig.add_subplot(711, ylim=(0.79, 0.81))
ax1.set_xticklabels([])
ax1.tick_params(axis='both', labelsize=20)
ax1.set_title('Cont', size=20)
ax1.plot(Cont_HCPAT_mean.T['Cont Joint HC'], color='darkblue', linewidth=4)
ax1.fill_between(Cont_HCPAT_mean.T.index, Cont_HCPAT_mean.T['Cont Joint HC']-Cont_HC_sem.T['Cont Joint HC'], Cont_HCPAT_mean.T['Cont Joint HC']+Cont_HC_sem.T['Cont Joint HC'], alpha=0.20, color='darkblue')
ax1.plot(Cont_HCPAT_mean.T['Cont Joint RA patients'], color='darkblue', linestyle='dashed', linewidth=4)
ax1.fill_between(Cont_HCPAT_mean.T.index, Cont_HCPAT_mean.T['Cont Joint RA patients']-Cont_PAT_sem.T['Cont Joint RA patients'], Cont_HCPAT_mean.T['Cont Joint RA patients']+Cont_PAT_sem.T['Cont Joint RA patients'], alpha=0.20, color='darkblue')
ax1.axvline(x=2, linewidth=15, color='grey', alpha=0.2)
ax1.axvline(x=4, linewidth=15, color='grey', alpha=0.2)
ax1.yaxis.set_major_locator(MaxNLocator(3))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2 = fig.add_subplot(712, ylim=(0.79, 0.81))
ax2.set_xticklabels([])
ax2.tick_params(axis='both', labelsize=20)
ax2.set_title('Default', size=20)
ax2.plot(Default_HCPAT_mean.T['Default Joint HC'], color='mediumslateblue', linewidth=4)
ax2.fill_between(Default_HCPAT_mean.T.index, Default_HCPAT_mean.T['Default Joint HC']-Default_HC_sem.T['Default Joint HC'], Default_HCPAT_mean.T['Default Joint HC']+Default_HC_sem.T['Default Joint HC'], alpha=0.20, color='mediumslateblue')
ax2.plot(Default_HCPAT_mean.T['Default Joint RA patients'], color='mediumslateblue', linestyle='dashed', linewidth=4)
ax2.fill_between(Default_HCPAT_mean.T.index, Default_HCPAT_mean.T['Default Joint RA patients']-Default_PAT_sem.T['Default Joint RA patients'], Default_HCPAT_mean.T['Default Joint RA patients']+Default_PAT_sem.T['Default Joint RA patients'], alpha=0.20, color='mediumslateblue')
ax2.axvline(x=3, linewidth=15, color='grey', alpha=0.2)
ax2.axvline(x=4, linewidth=15, color='grey', alpha=0.2)
ax2.yaxis.set_major_locator(MaxNLocator(3))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax3 = fig.add_subplot(713, ylim=(0.79, 0.81))
ax3.set_xticklabels([])
ax3.tick_params(axis='both', labelsize=20)
ax3.set_title('DorsAttn', size=20)
ax3.plot(DorsAttn_HCPAT_mean.T['DorsAttn Joint HC'], color='yellowgreen', linewidth=4)
ax3.fill_between(DorsAttn_HCPAT_mean.T.index, DorsAttn_HCPAT_mean.T['DorsAttn Joint HC']-DorsAttn_HC_sem.T['DorsAttn Joint HC'], DorsAttn_HCPAT_mean.T['DorsAttn Joint HC']+DorsAttn_HC_sem.T['DorsAttn Joint HC'], alpha=0.20, color='yellowgreen')
ax3.plot(DorsAttn_HCPAT_mean.T['DorsAttn Joint RA patients'], color='yellowgreen', linestyle='dashed', linewidth=4)
ax3.fill_between(DorsAttn_HCPAT_mean.T.index, DorsAttn_HCPAT_mean.T['DorsAttn Joint RA patients']-DorsAttn_PAT_sem.T['DorsAttn Joint RA patients'], DorsAttn_HCPAT_mean.T['DorsAttn Joint RA patients']+DorsAttn_PAT_sem.T['DorsAttn Joint RA patients'], alpha=0.20, color='yellowgreen')
ax3.axvline(x=2, linewidth=15, color='grey', alpha=0.2)
ax3.axvline(x=3, linewidth=15, color='grey', alpha=0.2)
ax3.axvline(x=4, linewidth=15, color='grey', alpha=0.2)
ax3.axvline(x=5, linewidth=15, color='grey', alpha=0.2)
ax3.yaxis.set_major_locator(MaxNLocator(3))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax4 = fig.add_subplot(714, ylim=(0.79, 0.81))
ax4.set_xticklabels([])
ax4.tick_params(axis='both', labelsize=20)
ax4.set_title('Limbic', size=20)
ax4.plot(Limbic_HCPAT_mean.T['Limbic Joint HC'], color='goldenrod', linewidth=4)
ax4.fill_between(Limbic_HCPAT_mean.T.index, Limbic_HCPAT_mean.T['Limbic Joint HC']-Limbic_HC_sem.T['Limbic Joint HC'], Limbic_HCPAT_mean.T['Limbic Joint HC']+Limbic_HC_sem.T['Limbic Joint HC'], alpha=0.20, color='goldenrod')
ax4.plot(Limbic_HCPAT_mean.T['Limbic Joint RA patients'], color='goldenrod', linestyle='dashed', linewidth=4)
ax4.fill_between(Limbic_HCPAT_mean.T.index, Limbic_HCPAT_mean.T['Limbic Joint RA patients']-Limbic_PAT_sem.T['Limbic Joint RA patients'], Limbic_HCPAT_mean.T['Limbic Joint RA patients']+Limbic_PAT_sem.T['Limbic Joint RA patients'], alpha=0.20, color='goldenrod')
ax4.axvline(x=3, linewidth=15, color='grey', alpha=0.2)
ax4.axvline(x=5, linewidth=15, color='grey', alpha=0.2)
ax4.yaxis.set_major_locator(MaxNLocator(3))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax5 = fig.add_subplot(715, ylim=(0.79, 0.81))
ax5.set_xticklabels([])
ax5.tick_params(axis='both', labelsize=20)
ax5.set_title('SalVentAttn', size=20)
ax5.plot(SalVentAttn_HCPAT_mean.T['SalVentAttn Joint HC'], color='firebrick', linewidth=4)
ax5.fill_between(SalVentAttn_HCPAT_mean.T.index, SalVentAttn_HCPAT_mean.T['SalVentAttn Joint HC']-SalVentAttn_HC_sem.T['SalVentAttn Joint HC'], SalVentAttn_HCPAT_mean.T['SalVentAttn Joint HC']+SalVentAttn_HC_sem.T['SalVentAttn Joint HC'], alpha=0.20, color='firebrick')
ax5.plot(SalVentAttn_HCPAT_mean.T['SalVentAttn Joint RA patients'], color='firebrick', linestyle='dashed', linewidth=4)
ax5.fill_between(SalVentAttn_HCPAT_mean.T.index, SalVentAttn_HCPAT_mean.T['SalVentAttn Joint RA patients']-SalVentAttn_PAT_sem.T['SalVentAttn Joint RA patients'], SalVentAttn_HCPAT_mean.T['SalVentAttn Joint RA patients']+SalVentAttn_PAT_sem.T['SalVentAttn Joint RA patients'], alpha=0.20, color='firebrick')
ax5.axvline(x=2, linewidth=15, color='grey', alpha=0.2)
ax5.axvline(x=3, linewidth=15, color='grey', alpha=0.2)
ax5.axvline(x=4, linewidth=15, color='grey', alpha=0.2)
ax5.axvline(x=5, linewidth=15, color='grey', alpha=0.2)
ax5.yaxis.set_major_locator(MaxNLocator(3))
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax6 = fig.add_subplot(716, ylim=(0.79, 0.81))
ax6.set_xticklabels([])
ax6.tick_params(axis='both', labelsize=20)
ax6.set_title('SomMot', size=20)
ax6.plot(SomMot_HCPAT_mean.T['SomMot Joint HC'], color='darkmagenta', linewidth=4)
ax6.fill_between(SomMot_HCPAT_mean.T.index, SomMot_HCPAT_mean.T['SomMot Joint HC']-SomMot_HC_sem.T['SomMot Joint HC'], SomMot_HCPAT_mean.T['SomMot Joint HC']+SomMot_HC_sem.T['SomMot Joint HC'], alpha=0.20, color='darkmagenta')
ax6.plot(SomMot_HCPAT_mean.T['SomMot Joint RA patients'], color='darkmagenta', linestyle='dashed', linewidth=4)
ax6.fill_between(SomMot_HCPAT_mean.T.index, SomMot_HCPAT_mean.T['SomMot Joint RA patients']-SomMot_PAT_sem.T['SomMot Joint RA patients'], SomMot_HCPAT_mean.T['SomMot Joint RA patients']+SomMot_PAT_sem.T['SomMot Joint RA patients'], alpha=0.20, color='darkmagenta')
ax6.axvline(x=2, linewidth=15, color='grey', alpha=0.2)
ax6.axvline(x=3, linewidth=15, color='grey', alpha=0.2)
ax6.axvline(x=5, linewidth=15, color='grey', alpha=0.2)
ax6.yaxis.set_major_locator(MaxNLocator(3))
ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax7 = fig.add_subplot(717, ylim=(0.79, 0.81))
ax7.tick_params(axis='both', labelsize=20)
timepoints = ['-2','-1','0','1', '2', '3']
ax7.set_xticklabels(timepoints)
ax7.set_title('Vis', size=20)
ax7.plot(Vis_HCPAT_mean.T['Vis Joint HC'], color='violet', linewidth=4)
ax7.fill_between(Vis_HCPAT_mean.T.index, Vis_HCPAT_mean.T['Vis Joint HC']-Vis_HC_sem.T['Vis Joint HC'], Vis_HCPAT_mean.T['Vis Joint HC']+Vis_HC_sem.T['Vis Joint HC'], alpha=0.20, color='violet')
ax7.plot(Vis_HCPAT_mean.T['Vis Joint RA patients'], color='violet', linestyle='dashed', linewidth=4)
ax7.fill_between(Vis_HCPAT_mean.T.index, Vis_HCPAT_mean.T['Vis Joint RA patients']-Vis_PAT_sem.T['Vis Joint RA patients'], Vis_HCPAT_mean.T['Vis Joint RA patients']+Vis_PAT_sem.T['Vis Joint RA patients'], alpha=0.20, color='violet')
ax7.axvline(x=3, linewidth=15, color='grey', alpha=0.2)
ax7.axvline(x=5, linewidth=15, color='grey', alpha=0.2)
ax7.yaxis.set_major_locator(MaxNLocator(3))
ax7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout(pad=2, h_pad=250)