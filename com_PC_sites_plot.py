## Plotting the degree of change in PC (integration) over time at the community level across stimulation sites in patients
## Note that code for plotting the degree of change in within-module degree z-score (z) under the same conditions required line 27 to be changed to  
## tnet = TenetoBIDS(datdir, selected_pipeline='teneto-temporal-degree-centrality', bids_filter=bids_filter).
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

patients = ['1253','1281','1324','1550','1551','1567','1659','2056',
            '2223','2350','2585','2613','2876','3006',
           '306','3578','3917','4716','4903','5174','5291',
           '5447','5749','6230','874','7712','8021','8712'] #tot 28

bids_filter = {'subject': patients,
               'run': [1,2],
               'task': ['joint','thumb']}

# Defining the TenetoBIDS object
tnet = TenetoBIDS(datdir, selected_pipeline='teneto-temporal-participation-coeff', bids_filter=bids_filter)

data = tnet.load_data()

uniquenet = pd.read_csv('/data/silfan/RA/pre/bids/derivatives/teneto-make-parcellation/sub-306/func/sub-306_run-1_task-joint_timeseries.tsv', sep='\t')

uniquenet.rename(columns={'Unnamed: 0':'networks'}, inplace=True)
uniquenet['networks'] = uniquenet['networks'].str.replace('\d+', '') 
uniquenet['networks'] = uniquenet['networks'].str.rstrip('_') 
uniquenet['networks'] = uniquenet['networks'].str.replace('_LH','') 
uniquenet['networks'] = uniquenet['networks'].str.replace('_RH','') 
uniquenet['networks'] = uniquenet['networks'].str.replace('Networks_','') 
uniquenet['networks'] = uniquenet['networks'].str.split('_').str[0] 

## Data preparation for plotting median PC of all nodes within each community over time (-2TR,-1TR,onset,+1TR,+2TR,+3TR)

# Painful stimulation to joint
netJ = []

for key in data.keys():
    file_ent = tnet.BIDSLayout.parse_file_entities(str(key))
    rootdir = "/data/silfan/RA/pre/bids/sub-" + str(key.split('_')[0].split('-')[1]) + "/func/sub-" + str(key.split('_')[0].split('-')[1]) + "_task-" + file_ent['task'] + "_run-" + str(file_ent['run']) + "_events.tsv"
    if "task-joint" in rootdir:
        events = pd.read_csv(rootdir, sep=';')
        events = events.loc[:, ~events.columns.str.contains('^Unnamed')]

        events['onsetTR'] = events['onset'].div(3).round(0) 
        events['onsetTR'] = list(events['onsetTR'].astype('int64'))
        events['onsetTR'] = events['onsetTR'].astype(str)

        pain_df = events[events['trial_type'] == 'pain'] 

        PC_df = data[key]
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

        netJ.append(net)

netJ_col = pd.concat(netJ, axis=1)
netJ_mean = netJ_col.groupby(by=netJ_col.columns, axis=1).mean()
netJ_mean = netJ_mean[['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']] 

#Painful stimulation to thumb
netT = []

for key in data.keys():
    file_ent = tnet.BIDSLayout.parse_file_entities(str(key))
    rootdir = "/data/silfan/RA/pre/bids/sub-" + str(key.split('_')[0].split('-')[1]) + "/func/sub-" + str(key.split('_')[0].split('-')[1]) + "_task-" + file_ent['task'] + "_run-" + str(file_ent['run']) + "_events.tsv"
    if "task-thumb" in rootdir:
        events = pd.read_csv(rootdir, sep=';')
        events = events.loc[:, ~events.columns.str.contains('^Unnamed')]

        events['onsetTR'] = events['onset'].div(3).round(0) 
        events['onsetTR'] = list(events['onsetTR'].astype('int64'))
        events['onsetTR'] = events['onsetTR'].astype(str)

        pain_df = events[events['trial_type'] == 'pain'] 

        PC_df = data[key]
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

        netT.append(net)

netT_col = pd.concat(netT, axis=1)
netT_mean = netT_col.groupby(by=netT_col.columns, axis=1).mean()
netT_mean = netT_mean[['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']] 

## Computing standard error of the mean (sem) 

subjects = numpy.concatenate([([i]*2) for i in [306,874,1253,1281,1324,1550,1551,1567,1659,2056,2223,2350,2585,2613,2876,3006,3578,3917,4716,4903,5174,5291,5447,5749,6230,7712,8021,8712]], axis=0) #tot=28
subjects = subjects.tolist() 

networks = ['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']
timepoints = ['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']
joint_sem_dict = {}
thumb_sem_dict = {}

for network in networks:
    for time in timepoints:
        # Joint
        J = netJ_col.loc[[network], [time]].T

        J = J.reset_index() 
        J = J[network] 
        J = J.to_frame() 

        J = J.assign(patients=subjects) 
        J = J.rename(columns = {network:'JOINT'})
        colswap = ['patients', 'JOINT']
        J = J.reindex(columns=colswap) 

        # Thumb
        T = netT_col.loc[[network], [time]] 
        T = T.T
        T = T.reset_index() 
        T = T[network] 

        JT = J.assign(THUMB=T) 

        joint = JT.groupby('patients')['JOINT'].agg(averageJOINT='mean').reset_index() # mean between runs 1 and 2 of the same subject 
        thumb = JT.groupby('patients')['THUMB'].agg(averageTHUMB='mean').reset_index() # mean between runs 1 and 2 of the same subject
        joithu = pd.merge(joint, thumb, on=['patients']) # merging joint and thumb data frames

        joithutime = joithu.assign(time=time)
        joithutimeROI = joithutime.assign(network=network)

        sem_J = sem(joithutimeROI['averageJOINT'])
        sem_T = sem(joithutimeROI['averageTHUMB'])

        joint_sem_dict['{}_{}'.format(network,time)] = sem_J
        thumb_sem_dict['{}_{}'.format(network,time)] = sem_T


joint_sem_df = pd.DataFrame()
joint_sem_df = joint_sem_df.append(joint_sem_dict, ignore_index=True) 

joint_sem_df = joint_sem_df.T.reset_index() 
joint_sem_df = joint_sem_df.rename(columns = {0:'sem', 'index':'networktime'})

joint_sem_df['network'],joint_sem_df['time'] = joint_sem_df['networktime'].str.split('_',1).str 
joint_sem_df = joint_sem_df.drop('networktime', 1) 

netJ_sem = joint_sem_df.groupby("network")["sem"].apply(lambda x: pd.Series(x.values)).unstack().add_prefix('time')
netJ_sem = netJ_sem.rename(columns={'time0': 'pre2TR', 'time1': 'pre1TR', 'time2': 'onset', 'time3': 'post1TR', 'time4': 'post2TR', 'time5': 'post3TR'})
netJ_sem.index.names = ['networks']
netJ_sem = netJ_sem.reindex(['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']) 

thumb_sem_df = pd.DataFrame()
thumb_sem_df = thumb_sem_df.append(thumb_sem_dict, ignore_index=True) 

thumb_sem_df = thumb_sem_df.T.reset_index() 
thumb_sem_df = thumb_sem_df.rename(columns = {0:'sem', 'index':'networktime'})

thumb_sem_df['network'],thumb_sem_df['time'] = thumb_sem_df['networktime'].str.split('_',1).str 
thumb_sem_df = thumb_sem_df.drop('networktime', 1) 

netT_sem = thumb_sem_df.groupby("network")["sem"].apply(lambda x: pd.Series(x.values)).unstack().add_prefix('time')
netT_sem = netT_sem.rename(columns={'time0': 'pre2TR', 'time1': 'pre1TR', 'time2': 'onset', 'time3': 'post1TR', 'time4': 'post2TR', 'time5': 'post3TR'})
netT_sem.index.names = ['networks']
netT_sem = netT_sem.reindex(['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']) 

## Plotting all 7 networks in the same panel

Cont_j = netJ_mean.iloc[[0]] 
Cont_j = Cont_j.rename(index={'Cont': 'Cont Joint'}) 
Cont_t = netT_mean.iloc[[0]] 
Cont_t = Cont_t.rename(index={'Cont': 'Cont Thumb'}) 
Cont_JT_mean = pd.concat([Cont_j, Cont_t]) 

Cont_j_sem = netJ_sem.iloc[[0]] 
Cont_j_sem = Cont_j_sem.rename(index={'Cont': 'Cont Joint'})
Cont_t_sem = netT_sem.iloc[[0]] 
Cont_t_sem = Cont_t_sem.rename(index={'Cont': 'Cont Thumb'}) 

Default_j = netJ_mean.iloc[[1]] 
Default_j = Default_j.rename(index={'Default': 'Default Joint'}) 
Default_t = netT_mean.iloc[[1]] 
Default_t = Default_t.rename(index={'Default': 'Default Thumb'}) 
Default_JT_mean = pd.concat([Default_j, Default_t]) 

Default_j_sem = netJ_sem.iloc[[1]] 
Default_j_sem = Default_j_sem.rename(index={'Default': 'Default Joint'}) 
Default_t_sem = netT_sem.iloc[[1]] 
Default_t_sem = Default_t_sem.rename(index={'Default': 'Default Thumb'}) 

DorsAttn_j = netJ_mean.iloc[[2]] 
DorsAttn_j = DorsAttn_j.rename(index={'DorsAttn': 'DorsAttn Joint'}) 
DorsAttn_t = netT_mean.iloc[[2]] 
DorsAttn_t = DorsAttn_t.rename(index={'DorsAttn': 'DorsAttn Thumb'}) 
DorsAttn_JT_mean = pd.concat([DorsAttn_j, DorsAttn_t]) 

DorsAttn_j_sem = netJ_sem.iloc[[2]] 
DorsAttn_j_sem = DorsAttn_j_sem.rename(index={'DorsAttn': 'DorsAttn Joint'}) 
DorsAttn_t_sem = netT_sem.iloc[[2]] 
DorsAttn_t_sem = DorsAttn_t_sem.rename(index={'DorsAttn': 'DorsAttn Thumb'}) 

Limbic_j = netJ_mean.iloc[[3]] 
Limbic_j = Limbic_j.rename(index={'Limbic': 'Limbic Joint'}) 
Limbic_t = netT_mean.iloc[[3]] 
Limbic_t = Limbic_t.rename(index={'Limbic': 'Limbic Thumb'}) 
Limbic_JT_mean = pd.concat([Limbic_j, Limbic_t]) 

Limbic_j_sem = netJ_sem.iloc[[3]] 
Limbic_j_sem = Limbic_j_sem.rename(index={'Limbic': 'Limbic Joint'}) 
Limbic_t_sem = netT_sem.iloc[[3]] 
Limbic_t_sem = Limbic_t_sem.rename(index={'Limbic': 'Limbic Thumb'}) 

SalVentAttn_j = netJ_mean.iloc[[4]] 
SalVentAttn_j = SalVentAttn_j.rename(index={'SalVentAttn': 'SalVentAttn Joint'}) 
SalVentAttn_t = netT_mean.iloc[[4]] 
SalVentAttn_t = SalVentAttn_t.rename(index={'SalVentAttn': 'SalVentAttn Thumb'}) 
SalVentAttn_JT_mean = pd.concat([SalVentAttn_j, SalVentAttn_t]) 

SalVentAttn_j_sem = netJ_sem.iloc[[4]] 
SalVentAttn_j_sem = SalVentAttn_j_sem.rename(index={'SalVentAttn': 'SalVentAttn Joint'}) 
SalVentAttn_t_sem = netT_sem.iloc[[4]] 
SalVentAttn_t_sem = SalVentAttn_t_sem.rename(index={'SalVentAttn': 'SalVentAttn Thumb'})

SomMot_j = netJ_mean.iloc[[5]] 
SomMot_j = SomMot_j.rename(index={'SomMot': 'SomMot Joint'}) 
SomMot_t = netT_mean.iloc[[5]] 
SomMot_t = SomMot_t.rename(index={'SomMot': 'SomMot Thumb'}) 
SomMot_JT_mean = pd.concat([SomMot_j, SomMot_t]) 

SomMot_j_sem = netJ_sem.iloc[[5]] 
SomMot_j_sem = SomMot_j_sem.rename(index={'SomMot': 'SomMot Joint'}) 
SomMot_t_sem = netT_sem.iloc[[5]] 
SomMot_t_sem = SomMot_t_sem.rename(index={'SomMot': 'SomMot Thumb'}) 

Vis_j = netJ_mean.iloc[[6]] 
Vis_j = Vis_j.rename(index={'Vis': 'Vis Joint'}) 
Vis_t = netT_mean.iloc[[6]] 
Vis_t = Vis_t.rename(index={'Vis': 'Vis Thumb'}) 
Vis_JT_mean = pd.concat([Vis_j, Vis_t]) 

Vis_j_sem = netJ_sem.iloc[[6]] 
Vis_j_sem = Vis_j_sem.rename(index={'Vis': 'Vis Joint'}) 
Vis_t_sem = netT_sem.iloc[[6]] 
Vis_t_sem = Vis_t_sem.rename(index={'Vis': 'Vis Thumb'})

fig = plt.figure(figsize=(5,655))

ax1 = fig.add_subplot(711, ylim=(0.79, 0.81))
ax1.set_title('Cont',size=20)
ax1.set_xticklabels([])
ax1.tick_params(axis='both', labelsize=20)
ax1.plot(Cont_JT_mean.T['Cont Joint'], color='forestgreen', linewidth=4)
ax1.fill_between(Cont_JT_mean.T.index, Cont_JT_mean.T['Cont Joint']-Cont_j_sem.T['Cont Joint'], Cont_JT_mean.T['Cont Joint']+Cont_j_sem.T['Cont Joint'], alpha=0.20, color='forestgreen')
ax1.plot(Cont_JT_mean.T['Cont Thumb'], color='forestgreen', linestyle='dashed', linewidth=4)
ax1.fill_between(Cont_JT_mean.T.index, Cont_JT_mean.T['Cont Thumb']-Cont_t_sem.T['Cont Thumb'], Cont_JT_mean.T['Cont Thumb']+Cont_t_sem.T['Cont Thumb'], alpha=0.20, color='forestgreen')
ax1.yaxis.set_major_locator(MaxNLocator(3))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2 = fig.add_subplot(712, ylim=(0.79, 0.81))
ax2.set_title('Default', size=20)
ax2.set_xticklabels([])
ax2.tick_params(axis='both', labelsize=20)
ax2.plot(Default_JT_mean.T['Default Joint'], color='darkorange', linewidth=4)
ax2.fill_between(Default_JT_mean.T.index, Default_JT_mean.T['Default Joint']-Default_j_sem.T['Default Joint'], Default_JT_mean.T['Default Joint']+Default_j_sem.T['Default Joint'], alpha=0.20, color='darkorange')
ax2.plot(Default_JT_mean.T['Default Thumb'], color='darkorange', linestyle='dashed', linewidth=4)
ax2.fill_between(Default_JT_mean.T.index, Default_JT_mean.T['Default Thumb']-Default_t_sem.T['Default Thumb'], Default_JT_mean.T['Default Thumb']+Default_t_sem.T['Default Thumb'], alpha=0.20, color='darkorange')
ax2.yaxis.set_major_locator(MaxNLocator(3))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax3 = fig.add_subplot(713, ylim=(0.79, 0.81))
ax3.set_title('DorsAttn', size=20)
ax3.set_xticklabels([])
ax3.tick_params(axis='both', labelsize=20)
ax3.plot(DorsAttn_JT_mean.T['DorsAttn Joint'], color='mediumslateblue', linewidth=4)
ax3.fill_between(DorsAttn_JT_mean.T.index, DorsAttn_JT_mean.T['DorsAttn Joint']-DorsAttn_j_sem.T['DorsAttn Joint'], DorsAttn_JT_mean.T['DorsAttn Joint']+DorsAttn_j_sem.T['DorsAttn Joint'], alpha=0.20, color='mediumslateblue')
ax3.plot(DorsAttn_JT_mean.T['DorsAttn Thumb'], color='mediumslateblue', linestyle='dashed', linewidth=4)
ax3.fill_between(DorsAttn_JT_mean.T.index, DorsAttn_JT_mean.T['DorsAttn Thumb']-DorsAttn_t_sem.T['DorsAttn Thumb'], DorsAttn_JT_mean.T['DorsAttn Thumb']+DorsAttn_t_sem.T['DorsAttn Thumb'], alpha=0.20, color='mediumslateblue')
ax3.yaxis.set_major_locator(MaxNLocator(3))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax4 = fig.add_subplot(714, ylim=(0.79, 0.81))
ax4.set_title('Limbic', size=20)
ax4.set_xticklabels([])
ax4.tick_params(axis='both', labelsize=20)
ax4.plot(Limbic_JT_mean.T['Limbic Joint'], color='limegreen', linewidth=4)
ax4.fill_between(Limbic_JT_mean.T.index, Limbic_JT_mean.T['Limbic Joint']-Limbic_j_sem.T['Limbic Joint'], Limbic_JT_mean.T['Limbic Joint']+Limbic_j_sem.T['Limbic Joint'], alpha=0.20, color='limegreen')
ax4.plot(Limbic_JT_mean.T['Limbic Thumb'], color='limegreen', linestyle='dashed', linewidth=4)
ax4.fill_between(Limbic_JT_mean.T.index, Limbic_JT_mean.T['Limbic Thumb']-Limbic_t_sem.T['Limbic Thumb'], Limbic_JT_mean.T['Limbic Thumb']+Limbic_t_sem.T['Limbic Thumb'], alpha=0.20, color='limegreen')
ax4.yaxis.set_major_locator(MaxNLocator(3))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax5 = fig.add_subplot(715, ylim=(0.79, 0.81))
ax5.set_title('SalVentAttn', size=20)
ax5.set_xticklabels([])
ax5.tick_params(axis='both', labelsize=20)
ax5.plot(SalVentAttn_JT_mean.T['SalVentAttn Joint'], color='orange', linewidth=4)
ax5.fill_between(SalVentAttn_JT_mean.T.index, SalVentAttn_JT_mean.T['SalVentAttn Joint']-SalVentAttn_j_sem.T['SalVentAttn Joint'], SalVentAttn_JT_mean.T['SalVentAttn Joint']+SalVentAttn_j_sem.T['SalVentAttn Joint'], alpha=0.20, color='orange')
ax5.plot(SalVentAttn_JT_mean.T['SalVentAttn Thumb'], color='orange', linestyle='dashed', linewidth=4)
ax5.fill_between(SalVentAttn_JT_mean.T.index, SalVentAttn_JT_mean.T['SalVentAttn Thumb']-SalVentAttn_t_sem.T['SalVentAttn Thumb'], SalVentAttn_JT_mean.T['SalVentAttn Thumb']+SalVentAttn_t_sem.T['SalVentAttn Thumb'], alpha=0.20, color='orange')
ax5.yaxis.set_major_locator(MaxNLocator(3))
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax6 = fig.add_subplot(716, ylim=(0.79, 0.81))
ax6.set_title('SomMot', size=20)
ax6.set_xticklabels([])
ax6.tick_params(axis='both', labelsize=20)
ax6.plot(SomMot_JT_mean.T['SomMot Joint'], color='peru', linewidth=4)
ax6.fill_between(SomMot_JT_mean.T.index, SomMot_JT_mean.T['SomMot Joint']-SomMot_j_sem.T['SomMot Joint'], SomMot_JT_mean.T['SomMot Joint']+SomMot_j_sem.T['SomMot Joint'], alpha=0.20, color='peru')
ax6.plot(SomMot_JT_mean.T['SomMot Thumb'], color='peru', linestyle='dashed', linewidth=4)
ax6.fill_between(SomMot_JT_mean.T.index, SomMot_JT_mean.T['SomMot Thumb']-SomMot_t_sem.T['SomMot Thumb'], SomMot_JT_mean.T['SomMot Thumb']+SomMot_t_sem.T['SomMot Thumb'], alpha=0.20, color='peru')
ax6.yaxis.set_major_locator(MaxNLocator(3))
ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax7 = fig.add_subplot(717, ylim=(0.79, 0.81))
ax7.tick_params(axis='both', labelsize=20)
timepoints = ['-2','-1','0','1', '2', '3']
ax7.set_xticklabels(timepoints)
ax7.set_title('Vis', size=20)
ax7.plot(Vis_JT_mean.T['Vis Joint'], color='cornflowerblue', linewidth=4)
ax7.fill_between(Vis_JT_mean.T.index, Vis_JT_mean.T['Vis Joint']-Vis_j_sem.T['Vis Joint'], Vis_JT_mean.T['Vis Joint']+Vis_j_sem.T['Vis Joint'], alpha=0.20, color='cornflowerblue')
ax7.plot(Vis_JT_mean.T['Vis Thumb'], color='cornflowerblue', linestyle='dashed', linewidth=4)
ax7.fill_between(Vis_JT_mean.T.index, Vis_JT_mean.T['Vis Thumb']-Vis_t_sem.T['Vis Thumb'], Vis_JT_mean.T['Vis Thumb']+Vis_t_sem.T['Vis Thumb'], alpha=0.20, color='cornflowerblue')
ax7.yaxis.set_major_locator(MaxNLocator(3))
ax7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout(pad=2, h_pad=250)