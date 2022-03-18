## Plotting the degree of change in PC (integration) over time at the nodel level across stimulation sites in patients
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
tnet=TenetoBIDS(datdir, selected_pipeline='teneto-temporal-participation-coeff', bids_filter=bids_filter)

data = tnet.load_data()

## Data preparation for plotting PC of every selected node over time (-2TR,-1TR,onset,+1TR,+2TR,+3TR)

# Painful stimulation to joint
nodesJ = []

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

        nodesJ.append(nodes)

nodesJ_col = pd.concat(nodesJ, axis=1)
nodesJ_mean = nodesJ_col.groupby(by=nodesJ_col.columns, axis=1).mean()
nodesJ_mean = nodesJ_mean[['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']] 

# Painful stimulation to thumb
nodesT = []

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

        nodesT.append(nodes)

nodesT_col = pd.concat(nodesT, axis=1)
nodesT_mean = nodesT_col.groupby(by=nodesT_col.columns, axis=1).mean()
nodesT_mean = nodesT_mean[['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']] 

## Computing standard error of the mean (sem) 

subjects = numpy.concatenate([([i]*2) for i in [306,874,1253,1281,1324,1550,1551,1567,1659,2056,2223,2350,2585,2613,2876,3006,3578,3917,4716,4903,5174,5291,5447,5749,6230,7712,8021,8712]], axis=0) #tot=28
subjects = subjects.tolist() 

ROIS = ['L AIns', 'R AIns', 'L PIns', 'R PIns', 'L ACgG', 'R ACgG']
timepoints = ['pre2TR', 'pre1TR', 'onset', 'post1TR', 'post2TR', 'post3TR']
joint_sem_dict = {}
thumb_sem_dict = {}

for ROI in ROIS:
    for time in timepoints:
        # Joint
        J = nodesJ_col.loc[[ROI], [time]].T

        J = J.reset_index() 
        J = J[ROI] 
        J = J.to_frame() 

        J = J.assign(patients=subjects)
        J = J.rename(columns = {ROI:'JOINT'})
        colswap = ['patients', 'JOINT']
        J = J.reindex(columns=colswap) 

        # Thumb
        T = nodesT_col.loc[[ROI], [time]] 
        T = T.T 
        T = T.reset_index() 
        T = T[ROI] 

        JT = J.assign(THUMB=T) 

        joint = JT.groupby('patients')['JOINT'].agg(averageJOINT='mean').reset_index() #mean between runs 1 and 2 of the same subject
        thumb = JT.groupby('patients')['THUMB'].agg(averageTHUMB='mean').reset_index() #mean between runs 1 and 2 of the same subject
        joithu = pd.merge(joint, thumb, on=['patients']) #merging joint and thumb data frames

        joithutime = joithu.assign(time=time)
        joithutimeROI = joithutime.assign(ROI=ROI)

        sem_J = sem(joithutimeROI['averageJOINT'])
        sem_T = sem(joithutimeROI['averageTHUMB'])

        joint_sem_dict['{}_{}'.format(ROI,time)] = sem_J
        thumb_sem_dict['{}_{}'.format(ROI,time)] = sem_T


joint_sem_df = pd.DataFrame()
joint_sem_df = joint_sem_df.append(joint_sem_dict, ignore_index=True) 

joint_sem_df = joint_sem_df.T.reset_index() 
joint_sem_df = joint_sem_df.rename(columns = {0:'sem', 'index':'nodetime'})

joint_sem_df['node'],joint_sem_df['time'] = joint_sem_df['nodetime'].str.split('_',1).str 
joint_sem_df = joint_sem_df.drop('nodetime', 1) 

nodesJ_sem = joint_sem_df.groupby("node")["sem"].apply(lambda x: pd.Series(x.values)).unstack().add_prefix('time')
nodesJ_sem = nodesJ_sem.rename(columns={'time0': 'pre2TR', 'time1': 'pre1TR', 'time2': 'onset', 'time3': 'post1TR', 'time4': 'post2TR', 'time5': 'post3TR'})
nodesJ_sem.index.names = ['ROIs']
nodesJ_sem = nodesJ_sem.reindex(["L AIns", "R AIns", "L PIns", "R PIns", "L ACgG", "R ACgG"]) 

thumb_sem_df = pd.DataFrame()
thumb_sem_df = thumb_sem_df.append(thumb_sem_dict, ignore_index=True) 

thumb_sem_df = thumb_sem_df.T.reset_index() 
thumb_sem_df = thumb_sem_df.rename(columns = {0:'sem', 'index':'nodetime'})

thumb_sem_df['node'],thumb_sem_df['time'] = thumb_sem_df['nodetime'].str.split('_',1).str 
thumb_sem_df = thumb_sem_df.drop('nodetime', 1) 

nodesT_sem = thumb_sem_df.groupby("node")["sem"].apply(lambda x: pd.Series(x.values)).unstack().add_prefix('time')
nodesT_sem = nodesT_sem.rename(columns={'time0': 'pre2TR', 'time1': 'pre1TR', 'time2': 'onset', 'time3': 'post1TR', 'time4': 'post2TR', 'time5': 'post3TR'})
nodesT_sem.index.names = ['ROIs']
nodesT_sem = nodesT_sem.reindex(["L AIns", "R AIns", "L PIns", "R PIns", "L ACgG", "R ACgG"]) 

## Plotting all 6 nodes in the same panel

LAIns_j = nodesJ_mean.iloc[[0]] 
LAIns_j = LAIns_j.rename(index={'L AIns': 'L AIns Joint'}) 
LAIns_t = nodesT_mean.iloc[[0]] 
LAIns_t = LAIns_t.rename(index={'L AIns': 'L AIns Thumb'}) 
LAIns_JT_mean = pd.concat([LAIns_j, LAIns_t]) 

LAIns_j_sem = nodesJ_sem.iloc[[0]] 
LAIns_j_sem = LAIns_j_sem.rename(index={'L AIns': 'L AIns Joint'}) 
LAIns_t_sem = nodesT_sem.iloc[[0]]
LAIns_t_sem = LAIns_t_sem.rename(index={'L AIns': 'L AIns Thumb'}) 

RAIns_j = nodesJ_mean.iloc[[1]] 
RAIns_j = RAIns_j.rename(index={'R AIns': 'R AIns Joint'}) 
RAIns_t = nodesT_mean.iloc[[1]] 
RAIns_t = RAIns_t.rename(index={'R AIns': 'R AIns Thumb'}) 
RAIns_JT_mean = pd.concat([RAIns_j, RAIns_t]) 

RAIns_j_sem = nodesJ_sem.iloc[[1]] 
RAIns_j_sem = RAIns_j_sem.rename(index={'R AIns': 'R AIns Joint'}) 
RAIns_t_sem = nodesT_sem.iloc[[1]] 
RAIns_t_sem = RAIns_t_sem.rename(index={'R AIns': 'R AIns Thumb'}) 

LPIns_j = nodesJ_mean.iloc[[2]] 
LPIns_j = LPIns_j.rename(index={'L PIns': 'L PIns Joint'}) 
LPIns_t = nodesT_mean.iloc[[2]] 
LPIns_t = LPIns_t.rename(index={'L PIns': 'L PIns Thumb'}) 
LPIns_JT_mean = pd.concat([LPIns_j, LPIns_t]) 

LPIns_j_sem = nodesJ_sem.iloc[[2]] 
LPIns_j_sem = LPIns_j_sem.rename(index={'L PIns': 'L PIns Joint'}) 
LPIns_t_sem = nodesT_sem.iloc[[2]] 
LPIns_t_sem = LPIns_t_sem.rename(index={'L PIns': 'L PIns Thumb'}) 

RPIns_j = nodesJ_mean.iloc[[3]] 
RPIns_j = RPIns_j.rename(index={'R PIns': 'R PIns Joint'}) 
RPIns_t = nodesT_mean.iloc[[3]] 
RPIns_t = RPIns_t.rename(index={'R PIns': 'R PIns Thumb'}) 
RPIns_JT_mean = pd.concat([RPIns_j, RPIns_t]) 

RPIns_j_sem = nodesJ_sem.iloc[[3]] 
RPIns_j_sem = RPIns_j_sem.rename(index={'R PIns': 'R PIns Joint'}) 
RPIns_t_sem = nodesT_sem.iloc[[3]] 
RPIns_t_sem = RPIns_t_sem.rename(index={'R PIns': 'R PIns Thumb'}) 

LACgG_j = nodesJ_mean.iloc[[4]] 
LACgG_j = LACgG_j.rename(index={'L ACgG': 'L ACgG Joint'}) 
LACgG_t = nodesT_mean.iloc[[4]] 
LACgG_t = LACgG_t.rename(index={'L ACgG': 'L ACgG Thumb'}) 
LACgG_JT_mean = pd.concat([LACgG_j, LACgG_t]) 

LACgG_j_sem = nodesJ_sem.iloc[[4]] 
LACgG_j_sem = LACgG_j_sem.rename(index={'L ACgG': 'L ACgG Joint'}) 
LACgG_t_sem = nodesT_sem.iloc[[4]] 
LACgG_t_sem = LACgG_t_sem.rename(index={'L ACgG': 'L ACgG Thumb'}) 

RACgG_j = nodesJ_mean.iloc[[5]] 
RACgG_j = RACgG_j.rename(index={'R ACgG': 'R ACgG Joint'}) 
RACgG_t = nodesT_mean.iloc[[5]] 
RACgG_t = RACgG_t.rename(index={'R ACgG': 'R ACgG Thumb'}) 
RACgG_JT_mean = pd.concat([RACgG_j, RACgG_t]) 

RACgG_j_sem = nodesJ_sem.iloc[[5]] 
RACgG_j_sem = RACgG_j_sem.rename(index={'R ACgG': 'R ACgG Joint'}) 
RACgG_t_sem = nodesT_sem.iloc[[5]] 
RACgG_t_sem = RACgG_t_sem.rename(index={'R ACgG': 'R ACgG Thumb'}) 

fig = plt.figure(figsize=(5,655))

ax1 = fig.add_subplot(611, ylim=(0.76, 0.80))
ax1.set_title('L AIns', size=20)
ax1.set_xticklabels([])
ax1.tick_params(axis='both', labelsize=20)
ax1.plot(LAIns_JT_mean.T['L AIns Joint'], color='mediumpurple', linewidth=4)
ax1.fill_between(LAIns_JT_mean.T.index, LAIns_JT_mean.T['L AIns Joint']-LAIns_j_sem.T['L AIns Joint'], LAIns_JT_mean.T['L AIns Joint']+LAIns_j_sem.T['L AIns Joint'], alpha=0.20, color='mediumpurple')
ax1.plot(LAIns_JT_mean.T['L AIns Thumb'], color='mediumpurple', linestyle='dashed', linewidth=4)
ax1.fill_between(LAIns_JT_mean.T.index, LAIns_JT_mean.T['L AIns Thumb']-LAIns_t_sem.T['L AIns Thumb'], LAIns_JT_mean.T['L AIns Thumb']+LAIns_t_sem.T['L AIns Thumb'], alpha=0.20, color='mediumpurple')
ax1.yaxis.set_major_locator(MaxNLocator(3))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2 = fig.add_subplot(612, ylim=(0.76, 0.80))
ax2.set_title('R AIns', size=20)
ax2.set_xticklabels([])
ax2.tick_params(axis='both', labelsize=20)
ax2.plot(RAIns_JT_mean.T['R AIns Joint'], color='mediumorchid', linewidth=4)
ax2.fill_between(RAIns_JT_mean.T.index, RAIns_JT_mean.T['R AIns Joint']-RAIns_j_sem.T['R AIns Joint'], RAIns_JT_mean.T['R AIns Joint']+RAIns_j_sem.T['R AIns Joint'], alpha=0.20, color='mediumorchid')
ax2.plot(RAIns_JT_mean.T['R AIns Thumb'], color='mediumorchid', linestyle='dashed', linewidth=4)
ax2.fill_between(RAIns_JT_mean.T.index, RAIns_JT_mean.T['R AIns Thumb']-RAIns_t_sem.T['R AIns Thumb'], RAIns_JT_mean.T['R AIns Thumb']+RAIns_t_sem.T['R AIns Thumb'], alpha=0.20, color='mediumorchid')
ax2.yaxis.set_major_locator(MaxNLocator(3))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax3 = fig.add_subplot(613, ylim=(0.76, 0.80))
ax3.set_title('L PIns', size=20)
ax3.set_xticklabels([])
ax3.tick_params(axis='both', labelsize=20)
ax3.plot(LPIns_JT_mean.T['L PIns Joint'], color='mediumturquoise', linewidth=4)
ax3.fill_between(LPIns_JT_mean.T.index, LPIns_JT_mean.T['L PIns Joint']-LPIns_j_sem.T['L PIns Joint'], LPIns_JT_mean.T['L PIns Joint']+LPIns_j_sem.T['L PIns Joint'], alpha=0.20, color='mediumturquoise')
ax3.plot(LPIns_JT_mean.T['L PIns Thumb'], color='mediumturquoise', linestyle='dashed', linewidth=4)
ax3.fill_between(LPIns_JT_mean.T.index, LPIns_JT_mean.T['L PIns Thumb']-LPIns_t_sem.T['L PIns Thumb'], LPIns_JT_mean.T['L PIns Thumb']+LPIns_t_sem.T['L PIns Thumb'], alpha=0.20, color='mediumturquoise')
ax3.yaxis.set_major_locator(MaxNLocator(3))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax4 = fig.add_subplot(614, ylim=(0.76, 0.80))
ax4.set_title('R PIns', size=20)
ax4.set_xticklabels([])
ax4.tick_params(axis='both', labelsize=20)
ax4.plot(RPIns_JT_mean.T['R PIns Joint'], color='pink', linewidth=4)
ax4.fill_between(RPIns_JT_mean.T.index, RPIns_JT_mean.T['R PIns Joint']-RPIns_j_sem.T['R PIns Joint'], RPIns_JT_mean.T['R PIns Joint']+RPIns_j_sem.T['R PIns Joint'], alpha=0.20, color='pink')
ax4.plot(RPIns_JT_mean.T['R PIns Thumb'], color='pink', linestyle='dashed', linewidth=4)
ax4.fill_between(RPIns_JT_mean.T.index, RPIns_JT_mean.T['R PIns Thumb']-RPIns_t_sem.T['R PIns Thumb'], RPIns_JT_mean.T['R PIns Thumb']+RPIns_t_sem.T['R PIns Thumb'], alpha=0.20, color='pink')
ax4.yaxis.set_major_locator(MaxNLocator(3))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax5 = fig.add_subplot(615, ylim=(0.76, 0.80))
ax5.set_title('L ACgG', size=20)
ax5.set_xticklabels([])
ax5.tick_params(axis='both', labelsize=20)
ax5.plot(LACgG_JT_mean.T['L ACgG Joint'], color='sandybrown', linewidth=4)
ax5.fill_between(LACgG_JT_mean.T.index, LACgG_JT_mean.T['L ACgG Joint']-LACgG_j_sem.T['L ACgG Joint'], LACgG_JT_mean.T['L ACgG Joint']+LACgG_j_sem.T['L ACgG Joint'], alpha=0.20, color='sandybrown')
ax5.plot(LACgG_JT_mean.T['L ACgG Thumb'], color='sandybrown', linestyle='dashed', linewidth=4)
ax5.fill_between(LACgG_JT_mean.T.index, LACgG_JT_mean.T['L ACgG Thumb']-LACgG_t_sem.T['L ACgG Thumb'], LACgG_JT_mean.T['L ACgG Thumb']+LACgG_t_sem.T['L ACgG Thumb'], alpha=0.20, color='sandybrown')
ax5.yaxis.set_major_locator(MaxNLocator(3))
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax6 = fig.add_subplot(616, ylim=(0.76, 0.80))
ax6.set_title('R ACgG', size=20)
timepoints = ['-2','-1','0','1', '2', '3']
ax6.set_xticklabels(timepoints)
ax6.tick_params(axis='both', labelsize=20)
ax6.plot(RACgG_JT_mean.T['R ACgG Joint'], color='khaki', linewidth=4)
ax6.fill_between(RACgG_JT_mean.T.index, RACgG_JT_mean.T['R ACgG Joint']-RACgG_j_sem.T['R ACgG Joint'], RACgG_JT_mean.T['R ACgG Joint']+RACgG_j_sem.T['R ACgG Joint'], alpha=0.20, color='khaki')
ax6.plot(RACgG_JT_mean.T['R ACgG Thumb'], color='khaki', linestyle='dashed', linewidth=4)
ax6.fill_between(RACgG_JT_mean.T.index, RACgG_JT_mean.T['R ACgG Thumb']-RACgG_t_sem.T['R ACgG Thumb'], RACgG_JT_mean.T['R ACgG Thumb']+RACgG_t_sem.T['R ACgG Thumb'], alpha=0.20, color='khaki')
ax6.yaxis.set_major_locator(MaxNLocator(3))
ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout(pad=2, h_pad=250)