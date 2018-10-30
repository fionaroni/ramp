#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:39:58 2018

@author: uber
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 15)

# =============================================================================
# Read in raw data, clean for processing
# =============================================================================
# sfdc
xls = pd.ExcelFile("finalreport1539987601573_resaved.xls")
sfdc = xls.parse() # report pulled from SFDC- all CW and FT from 4/30/18-10/19/2018

# sfdc - drop needless columns
#sfdc.drop(sfdc.columns[[0,2]], axis=1, inplace=True) # remove empty, emp type, antic start

xls = pd.ExcelFile("Copy of EXT HC Tracker.xlsx")
reps = xls.parse('Sales EXT Roster')
# reps - drop needless columns
to_drop = reps.columns[11:]
reps.drop(reps.columns[[0,1,3,7]], axis=1, inplace=True) # remove first, last, emp type, antic start
reps.drop(to_drop, axis=1, inplace=True)
print("reps cleaned: ", reps.shape) # (237, 6)

# print col headers for each dataset
print("reps cols: ", list(reps.columns.values))
#[Full Name', 'Office Location', 'Uber Email', 'Rep Status', 'Actual Start Date', 'End Date']
print("sfdc cols: ", list(sfdc.columns.values))
#['Opportunity Name', 'Opportunity Owner Email', 'Account Name', 'Close Date', 'Eats First Trip Date', 'Days from Closed Won to First Trip', 'Opportunity Owner', 'Stage']

# =============================================================================
# SFDC- boolean mask to drop names that aren't in reps df; we want to analyze only the records with start/end dates
# =============================================================================
print("before drop rows: ", sfdc.shape) # (50336, 9)
sfdc_onlyreps = sfdc[sfdc["Opportunity Owner Email"].isin(reps['Uber Email'])]
print("after drop rows: ", sfdc_onlyreps.shape) # (2098, 9)

# =============================================================================
# Merge sfdc_onlyreps and reps
# =============================================================================
# Use RA as lookup (Uber Email) to add new columns to sfdc_onlyreps
result = pd.merge(sfdc_onlyreps, reps[['Office Location', 'Uber Email', 'Rep Status', 'Actual Start Date', 'End Date']], how='left', left_on="Opportunity Owner Email", right_on="Uber Email")
print("shape after merge: ", result.shape) # (2098, 14)
print("cols after merge: ", list(result.columns.values))

# =============================================================================
# Pre-processing & cleaning before analysis 
# =============================================================================
# if "Rep Status" column is "Active," fill End Date column with today's date
result.loc[result['Rep Status'] == 'Active', 'End Date'] = pd.datetime.now().date()

# calculate Week Number that Close Date falls in
result["Close Date"] = pd.to_datetime(result["Close Date"])
result["Days bt Start+Close"] = result["Close Date"] - result["Actual Start Date"]
JustDays = result["Days bt Start+Close"].dt.days #extract just the day, float to int
result["Week #"] = JustDays/7
result["Week #"] = result["Week #"].astype(int)
       
# =============================================================================
# Analysis - All Reps
# =============================================================================

# Calculate #CW per rep
result["Total CW/rep"] = result.groupby("Opportunity Owner Email")["Close Date"].transform("count")

# Calculate total CW for that rep corresponding to the Week #
per_week = result.groupby(['Week #','Office Location','Opportunity Owner Email']).count()['Close Date']
# Series to df
#print("type of per_week: ", type(per_week))
per_week = per_week.to_frame()
print("type of per_week: ", type(per_week))
per_week.reset_index(inplace=True)
#print(per_week)

# line: median CW per rep per week per city
medianCW_city = per_week.groupby(['Week #','Office Location']) \
                                 .agg({'Close Date':'median'})
medianCW_city.rename(columns={'Close Date':'median_cw'}, inplace=True)
print("after groupby:", medianCW_city)
print(medianCW_city.columns.values, medianCW_city.shape)    
medianCW_city.reset_index(inplace=True) 
print(medianCW_city.columns.values, medianCW_city.shape)    

fig, ax = plt.subplots(1,1, figsize=(20,10))
for n, group in medianCW_city.groupby('Office Location'):
    group.plot(x='Week #', y='median_cw', ax=ax, label=n)
ax.set_xlim(xmin=0)
plt.legend(prop={'size': 15})
plt.title('Median CW per rep per city')
plt.xticks(np.arange(1,medianCW_city['Week #'].max()))
plt.ylabel("CW")
plt.savefig('allreps_WoWcitycut.png')
plt.show()
plt.close()   

# boxplot: #CW per week
#fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20,15))
fig, ax1 = plt.subplots(1,1,figsize=(20,15))
sns.boxplot(x="Week #", y="Close Date", data=per_week, ax=ax1).set_title('WoW median CW')
ax1.set_ylabel("CW")
#sns.swarmplot(x="Week #", y="Close Date", data=per_week, color=".25")
medians = per_week.groupby(['Week #'])['Close Date'].median().values
median_labels = [str(np.round(s, 2)) for s in medians]

pos = range(len(medians))
for tick,label in zip(pos,ax1.get_xticklabels()):
    ax1.text(pos[tick], medians[tick] + 0.5, median_labels[tick], 
            horizontalalignment='center', size='large', color='k', weight='semibold')
plt.savefig('allreps_WoWmedianCW.png')
plt.show()
plt.close()

## boxplot: cumulative CW over time 
#sns.boxplot(x="Week #", y="Total CW/rep", hue="Rep Status", data=result, palette="Set3", ax=ax2).set_title('WoW cumulative CW')
#ax2.set_ylabel("Cumulative CW")
#plt.show()
#plt.close()

# =============================================================================
# Analysis - Rep Status: Active, Lead Source: excludes Inbounds
# =============================================================================
# prep result_actives
result_actives = result[(result['Lead Source'].isin(['Outbound','Outbound AE','Cold',pd.np.nan])) & (result['Rep Status'] == 'Active')]
print("unique values in Rep Status col: ", result_actives['Rep Status'].unique())
print("unique values in Lead Source col: ", result_actives['Lead Source'].unique())
result_actives["Total CW/rep"] = result.groupby("Opportunity Owner Email")["Close Date"].transform("count")

per_week2 = result_actives.groupby(['Week #','Office Location','Opportunity Owner Email']).count()['Close Date']
per_week2 = per_week2.to_frame()
per_week2.reset_index(inplace=True)
medianCW_city2 = per_week2.groupby(['Week #','Office Location']).agg({'Close Date':'median'})
medianCW_city2.rename(columns={'Close Date':'median_cw'}, inplace=True)
print("after groupby:", medianCW_city2)
print(medianCW_city2.columns.values, medianCW_city2.shape)    
medianCW_city2.reset_index(inplace=True) 
print(medianCW_city2.columns.values, medianCW_city.shape)
 
# boxplot: median CW per week (active reps, no inbounds)
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20,10))
sns.boxplot(x="Week #", y="Close Date", data=per_week2, ax=ax1).set_title('WoW median CW - Rep Status: Active, Lead Source: excludes Inbounds')
ax1.set_ylabel("median CW")
#sns.swarmplot(x="Week #", y="Close Date", data=per_week, color=".25")
medians2 = per_week2.groupby(['Week #'])['Close Date'].median().values
median_labels2 = [str(np.round(s, 2)) for s in medians2]
pos = range(len(medians2))
for tick,label in zip(pos,ax1.get_xticklabels()):
    ax1.text(pos[tick], medians2[tick] + 0.5, median_labels2[tick], 
            horizontalalignment='center', size='medium', color='k', weight='semibold')
                       
# boxplot: mean CW per week (active reps, no inbounds)
sns.boxplot(x="Week #", y="Close Date", data=per_week2, ax=ax2).set_title('WoW mean CW - Rep Status: Active, Lead Source: excludes Inbounds')
ax2.set_ylabel("mean CW")
means = per_week2.groupby(['Week #'])['Close Date'].mean().values 
mean_labels = [str(np.round(s, 2)) for s in means]
pos = range(len(means))
for tick,label in zip(pos,ax2.get_xticklabels()):
    ax2.text(pos[tick], means[tick] + 0.5, mean_labels[tick], 
            horizontalalignment='center', size='medium', color='k', weight='semibold')
plt.savefig('active-no-inbounds_WoW.png')
plt.show()
plt.close()

# =============================================================================
# Points Analysis - Rep Status: Active, Lead Source: exclude nan, Cold, Referral
# Assign points for Inbound (0.8) vs. Outbound leads (1.0)
# =============================================================================
print("Points assignment for Inbound leads versus Outbound leads")

# Rep Status: Active, Lead Source: exclude nan, Cold, Referral
exclude_leads = result[(result['Lead Source'].isin(['Outbound','Inbound: Website',\
                        'Inbound: Self Sign-Up','Inbound: Other','Outbound AE',\
                        'Inbound: SelfServe'])) & (result['Rep Status'] == 'Active')]
print("unique values in Lead Source col: ", exclude_leads['Lead Source'].unique())

# broadcast function - 'Lead Source' column 
def assign_points(df):
    if df['Lead Source'] == 'Outbound' or df['Lead Source'] == 'Outbound AE':
        return 1
    else: # Inbound: Website, SSU, Other, SelfServe
        return 0.8

exclude_leads['Points'] = exclude_leads.apply(assign_points, axis=1)

# calculate points per head
points_groupby = exclude_leads.groupby(['Week #','Opportunity Owner Email']).sum()['Points']
points_df = points_groupby.to_frame()
points_df.reset_index(inplace=True)

# boxplot: median points per week (active reps, no inbounds)
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20,10))
sns.boxplot(x="Week #", y="Points", data=points_df, ax=ax1).set_title('WoW medians Points per head per week')
ax1.set_ylabel("median CW")
pt_medians = points_df.groupby(['Week #'])['Points'].median().values
pt_median_labels = [str(np.round(s, 2)) for s in pt_medians]
pos = range(len(pt_medians))
for tick,label in zip(pos, ax1.get_xticklabels()):
    ax1.text(pos[tick], pt_medians[tick] + 0.5, pt_median_labels[tick], 
            horizontalalignment='center', size='medium', color='k', weight='semibold')
                       
# boxplot: mean points per week (active reps, no inbounds)
sns.boxplot(x="Week #", y="Points", data=points_df, ax=ax2).set_title('WoW mean points per head per week')
ax2.set_ylabel("mean CW")
pt_means = points_df.groupby(['Week #'])['Points'].mean().values 
pt_mean_labels = [str(np.round(s, 2)) for s in pt_means]
pos = range(len(pt_means))
for tick,label in zip(pos, ax2.get_xticklabels()):
    ax2.text(pos[tick], pt_means[tick] + 0.5, pt_mean_labels[tick], 
            horizontalalignment='center', size='medium', color='k', weight='semibold')
plt.savefig('active-points_WoW.png')
plt.show()
plt.close()



# =============================================================================
# First Trips
# =============================================================================

# process - convert FT date from str to datetime
result["Eats First Trip Date"] = pd.to_datetime(result["Eats First Trip Date"])

# 574 of 1466 records did not have FT (39%)
result["Eats First Trip Date"].isna().sum() # 574

# assuming FT occurred, how long between CW and FT?
only_ft = result[result["Eats First Trip Date"].notnull()] # remove all NaT values
print(len(only_ft)) # 574
cw_ft = only_ft[only_ft["Days from Closed Won to First Trip"] >= 0]
# histogram: distribution of days b/t CW and FT, assuming CW occur before FT
cw_ft["Days from Closed Won to First Trip"].hist(bins=35)
plt.savefig('FTdays.png')
plt.show()
plt.close()
print("mean: ", cw_ft["Days from Closed Won to First Trip"].mean())
print("median: ", cw_ft["Days from Closed Won to First Trip"].median())
# histogram: distribution of days b/t CW and FT, by Rep Status
cw_ft["Days from Closed Won to First Trip"].hist(by=result['Rep Status'], bins=25, stacked=True)
plt.savefig('FTdays_active-departed.png')
plt.show()
plt.close()
cw_ft_reps = cw_ft.groupby('Rep Status').agg({'Days from Closed Won to First Trip':[np.mean, np.median]})
print(cw_ft_reps)

# =============================================================================
# future study-- why are some Close Dates after FT?
# new_result = result[result["Close Date"] > result["Eats First Trip Date"]]
#print(len(new_result)) 
# =============================================================================


# =============================================================================
# # duplicate code for FTE - pending Gabe's sheets
# =============================================================================


# final df
today = pd.datetime.now().date()
result.to_csv(f'result.csv', index=False)
per_week.to_csv(f'per_week.csv', index=False)