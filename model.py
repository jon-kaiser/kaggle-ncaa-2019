#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os

# open regular season detailed results file
RegDetails = pd.DataFrame(pd.read_csv('Stage2DataFiles/RegularSeasonDetailedResults.csv', index_col=[0,1,2,4]))
# open team lookup table
teams = pd.DataFrame(pd.read_csv('Stage2DataFiles/Teams.csv', index_col=0))

# To get all the games a team won in a given year:  RegDetails.loc[YYYY,:,TeamID,:]
# All the games a team lost in a given year:  RegDetails.loc[YYYY,:,:,TeamID]

wl = pd.DataFrame()
print("\nCalculating win/loss records and average points scored and allowed per team, per year (2003-2019)")
for y in range(2003,2020):
  print("Year {}".format(y), end='\r')
  for t in teams.index:
    if 2003 > teams.loc[t]['LastD1Season'] or 2019 < teams.loc[t]['FirstD1Season']:
      continue
    wins = RegDetails.loc[y,:,t,:]
    nwins = wins.shape[0]
    PointSc = wins['WScore']
    PointAll = wins['LScore']
    losses = RegDetails.loc[y,:,:,t]
    nlosses = losses.shape[0]
    PointSc = PointSc.append(losses['LScore'])
    PointAll = PointAll.append(losses['WScore'])
    if nwins == 0:
      wperc = 0
    else:
      wperc = nwins / (nwins + nlosses)
    wl = wl.append(pd.DataFrame([{'Year':y, 'TeamID':t, 'Wins': nwins, 'Losses': nlosses, 'WinPerc': wperc, 'PointScored': PointSc.mean(), 'PointAllowed': PointAll.mean()}]))

print("")
wl = wl.set_index(['Year','TeamID'])


# Now I have the win/loss record and average points scored and given up for each team in each season.

# Now go through each game and calculate how much more or less a team scored against the average the opponent allowed and vice versa see how many points they gave up relative to the amount of points the opponent scores on average. Then multiply each of those values by the opponents win percentage in an attempt to normalize those values by how good the opponent is (strength of schedule)

results = RegDetails.reset_index().set_index('Season')
ratings = wl.reindex(index=wl.index, columns=['Name', 'OffRate', 'DefRate'], fill_value=0)
print("Calculating offensive and defensive ratings per team, per year (2003-2019)")
for y in range(2003, 2020):
  print("Year {}".format(y), end='\r')
  for r in results.loc[y].itertuples():
    winid = r.WTeamID
    winscored = r.WScore
    loseid = r.LTeamID
    losescored = r.LScore
    wdiffscored = winscored - wl.loc[y,loseid]['PointAllowed']
    wdiffallowed = wl.loc[y,loseid]['PointScored'] - losescored
    winperc = wl.loc[y,winid]['WinPerc']
    ldiffscored = losescored - wl.loc[y,winid]['PointAllowed']
    ldiffallowed = wl.loc[y,winid]['PointScored'] - winscored
    loseperc = wl.loc[y,loseid]['WinPerc']
    if wdiffscored > 0:
      woffval = wdiffscored * loseperc
    else:
      woffval = wdiffscored * (1 - loseperc)
    if wdiffallowed > 0:
      wdefval = wdiffallowed * loseperc
    else:
      wdefval = wdiffallowed * (1 - loseperc)
    if ldiffscored > 0:
      loffval = ldiffscored * winperc
    else:
      loffval = ldiffscored * (1 - winperc)
    if ldiffallowed > 0:
      ldefval = ldiffallowed * winperc
    else:
      ldefval = ldiffallowed * (1-winperc)
    ratings.loc[y,winid] = [teams.loc[winid]['TeamName'], ratings.loc[y,winid].OffRate + woffval, ratings.loc[y,winid].DefRate + wdefval]
    ratings.loc[y,loseid] = [teams.loc[loseid]['TeamName'], ratings.loc[y,loseid].OffRate + loffval, ratings.loc[y,loseid].DefRate + ldefval]

print("")

# This will save the offensive and defensive ratings for every team from 2019. This can be used to examine and assess these ratings.
print("Saving Offensive and Defensive ratings for each team for the 2019 season")
if not os.path.isdir("SavedOutputs"):
  os.mkdir("SavedOutputs")
ratings.loc[2019].to_csv('SavedOutputs/TeamsRatings2019.csv')


# Maybe plot some of the ratings?


# Now create a training set. It will somehow consist of the offensive and defensive ratings and the result of the game (Win or Loss). My guess right now is that it will be the difference between Team1 offensive rating and Team2 defensive rating. And then Team1 defensive rating minus Team2 offensive rating.
# DiffOff = Team1Off - Team2Def
# DiffDef = Team1Def - Team2Off
#  DiffOff, DiffDef, Result
#  145.6, -98.5, L
#  -103.5, -68.2, L
# ...

print("Creating training set")
train = pd.DataFrame()
for y in range(2003, 2020):
  print("Year {}".format(y), end='\r')
  for r in results.loc[y].itertuples():
    offdiff = ratings.loc[y,r.WTeamID]['OffRate'] - ratings.loc[y,r.LTeamID]['DefRate']
    defdiff = ratings.loc[y,r.WTeamID]['DefRate'] - ratings.loc[y,r.LTeamID]['OffRate']
    if r.LTeamID < r.WTeamID:
      offdiff = offdiff * -1
      defdiff = defdiff * -1
      train = train.append(pd.DataFrame([{'Year':y, 'Team1':r.LTeamID, 'Team2': r.WTeamID, 'OffDiff': offdiff, 'DefDiff': defdiff, 'Result': 0}]))
    else:
      train = train.append(pd.DataFrame([{'Year':y, 'Team1':r.WTeamID, 'Team2': r.LTeamID, 'OffDiff': offdiff, 'DefDiff': defdiff, 'Result': 1}]))

print("")
train = train.set_index(['Year', 'Result'])


# Plot some of the training set data?

plt.scatter(train.loc[2019,1]['OffDiff'].values, train.loc[2019,1]['DefDiff'].values, c='g', alpha=0.1, label="Win")
plt.scatter(train.loc[2019,0]['OffDiff'].values, train.loc[2019,0]['DefDiff'].values, c='r', alpha=0.1, label="Loss")
plt.grid(True)
plt.ylabel("Def Rating Diff")
plt.xlabel("Off Rating Diff")
plt.title("2019 Training Data")
plt.legend()
plt.savefig("SavedOutputs/TrainResults2019.png")
# Show plot
# plt.show()


# Once that training set is made, create the model. Use Logistic Regression

train_x = train.iloc[:,0:2].values
year, result = zip(*train.index.values)
train_y = list(result)
model = LogisticRegression()
model.fit(train_x, train_y)


# Once the model is made and the parameters calculated, use the model to run the test data.
# In the first part of the competition we use the model to predict past tournament games from (2014-2018). Since we already know the results of those competitions we can build and train and optimize the model. Reminder, if we do this, we can't train the model using whichever year we're testing on. So if we're testing the 2014 year, we have to leave 2014 training data out of the set, etc. for 2015, 2016...
# The first stage came and went. We didn't have time to test on completed games and then tweak the model and optimize it. Oh well, full steam ahead.

# The second stage of the competition will begin on Sun March 17 and submission ends on Thur March 21. That is the time between the announcement of all the teams that will compete in the 2019 tournament and the beginning of the tournament. In this stage, we predict the probability of winning of each matchup between every team.

# tournament teams by year
tt = pd.DataFrame(pd.read_csv('Stage2DataFiles/NCAATourneySeeds.csv', index_col=0))
# Just 2019 tourney teams
ttarray = tt.loc[2019]['TeamID'].values
ttarray.sort()
for i in range(0,ttarray.size):
  for j in range(i+1,ttarray.size):
# Get Off and Def diffs for these two teams
    test_offdiff = ratings.loc[2019,ttarray[i]]['OffRate'] - ratings.loc[2019,ttarray[j]]['DefRate']
    test_defdiff = ratings.loc[2019,ttarray[i]]['DefRate'] - ratings.loc[2019,ttarray[j]]['OffRate']
    prob = model.predict_proba(np.array([[test_defdiff, test_offdiff]]))
    print("2019_{}_{},{:.4f}".format(ttarray[i],ttarray[j], prob[0,1]))
#    print("2019_{}_{},{:.4f} \t {}-{}".format(ttarray[i],ttarray[j], prob[0,1], teams.loc[ttarray[i]]['TeamName'], teams.loc[ttarray[j]]['TeamName']))





# Predict probabilities
# r = model.predict_proba(np.array([[100,100],[-100,-100],[100,-100],[-100,100],[0,0]]))

# Finito. Give us our prize money NOW!!!

