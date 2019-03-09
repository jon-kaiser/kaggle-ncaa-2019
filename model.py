#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

# open regular season detailed results file
RegDetails = pd.DataFrame(pd.read_csv('DataFiles/RegularSeasonDetailedResults.csv', index_col=[0,1,2,4]))
# open team lookup table
teams = pd.DataFrame(pd.read_csv('DataFiles/Teams.csv', index_col=0))

# To get all the games a team won in a given year:  RegDetails.loc[YYYY,:,TeamID,:]
# All the games a team lost in a given year:  RegDetails.loc[YYYY,:,:,TeamID]

wl = pd.DataFrame()
print("\nCalculating win/loss records and average points scored and allowed per team, per year (2003-2018)")
for y in range(2003,2019):
  print("Year {}".format(y), end='\r')
  for t in teams.index:
    if 2003 > teams.loc[t]['LastD1Season'] or 2018 < teams.loc[t]['FirstD1Season']:
      continue
    wins = RegDetails.loc[y,:,t,:]
    nwins = wins.shape[0]
    PointSc = wins['WScore']
    PointAll = wins['LScore']
    losses = RegDetails.loc[y,:,:,t]
    nlosses = losses.shape[0]
    if nwins == 0:
      wperc = 0
    else:
      wperc = nwins / (nwins + nlosses)
    PointSc = PointSc.append(losses['LScore'])
    PointAll = PointAll.append(wins['WScore'])
    wl = wl.append(pd.DataFrame([{'Year':y, 'TeamID':t, 'Wins': nwins, 'Losses': nlosses, 'WinPerc': wperc, 'PointScored': PointSc.mean(), 'PointAllowed': PointAll.mean()}]))

print("")
wl = wl.set_index(['Year','TeamID'])


# Now I have the win/loss record and average points scored and given up for each team in each season.

# Now go through each game and calculate how much more or less a team scored against the average the opponent allowed and vice versa see how many points they gave up relative to the amount of points the opponent scores on average. Then multiply each of those values by the opponents win percentage in an attempt to normalize those values by how good the opponent is (strength of schedule)

results = RegDetails.reset_index().set_index('Season')
ratings = wl.reindex(index=wl.index, columns=['Name', 'OffRate', 'DefRate'], fill_value=0)
print("Calculating offensive and defensive ratings per team, per year (2003-2018)")
for y in range(2003, 2019):
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

# This will save the offensive and defensive ratings for every team from 2018. This can be used to examine and assess these ratings.
print("Saving Offensive and Defensive ratings for each team for the 2018 season")
if not os.path.isdir("SavedOutputs"):
  os.mkdir("SavedOutputs")
ratings.loc[2018].to_csv('SavedOutputs/TeamsRatings2018.csv')


# Maybe plot some of the ratings?


# Now create a training set. It will somehow consist of the offensive and defensive ratings and the result of the game (Win or Loss). My guess right now is that it will be the difference between Team1 offensive rating and Team2 defensive rating. And then Team1 defensive rating minus Team2 offensive rating.
# DiffOff = Team1Off - Team2Def
# DiffDef = Team1Def - Team2Off
#  DiffOff, DiffDef, Result
#  145.6, -98.5, L
#  -103.5, -68.2, L
# ...


# Maybe plot some of the training set data?



# Once that training set is made, create the model. Use Logistic Regression?



# Once the model is made and the parameters calculated, use the model to run the test data.
# In the first part of the competition we use the model to predict past tournament games from (2014-2018). Since we already know the results of those competitions we can build and train and optimize the model. Reminder, if we do this, we can't train the model using whichever year we're testing on. So if we're testing the 2014 year, we have to leave 2014 training data out of the set, etc. for 2015, 2016...

# One of the components here is some infrastructure to read in the list of the teams that are in the tournament. And then run the model for each permutation of team vs team. And then we need some code built to save the output.

# The second stage of the competition will begin on Sun March 17 and submission ends on Thur March 21. That is the time between the announcement of all the teams that will compete in the 2019 tournament and the beginning of the tournament. In this stage, we predict the probability of winning of each matchup between every team.


# Finito. Give us our prize money NOW!!!

