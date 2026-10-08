from pathlib import Path
import pandas as pd, numpy as np
p=Path(__file__).resolve().parent; (p/'data').mkdir(parents=True,exist_ok=True)
base=pd.read_csv(p.parent/'Linear_Regression_2.1'/'data'/'Data2.1.csv')
raw=pd.read_csv(p.parent/'Data2.2.csv')
# Align records by team and opponent, using the source's unique team-match records.
assert not raw.duplicated(['Team','Opponent']).any(), 'Ambiguous repeated matchups: use date keys'
h=raw.set_index(['Team','Opponent']); a=raw.set_index(['Opponent','Team'])
keys=pd.MultiIndex.from_frame(base[['home_team','away_team']]); H=h.reindex(keys); A=a.reindex(keys)
assert H.notna().all().all() and A.notna().all().all()
# Home team's conceded average is recorded as Opp Avg Conceded in the away-team row.
# The corresponding recent conceded average is likewise in the away-team row.
new=base[['home_team','away_team','date']].copy()
new['FIFA Points Difference']=H['FIFA points'].to_numpy()-A['FIFA points'].to_numpy()
new['Avg Scored Difference']=H['Avg Scored'].to_numpy()-A['Avg Scored'].to_numpy()
new['Avg Conceded Difference']=A['Opp Avg Conceded'].to_numpy()-H['Opp Avg Conceded'].to_numpy()
new['Win Rate Difference']=H['Win Rate'].to_numpy()-A['Win Rate'].to_numpy()
new['Last 5 Avg Scored Difference']=H['Last 5 Avg Scored'].to_numpy()-A['Last 5 Avg Scored'].to_numpy()
new['Last 5 Avg Conceded Difference']=A['Last 5 Opp Avg Conceded'].to_numpy()-H['Last 5 Opp Avg Conceded'].to_numpy()
new['Rest Days Difference']=H['Rest Days'].to_numpy()-A['Rest Days'].to_numpy()
new['Home FIFA Points']=H['FIFA points'].to_numpy()
new['Goal Difference']=H['Goals'].to_numpy()-A['Goals'].to_numpy()
assert len(new)==104 and new.notna().all().all() and (new['Goal Difference']==base['Goal Difference']).all()
new.to_csv(p/'data'/'Data2.1.csv',index=False)
print('Created',len(new),'rows with',len(new.columns)-4,'predictors')
print(new.corr(numeric_only=True)['Goal Difference'].round(3).to_string())
