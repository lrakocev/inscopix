import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import curve_fit
import itertools

def get_mean(cell_dict):
	vals = list(cell_dict.values())
	int_vals = [int(i) for i in vals if i != 'nan']
	return np.mean(int_vals)

def func(x, a, b, c):
	rew, cost = x
	rew_sq = np.array([val*val for val in rew])
	# (c*rew^2/2+rew)
	return (1 / (1 + np.exp(-a*(c*rew_sq/2+rew)+b))) * (1 / (1 + np.exp(-a*cost+b)))

def get_utility(rew, cost):
	rewards = np.array([1,2,3,4]*2)
	costs = np.array([1]*4 + [3]*4)

	ys = func((rewards,costs), 1, 0, 1)
	popt, pcov = curve_fit(func, (rewards,costs), ys)	
	utils = func((rewards,costs), *popt)
	
	xs = list(zip(rewards,costs))
	util_dict = dict(zip(xs, utils))

	rew = int(rew)
	cost = int(cost)

	return util_dict[(rew, cost)] if rew!=0 and cost!=0 else 0

## getting data	

if __name__ == "__main__":

	conn = psycopg2.connect(database='live_database', host='10.10.21.128', user='postgres', port='5432', password='1234')

	trial_cursor = conn.cursor()
	neuron_cursor = conn.cursor()

	selectTrialData = "SELECT cost_level, reward_level, decision_made, tasktypedone, sessionid, localindex FROM trial_table where tasktypedone!='L2'"
	selectNeuronData = "SELECT celltrace, tasktypedone, sessionid, localindex FROM inscopix_table where tasktypedone!='L2'"

	trial_cursor.execute(selectTrialData)
	neuron_cursor.execute(selectNeuronData)

	trial_table = trial_cursor.fetchall()
	neuron_table = neuron_cursor.fetchall()

	trial_df = pd.DataFrame(trial_table)
	neuron_df = pd.DataFrame(neuron_table)

	trial_df.columns = ['cost_level', 'reward_level', 'decision_made', 'tasktypedone', 'sessionid', 'localindex']
	neuron_df.columns=['celltrace', 'tasktypedone', 'sessionid', 'localindex']

	joint_df = pd.merge(trial_df, neuron_df, how='outer', on=['sessionid', 'localindex'])

	## cleaning up data 

	joint_df['reward_level'] = joint_df['reward_level'].fillna('0')
	joint_df['cost_level'] = joint_df['cost_level'].fillna('0')

	joint_df['utility'] = joint_df.apply(lambda x: get_utility(x['reward_level'], x['cost_level']),axis=1)
	joint_df= joint_df[joint_df['utility'] != 0]

	joint_df['celltrace'] = joint_df['celltrace'].apply(lambda x: yaml.load(x, Loader=yaml.Loader))
	joint_df['avg_celltrace'] = joint_df['celltrace'].apply(lambda x: get_mean(x))

	joint_df.to_csv("inscopix_df.xlsx")
