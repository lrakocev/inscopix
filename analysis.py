import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import curve_fit
import itertools

joint_df = pd.read_csv('inscopix_df.xlsx')
joint_df['celltrace']=joint_df['celltrace'].apply(eval)

fin_df = joint_df.join(pd.DataFrame(joint_df.pop('celltrace').values.tolist()))

cols = [i for i in fin_df.columns if 'C' in str(i)]
num_neurons = 1
col_combos = [('C04', 'C12', 'C2', 'C07', 'C13')] # [c for c in itertools.combinations(cols, num_neurons)]

## plotting + helpers

#for i in range(0,len(col_combos)+1):

i = 0 
neurons = list(col_combos[i])
avg_name = 'avg_' + str(i)

fin_df[avg_name] = fin_df[neurons].mean(axis=1)

avg_util = fin_df.groupby(['utility'])[avg_name].mean()
avg_util = avg_util.reset_index()

serr_util = fin_df.groupby(['utility'])[avg_name].sem()
serr_util = serr_util.reset_index()

cost_1_df = fin_df[fin_df['cost_level'] == 1]
cost_3_df = fin_df[fin_df['cost_level'] == 3]

avg_rew_cost_1 = cost_1_df.groupby(['reward_level'])[avg_name].mean()
avg_rew_cost_1  = avg_rew_cost_1.reset_index()

serr_rew_cost_1 = cost_1_df.groupby(['reward_level'])[avg_name].sem()
serr_rew_cost_1 = serr_rew_cost_1.reset_index()

avg_rew_cost_3 = cost_3_df.groupby(['reward_level'])[avg_name].mean()
avg_rew_cost_3  = avg_rew_cost_3.reset_index()

serr_rew_cost_3 = cost_3_df.groupby(['reward_level'])[avg_name].sem()
serr_rew_cost_3 = serr_rew_cost_3.reset_index()


avg_cost = fin_df.groupby(['cost_level'])[avg_name].mean()
avg_cost = avg_cost.reset_index()

serr_cost = fin_df.groupby(['cost_level'])[avg_name].sem()
serr_cost = serr_cost.reset_index()

title = "".join(neurons)
plt.figure(1)
# utility against trace
plt.plot(avg_util['utility'], avg_util[avg_name])
plt.errorbar(avg_util['utility'], avg_util[avg_name], serr_util[avg_name])
plt.xlabel('utility')
plt.ylabel('avg cell trace across selected neurons + trials')
plt.title(title)
plt.legend()
plt.savefig("fig_1",format="pdf")

plt.figure(2)

title = "LOW COST: " + "".join(neurons)
# utility against trace
plt.plot(avg_rew_cost_1['reward_level'], avg_rew_cost_1[avg_name])
plt.errorbar(avg_rew_cost_1['reward_level'], avg_rew_cost_1[avg_name], serr_rew_cost_1[avg_name])
plt.xlabel('reward level')
plt.ylabel('avg cell trace across selected neurons + trials')
plt.title(title)
plt.legend()
plt.savefig("fig_2",format="pdf")

plt.figure(3)

title = "HIGH COST: " + "".join(neurons)
# utility against trace
plt.plot(avg_rew_cost_3['reward_level'], avg_rew_cost_3[avg_name])
plt.errorbar(avg_rew_cost_3['reward_level'], avg_rew_cost_3[avg_name], serr_rew_cost_3[avg_name])
plt.xlabel('reward level')
plt.ylabel('avg cell trace across selected neurons + trials')
plt.title(title)
plt.legend()
plt.savefig("fig_3",format="pdf")


plt.figure(4)

title = "COST BARS: " + "".join(neurons)
# utility against trace
plt.bar(avg_cost['cost_level'], avg_cost[avg_name])
plt.errorbar(avg_cost['cost_level'], avg_cost[avg_name], serr_cost[avg_name])
plt.xlabel('cost level')
plt.ylabel('avg cell trace across selected neurons + trials')
plt.title(title)
plt.legend()
plt.savefig("fig_4",format="pdf")
