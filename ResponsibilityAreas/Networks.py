################################################################################################
###    This file is an initial attempt to automate the cluster visualization for any grid    ###
# The clusters are computed using one run of spectral clustering with the action+congestion    #
# weights. Everything is hardcoded for now, but can be easily modified to take in the clusters #
# and zero_action_substations as input or compute on the fly. "zero_action_substations" are    #
# those masked out.                                                                            #
#                                                                                              #
#  The function "plot_clusters" is custom.                                                     #
#                                                                                              #   
# You can simply run the code to see what it does.                                             #
################################################################################################

import grid2op
from grid2op.PlotGrid import PlotMatplot
import matplotlib.pyplot as plt

### Optional: check available environments
#print("local envs:", grid2op.list_available_local_env())
#print("remote envs:", grid2op.list_available_remote_env())
#print("test envs:", grid2op.list_available_test_env())

### 3 ENVIRONMENTS chosen: 14, 36, 118 SUBSTATIONS ###
env = ['l2rpn_case14_sandbox', 'l2rpn_icaps_2021_small', 'l2rpn_neurips_2020_track2_small']

average_congestion_case14 = [0.31120053, 0.3033225, 0.30382577, 0.23219728, 0.67197347, 0.19390275,
    0.40178862, 0.64066064, 0.5332323,  0.7970488, 0.14247213, 0.31393358, 0.36672422, 
    0.47334263, 0.4611472,  0.43447793, 0.41216594, 0.5247531, 0.46909025, 0.4043246]

### Plotting the layout and congestion of the grid ###
# fig = plot_helper.plot_layout()
# plt.show()

# fig = plot_helper.plot_avg_congestion(average_congestion_case14)
# plt.show()

### Results from other trials ###
## 36 SUBSTATIONS ##
# zero_action_substations = [0, 2, 3, 5, 6, 8, 10, 11, 15, 17, 19, 20, 24, 25, 30, 31, 34]

# (Partition into) 4 CLUSTERS 
# clusters = [[0, 2, 4, 5, 6, 7, 8, 9], [1, 3, 10, 11, 12, 13, 14, 15, 16, 17, 35], [18, 19, 20, 21, 22, 23, 24, 25], [26, 27, 28, 29, 30, 31, 32, 33, 34]]

## 118 SUBSTATIONS ##
zero_action_substations = [ 0 ,1 , 5 , 6 , 8 , 9 ,12 ,13 ,15 ,19 ,20 ,21 ,25 ,27 ,28 ,32 ,34 ,35
                            ,37 ,38 ,40 ,42 ,43 ,47 ,49 ,51 ,52 ,56 ,57 ,62 ,63 ,66 ,70 ,71 ,72 ,73
                            ,77 ,78 ,80 ,83 ,85 ,86 ,87 ,90 ,92 ,94 ,96 ,97 ,98,100,101,107,108,110
                            ,111,113,114,115,116,117]
# 4 CLUSTERS
#clusters = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 112, 113, 114, 116], 
            # [23, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 117], 
            # [67, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 115], 
            # [102, 103, 104, 105, 106, 107, 108, 109, 110, 111]]

# 7 CLUSTERS
#clusters = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 116], [16, 17, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 112, 113, 114], [18, 19, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], [23, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 115, 117], [44, 45, 46, 47, 48, 64], [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66], [99, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]]

# 15 CLUSTERS
clusters = [[0, 1, 2, 3, 6, 10, 11, 12, 13, 14, 15, 16, 116], [4, 5, 7, 8, 9], [17, 18, 19, 33, 35, 42], [20, 21, 22, 23, 70, 71, 72], [24, 25, 29, 32, 34, 36, 37, 38, 39, 40, 41], [26, 27, 28, 30, 31, 112, 113, 114], [43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57], [58, 59, 60, 61, 62, 63, 64, 65, 66], [67, 78, 79, 80, 95, 96, 97, 98, 99, 100, 115], [68, 69, 73, 74, 75, 76, 77, 117], [81, 82, 83, 84, 85, 86, 87, 88], [89, 90, 91, 92, 93, 94, 101], [102, 103, 104, 105, 106, 107], [108, 109, 110, 111]]

env = grid2op.make(env[2])
plot_helper = PlotMatplot(env.observation_space)
fig = plot_helper.plot_clusters(clusters, zero_action_substations)