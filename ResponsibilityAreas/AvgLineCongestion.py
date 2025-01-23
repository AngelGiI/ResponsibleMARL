################################################################################################
###    This file is where the average line congestion parameter is computed.                 ###
#                                                                                              #
#  It runs the DoNothingAgent over several scenarios while keeping track of the congestion     #
#  of each line. It could easily be modified so that it writes into an output file instead     #
#  of printing.                                                                                #
#                                                                                              #   
#  You can simply run the code to see what it does.                                            #
################################################################################################

import os
import re
import json
import numpy as np
import grid2op
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from grid2op.PlotGrid import PlotMatplot
from grid2op.Agent import DoNothingAgent

# Constants and configurations
display_tqdm = False
CASE_NAME = "l2rpn_case14_sandbox"
CHRONIC_START_IDX = 3
CHRONIC_STEP = 40
CHRONIC_END_IDX = 40 * 26

# Initialize environment and agent
env = grid2op.make(CASE_NAME)
plot_helper = PlotMatplot(env.observation_space)
my_agent = DoNothingAgent(env.action_space)
data_path = os.path.join("..", "data", CASE_NAME)

# Set chronics
chronics_used = list(range(CHRONIC_START_IDX, CHRONIC_END_IDX, CHRONIC_STEP))

# Load chronics
env.chronics_handler.set_filter(lambda path: re.match(".*(%s)$" % "|".join(map(str, chronics_used)), path) is not None)
env.reset()

# Main simulation loop
all_rhos, all_gens, all_loads = [], [], []

print("Total maximum number of timesteps possible: {}".format(env.chronics_handler.max_timestep()))
for chronic in chronics_used:
    obs = env.reset()
    all_rhos.append(obs.rho)
    reward = env.reward_range[0]
    done = False
    nb_step = 0

    with tqdm(total=env.chronics_handler.max_timestep(), disable=not display_tqdm) as pbar:
        while True:
            action = my_agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            pbar.update(1)
            if done:
                break
            all_rhos.append(obs.rho)
            all_gens.append(obs.gen_p)
            all_loads.append(obs.load_p)
            nb_step += 1

    print(f"Chronic {chronic}: Number of timesteps computed: {nb_step}")

print("Total observations: {}".format(len(all_rhos)))

# Convert to a numpy array for easier manipulation
all_rhos_array = np.array(all_rhos)
all_gens = np.array(all_gens)
all_loads = np.array(all_loads)

# Calculate the average rho per power line
average_rhos_per_line = np.mean(all_rhos_array, axis=0)
average_gen = np.mean(all_gens, axis=0)
average_load = np.mean(all_loads, axis=0)


print("Average rho per power line: {}".format(average_rhos_per_line))
print("Average rho overall: {}".format(np.mean(all_rhos_array)))
print("Maximum rho: {}".format(np.max(all_rhos_array)))
print("")
print("Average generation per timestep: {}".format(np.mean(average_gen)))
print("Average load per timestep: {}".format(np.mean(average_load)))