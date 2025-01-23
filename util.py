import os
import random
import numpy as np
import csv
import json
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from collections import defaultdict
from statistics import mean, stdev
import re
from scipy import stats

colors = [i for i in mcolors.TABLEAU_COLORS.keys()]


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(list(self.memory), n)


def show_results(res_dir, case, n, name=""):
    """
    This function plots the test results that are saved during the training of the agent. It shows the result
    for each seed in [0-n].
    Next to a plot the output will give a pandas dataframe with a summary of the results with the following columns:
    Maximum steps,
        The maximum number of steps this agent was able to achieve in an episode (max 864)
    Agent updates (steps),
        After how many updates of the agent (training iterations) was the agent able to achieve this maximum number
        of steps.
    Best mean,
        Best mean score that the agent got during the training.
    Agent updates (mean)
        After how many updates of the agent (training iterations) was the agent able to achieve this best mean score.

    Parameters
    ----------

    res_dir: ``str`` or path
        Name of the directory where the results of a trained agents are saved.
    case: ``str``
        Name of the agents e.g. "case_5_ppo"
    n: ``int``
        number of seeds for which this agent is trained. The results will be plotted per seed: e.g.
        case_5_ppo_0 ... case_5_ppo_4
    name: ``str``
        Optional. This will be the last part of the title of the plot, which is of the form
        "Training *agent* - *case* *name*"

    Example
    ----------
    case = 'final_case5_sacd'
    seeds = 5
    summary, res_sacd = show_results(res_dir, case, seeds, name='')

    """
    df = pd.DataFrame()
    if type(n) is int:
        seeds = range(n)
    else:
        seeds = n
        n = len(n)

    best_steps = np.zeros(n)
    best_mean = np.zeros(n)
    first = np.zeros(n)
    it = np.zeros(n)

    script_path = os.path.join(res_dir, f"{case}_{seeds[0]}")
    with open(os.path.join(script_path, "param.json"), "r") as f:
        params = json.load(f)

    # params_to_print = ["name", "batch_size", "update_start", "gamma", "lr"]
    # [print(f"{key}:\t {params.get(key)}") for key in params_to_print]
    [print(f"{key}:\t {params.get(key)}") for key in params.keys()]

    name = f"{params['agent'].upper()} - case {params['case']} {name}"
    for i, seed in enumerate(seeds):
        steps = pd.read_csv(f"{res_dir}/{case}_{seed}/step.csv", names=range(5))
        score = pd.read_csv(f"{res_dir}/{case}_{seed}/score.csv")
        sem = score.iloc[:, 1:].sem(axis=1)
        mean = score.iloc[:, 1:].mean(axis=1)
        plt.fill_between(
            score["env_interactions"], mean - sem, mean + sem, interpolate=True, color=colors[seed], alpha=0.1
        )
        plt.plot(score["env_interactions"], mean, label=f"seed {seed}", color=colors[seed])
        df["env_interactions"] = score["env_interactions"]
        df[f"mean s{i}"] = mean
        best_steps[i] = steps.mean(axis=1).max()
        first[i] = steps.mean(axis=1).argmax()
        best_mean[i] = mean.max()
        it[i] = mean.argmax()

    df["env_interactions"] = score["env_interactions"]
    summary = pd.DataFrame(
        {"Max steps": best_steps, "Agent updates (steps)": first, "Best mean": best_mean, "Agent updates (mean)": it}
    )
    summary.index.name = "seed"

    plt.title(label=f"Training {name}", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=24)
    plt.ylim(ymin=-100, ymax=105)
    plt.xlim(xmin=0, xmax=params["nb_steps"])
    plt.xlabel("environment interactions", fontsize=16)
    plt.ylabel("mean score", fontsize=16)
    plt.legend(fontsize=18, loc="lower right")
    f = plt.gcf()
    f.set_size_inches(15, 7)
    return summary, df


def plot_fullep_score(res_dir, case, n, name="", nb_steps=1000):
    df = pd.DataFrame()
    seeds = range(n)
    for i, seed in enumerate(seeds):
        full_score = pd.read_csv(f"{res_dir}/{case}_{seed}/full_score.csv")
        sem = full_score.iloc[:, 1:].sem(axis=1)
        mean = full_score.iloc[:, 1:].mean(axis=1)
        df["env_interactions"] = full_score["env_interactions"]
        df[f"mean s{i}"] = mean
        plt.fill_between(
            full_score["env_interactions"], mean - sem, mean + sem, interpolate=True, color=colors[seed], alpha=0.1
        )
        plt.plot(full_score["env_interactions"], mean, label=f"seed {seed}", color=colors[seed])

    plt.hlines(80, xmin=0, xmax=nb_steps, label=f"Completed Episode Score")
    plt.title(label=f"Training {name}", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=24)
    plt.ylim(ymin=-100, ymax=105)
    plt.xlim(xmin=0, xmax=nb_steps)
    plt.xlabel("environment interactions", fontsize=16)
    plt.ylabel("mean score", fontsize=16)
    plt.legend(fontsize=18, loc="lower right")
    f = plt.gcf()
    f.set_size_inches(15, 7)
    return df


def compare_results(
    cases,
    list_data_frames,
    style="default",
    nb_steps=1000,
    ymin=-105,
    ymax=105,
    ylabel="mean score",
    title="Training Scores",
    window_size=10,  # Add window size parameter for smoothing
    max_colors=10  # Add max colors parameter
):
    """
    With this function you can plot the results of multiple different agents.

    Parameters
    ----------
    cases: ``list``
        A list of strings that will represent the labels in the plot.
    list_data_frames: ``list``
        A list of data frames that should correspond to the labels in cases. Note that we need
        size(cases) == size(list_data_frames)
    style: ``str`` or ``list``
        If you use
        import scienceplots
        you can create nice and clear scientific plots.
    window_size: ``int``
        The window size for the rolling average smoothing.
    max_colors: ``int``
        Maximum number of different colors to use in the plot.
    """
    plt.style.use(style)
    colors = plt.cm.get_cmap('tab10', max_colors).colors  # Get a colormap with enough colors
    
    for idx, all_scores in enumerate(list_data_frames):

        # Ensure the indices are aligned
        all_scores = all_scores.set_index("env_interactions")
        all_scores = all_scores.reindex(np.arange(0, nb_steps + 1, 1000)).interpolate(method='linear').ffill().bfill()
        
        mean = all_scores.mean(axis=1)
        smoothed_mean = mean.rolling(window=window_size, min_periods=1).mean()  # Apply rolling average

        
        plt.plot(np.arange(0, nb_steps + 1, 1000), smoothed_mean, label=cases[idx], color=colors[idx % max_colors])

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(xmin=0, xmax=nb_steps)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.xlabel("environment interactions", fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    #plt.legend(fontsize=18, loc="lower right")
    f = plt.gcf()
    f.set_size_inches(15, 7)
    return f



def compare_all_results(
    cases,
    list_data_frames,
    style="default",
    nb_steps=1000,
    ymin=-105,
    ymax=105,
    ylabel="Mean score",
    title="Training Scores",
    confidence_level=0.95  # Add confidence level parameter
):
    plt.style.use(style)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Define hardcoded labels for each case
    case_labels = {
        "case_14_ppo_ppo": "PPO",
        "case_14_ippo_ippo": "IPPO CAPA",
        "case_14_ippo_i_raippo": "IPPO I",
        "case_14_2k_hac_ac_urgent": "HAC ac",
        #"case_14_2k_hac_ac_urgent": "RAIPPO",
        #"case_14_2k_hac_ac_urgent": "Urgent",
        "case_14_2k_hac_ac_prob": "Probabilistic",
        "case_14_2k_hac_ac_cyclic": "Cyclic",
        "case_14_2k_hac_ac_random": "Random",
        "case_14_2k_hac_a_urgent": "HAC a",
        "case_14_2k_sc_ac_urgent": "SC ac",
        "case_14_2k_u_urgent": "Unw",
        "case_14_3k_u_urgent": "Unw",
        "case_14_3k_a_urgent": "Action",
        "case_14_3k_louvain_urgent": "Louvain",
        "case_14_3k_hac_ac_urgent": "HAC ac",
        "case_14_4k_sc_a_urgent": "SC a",
        "case_14_4k_sc_ac_urgent": "SC ac",
        "case_14_4k_hac_a_urgent": "HAC a",
        "case_14_4k_hac_ac_urgent": "HAC ac",
    }

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)  # Calculate z-score for the confidence level

    for idx, all_scores in enumerate(list_data_frames):
        mean = all_scores.filter(regex='_mean').mean(axis=1)
        std = all_scores.filter(regex='_std').mean(axis=1)
        sem = std / np.sqrt(all_scores.shape[1])  # Calculate SEM
        ci = sem * z_score  # Calculate confidence interval

        plt.fill_between(
            all_scores["env_interactions"], mean - ci, mean + ci, interpolate=True, color=colors[idx], alpha=0.1
        )
        case_label = case_labels.get(cases[idx], cases[idx])  # Get the label from the dictionary or use the case name
        plt.plot(all_scores["env_interactions"], mean, label=case_label, color=colors[idx])

    #plt.title(label=title, fontsize=20, fontweight="bold")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(xmin=0, xmax=nb_steps)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.xlabel("Environment interactions", fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.legend(fontsize=18, loc="lower right")
    f = plt.gcf()
    f.set_size_inches(15, 7)
    return f

def compare_step_results(cases, step_data, style, nb_steps):
    plt.style.use(style)
    fig, ax = plt.subplots()

    for case, data in zip(cases, step_data):
        mean_step = data.mean(axis=1)
        x_values = range(0, len(mean_step) * (nb_steps // len(mean_step)), nb_steps // len(mean_step))
        ax.plot(x_values, mean_step, label=case)


    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
    ax.set_xlabel('Environment interactions', fontsize=24)
    ax.set_ylabel('Mean step', fontsize=24)
    #ax.set_title('Training Steps')
    ax.legend()
    
    return fig

def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

def compare_all_step_results(cases, all_step_data, style, nb_steps, window_size=100):
    plt.style.use(style)
    fig, ax = plt.subplots()
    
    for case in cases:
        case_data = all_step_data.get(case, [])
        if not case_data:
            continue
        
        case_labels = {
            "case_14_ppo_ppo": "PPO",
            "case_14_ippo_ippo": "IPPO CAPA",
            "case_14_ippo_i_raippo": "IPPO I",
            #"case_14_2k_hac_ac_urgent": "RAIPPO",
            #"case_14_2k_hac_ac_urgent": "HAC ac",
            "case_14_2k_hac_ac_urgent": "Urgent",
            "case_14_2k_hac_ac_prob": "Probabilistic",
            "case_14_2k_hac_ac_cyclic": "Cyclic",
            "case_14_2k_hac_ac_random": "Random",
            "case_14_2k_hac_a_urgent": "HAC a",
            "case_14_2k_sc_ac_urgent": "SC ac",
            "case_14_2k_u_urgent": "Unw",
            "case_14_3k_u_urgent": "Unw",
            "case_14_3k_a_urgent": "Action",
            "case_14_3k_louvain_urgent": "Louvain",
            "case_14_3k_hac_ac_urgent": "HAC ac",
            "case_14_4k_sc_a_urgent": "SC a",
            "case_14_4k_sc_ac_urgent": "SC ac",
            "case_14_4k_hac_a_urgent": "HAC a",
            "case_14_4k_hac_ac_urgent": "HAC ac",
        }

        # Pad shorter sequences with NaN
        max_length = 101
        padded_case_data = []
        for data in case_data:
            if len(data) < max_length:
                padding = pd.DataFrame(np.nan, index=range(max_length - len(data)), columns=data.columns)
                padded_data = pd.concat([data, padding], ignore_index=True)
            else:
                padded_data = data
            padded_case_data.append(padded_data)

        # Stack the data to compute mean and std
        stacked_data = np.dstack([data.values for data in padded_case_data])
        mean_step = np.nanmean(stacked_data, axis=2).flatten()
        std_step = np.nanstd(stacked_data, axis=2).flatten()
        
        # Apply smoothing
        mean_step_smooth = smooth_data(pd.Series(mean_step), window_size)
        std_step_smooth = smooth_data(pd.Series(std_step), window_size)
        
        # Limit std to be within 0 and 864
        std_step_smooth = np.clip(std_step_smooth, 0, 864)
        
        x_values = range(0, len(mean_step_smooth) * (nb_steps // len(mean_step_smooth)), nb_steps // len(mean_step_smooth))
        case_label = case_labels.get(case, case)  # Get the label from the dictionary or use the case name
        ax.plot(x_values, mean_step_smooth, label=case_label)
        ax.fill_between(x_values, mean_step_smooth - std_step_smooth, mean_step_smooth + std_step_smooth, alpha=0.2)

    ax.set_ylim(top=864)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
    ax.set_xlabel('Environment interactions', fontsize=24)
    ax.set_ylabel('Mean step', fontsize=24)
    #ax.set_title('Training Steps')
    ax.legend(fontsize=18)
    
    return fig

def get_trial_res(trial_dir, n_seeds):
    df = pd.DataFrame()
    for seed in range(n_seeds):
        try:
            score = pd.read_csv(f"{trial_dir}/seed_{seed}/score.csv")
            mean = score.iloc[:, 1:].mean(axis=1)
            df["env_interactions"] = score["env_interactions"]
            df[f"mean s{seed}"] = mean
        except:
            print(f"{trial_dir} does not have any values for seed {seed}, this trial probably got pruned")
    return df


def plot_trials(output_dir, n_trials, n_seeds, max_steps, name=""):
    """
    This function is used to plot the results of the parameter tunning. For usage see also param_tunning.py

    Parameters
    ----------
    output_dir: ``str``
        Directory where the trial results are stored.
    n_trials: ``int`` or ``list`` or ``range``
        trials to plot. If integer all trials in [0 ... n_trials] will be plotted.
    n_seeds: ``int``
        number of seeds that have been tested for each trial.
    max_steps: ``int``
        Maximum number of environment interactions (can be less if the process was stopped in the middle.)

    Example
    __________
    For usage see also param_tunning.py

    dir_name = "final_param_case5_dppo_tuning"
    trials = range(16,21)
    n_seeds = 3
    max_steps = 5000
    plot_trials(dir_name, trials, n_seeds, max_steps)


    or specifically select trials with better results:

    dir_name = "final_param_case14_ppo_tuning"
    param_res = get_param_res(dir_name)
    n_seeds = 3
    max_steps = 50_000
    trials = list(param_res.loc[(param_res['value'] > 60)]['number'])
    plot_trials(dir_name, trials, n_seeds, max_steps)

    """
    plt.style.use("default")
    fig = plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if isinstance(n_trials, int):
        trials = range(n_trials)
    elif isinstance(n_trials, list) or isinstance(n_trials, range):
        trials = n_trials
    else:
        raise TypeError("n_trials should be integer, list or range!")
    for t in range(trials):
        trial_path = os.path.join(output_dir, f"trial_{t}")
        if os.path.exists(trial_path):
            mean_scores = get_trial_res(trial_path, n_seeds)
            sem = mean_scores.iloc[:, 1:].sem(axis=1)
            mean = mean_scores.iloc[:, 1:].mean(axis=1)
            plt.fill_between(
                mean_scores["env_interactions"],
                mean - sem,
                mean + sem,
                interpolate=True,
                color=colors[t % len(colors)],
                alpha=0.1,
            )
            plt.plot(mean_scores["env_interactions"], mean, label=f"trial {t}", color=colors[t % len(colors)])
    plt.title(label=f"Mean scores - {name} ", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(xmin=0, xmax=max_steps)
    plt.ylim(ymin=-100, ymax=105)
    plt.xlabel("environment interactions", fontsize=16)
    plt.ylabel("mean score", fontsize=16)
    plt.legend(fontsize=18, loc="lower right")
    # fig = plt.gcf()
    fig.set_size_inches(15, 7)
    fig.savefig(os.path.join(output_dir, "images/mean_scores"))
    plt.clf()
    plt.cla()


# plot functionalities
def plot_action_gantt(folder):
    files = [file for file in os.listdir(folder) if file.endswith("_topo.npy") & file.startswith("Ch")]
    log_folder = os.path.join(folder, "eval_actions_gantts")
    os.makedirs(log_folder, exist_ok=True)
    if files:
        for file in files:
            topo = np.load(os.path.join(folder, file))
            test_case = "_".join(file.split("_", 2)[:2])

            fig = plt.figure(figsize=(20, topo.shape[1] / 2))
            ax = fig.add_subplot(1, 1, 1)
            plt.title(test_case, fontsize=20, fontweight="bold")
            colors = ["limegreen", "tab:red", "silver"]
            curr = 1
            for j in range(topo.shape[1]):
                start_pos = 0
                end_pos = 0
                for i in range(topo.shape[0]):
                    if topo[:, j][i] != curr:
                        plt.hlines(
                            y=j,
                            xmin=start_pos,
                            xmax=end_pos,
                            color=colors[curr - 1] if curr > 0 else colors[curr],
                            lw=20,
                            label="bus " + str(curr),
                        )
                        curr = topo[:, j][i]
                        start_pos = end_pos
                    end_pos += 1
                plt.hlines(
                    y=j,
                    xmin=start_pos,
                    xmax=end_pos,
                    color=colors[curr - 1] if curr > 0 else colors[curr],
                    lw=20,
                    label="bus " + str(curr),
                )

            ax.set_ylabel("element", fontsize=18)
            ax.set_xlabel("time step", fontsize=18)
            plt.xticks(np.arange(864, step=100), fontsize=16)
            plt.yticks(np.arange(topo.shape[1]), fontsize=16)
            handles, labels = ax.get_legend_handles_labels()
            indexes = [labels.index(x) for x in set(labels)]
            ax.legend(
                [handle for i, handle in enumerate(handles) if i in indexes],
                [label for i, label in enumerate(labels) if i in indexes],
                fontsize=22,
                loc="lower right",
            )
            plt.savefig(os.path.join(log_folder, f"actions_%s.png" % test_case))
            plt.show()
    else:
        print(f"The folder %s does not contain any Ch*.npy files." % folder)

def plot_safe_gantt_np(folder, plot_title="Env check is safe "):
    files = [file for file in os.listdir(folder) if file.endswith("safe.npy") & file.startswith("Ch")]
    log_folder = os.path.join(folder, "eval_safe")
    os.makedirs(log_folder, exist_ok=True)
    if files:
        fig = plt.figure(figsize=(20, len(files) / 2))
        fig.suptitle(plot_title)
        ax = fig.add_subplot(1, 1, 1)
        colors = ["tab:red", "limegreen"]
        y_ticks = []
        for i, file in enumerate(files):
            is_safe = np.load(os.path.join(folder, file))
            y_ticks.append("_".join(file.split("_", 2)[:2]))
            # is_safe = df["safe"]
            curr = True
            start_pos = 0
            end_pos = 0
            for safe in is_safe:
                if safe != curr:
                    plt.hlines(y=i, xmin=start_pos, xmax=end_pos, color=colors[curr], lw=20, label=str(curr))
                    curr = safe
                    start_pos = end_pos
                end_pos += 1
            plt.hlines(y=i, xmin=start_pos, xmax=end_pos, color=colors[curr], lw=20, label=str(curr))

        ax.set_ylabel("Epochs")
        ax.set_xlabel("Timesteps")
        plt.xticks(np.arange(864, step=100))
        plt.yticks(np.arange(len(y_ticks)), y_ticks)
        handles, labels = ax.get_legend_handles_labels()
        indexes = [labels.index(x) for x in set(labels)]
        ax.legend(
            [handle for i, handle in enumerate(handles) if i in indexes],
            [label for i, label in enumerate(labels) if i in indexes],
            loc="best",
        )
        plt.savefig(os.path.join(log_folder, f"safe.png"))
        plt.show()
    else:
        print(f"The folder %s does not contain any Ch*topoAnalytics.csv files." % folder)


def plot_safe_gantt(folder):
    files = [file for file in os.listdir(folder) if file.endswith("topoAnalytics.csv") & file.startswith("Ch")]
    log_folder = os.path.join(folder, "eval_safe")
    os.makedirs(log_folder, exist_ok=True)
    if files:
        fig = plt.figure(figsize=(20, len(files) / 2))
        fig.suptitle("Env check is safe ")
        ax = fig.add_subplot(1, 1, 1)
        colors = ["tab:red", "limegreen"]
        y_ticks = []
        for i, file in enumerate(files):
            df = pd.read_csv(os.path.join(folder, file))
            y_ticks.append("_".join(file.split("_", 2)[:2]))
            is_safe = df["safe"]
            curr = True
            start_pos = 0
            end_pos = 0
            for safe in is_safe:
                if safe != curr:
                    plt.hlines(y=i, xmin=start_pos, xmax=end_pos, color=colors[curr], lw=20, label=str(curr))
                    curr = safe
                    start_pos = end_pos
                end_pos += 1
            plt.hlines(y=i, xmin=start_pos, xmax=end_pos, color=colors[curr], lw=20, label=str(curr))

        ax.set_ylabel("Epochs")
        ax.set_xlabel("Timesteps")
        plt.xticks(np.arange(864, step=100))
        plt.yticks(np.arange(len(y_ticks)), y_ticks)
        handles, labels = ax.get_legend_handles_labels()
        indexes = [labels.index(x) for x in set(labels)]
        ax.legend(
            [handle for i, handle in enumerate(handles) if i in indexes],
            [label for i, label in enumerate(labels) if i in indexes],
            loc="best",
        )
        plt.savefig(os.path.join(log_folder, f"safe.png"))
        plt.show()
    else:
        print(f"The folder %s does not contain any Ch*topoAnalytics.csv files." % folder)

def write_depths_to_csv(all_depths, output_file):
    # Extract unique chronics and sort them
    chronics = sorted(set(key[0] for key in all_depths.keys()))
    
    # Create a list to hold depth data for each chronix
    depth_data = [[] for _ in range(len(chronics))]
    
    # Fill depth data list with depths from the dictionary
    for timestep in range(864):  # Assuming 864 timesteps
        for idx, chron_id in enumerate(chronics):
            if (chron_id, timestep) in all_depths:
                depth_data[idx].append(all_depths[(chron_id, timestep)])
            else:
                depth_data[idx].append(np.nan)
    
    # Write depth data to CSV
    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header row with chronix IDs
        writer.writerow(["Timestep"] + [f"Chron_{chron_id}" for chron_id in chronics])
        
        # Write depth data for each timestep
        for timestep, depths in enumerate(zip(*depth_data)):
            writer.writerow([timestep] + list(depths))

def write_action_counts_to_csv(file_path, action_counts_dict):
    with open(file_path, "a", newline="") as cf:
        writer = csv.writer(cf)
        for counts in action_counts_dict.values():
            # Write the action counts for each chronix without the chronix id
            writer.writerow(counts.values())

def write_steps_ol_to_csv(steps_overloaded, filename):
    # Extract thresholds dynamically by flattening all values in the nested dictionaries and using set to avoid duplicates.
    thresholds = sorted(set(threshold for chronic in steps_overloaded.values() for threshold in chronic))
    
    # Open the file for writing
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row with chronic IDs
        header = ['Threshold'] + sorted(steps_overloaded.keys())
        writer.writerow(header)
        
        # Iterate over each threshold and write data rows
        for threshold in thresholds:
            row = [threshold]
            for chron_id in sorted(steps_overloaded.keys()):
                row.append(steps_overloaded[chron_id].get(threshold, 0))  # Use .get to avoid KeyError
            writer.writerow(row)
            # print(f'Writing row: {row}')  # Print statement to trace what is being written

def write_depths_to_csv(all_depths, output_file):
    # Extract unique chronics and sort them
    chronics = sorted(set(key[0] for key in all_depths.keys()))
    
    # Create a list to hold depth data for each chronix
    depth_data = [[] for _ in range(len(chronics))]
    
    # Fill depth data list with depths from the dictionary
    for timestep in range(864):  # Assuming 864 timesteps
        for idx, chron_id in enumerate(chronics):
            if (chron_id, timestep) in all_depths:
                depth_data[idx].append(all_depths[(chron_id, timestep)])
            else:
                depth_data[idx].append(np.nan)
    
    # Write depth data to CSV
    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header row with chronix IDs
        writer.writerow(["Timestep"] + [f"Chron_{chron_id}" for chron_id in chronics])
        
        # Write depth data for each timestep
        for timestep, depths in enumerate(zip(*depth_data)):
            writer.writerow([timestep] + list(depths))

def write_action_counts_to_csv(file_path, action_counts_dict):
    # Open the file for writing
    with open(file_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        
        # Prepare header row with substation IDs
        # Assuming all chronics have the same subs and the first chronic can represent the subs
        first_chronic = next(iter(action_counts_dict.values()))
        header = ['Chron'] + [f"sub_{sub}" for sub in first_chronic.keys()]  # Substation IDs are the keys in the inner dict
        writer.writerow(header)
        
        # Write each chronic's action counts
        for chronic_id, counts in action_counts_dict.items():
            row = [chronic_id] + list(counts.values())
            writer.writerow(row)

def write_ra_action_counts_to_csv(file_path, action_counts_dict):
    # Open the file for writing
    with open(file_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        
        # Prepare header row with responsability area IDs
        # Assuming all chronics have the same RA and the first chronic can represent the RAs 
        first_chronic = next(iter(action_counts_dict.values()))
        header = ['Chron'] + [f"ra_{ra}" for ra in first_chronic.keys()]  # RA IDs are the keys in the inner dict
        writer.writerow(header)
        
        # Write each chronic's action counts
        for chronic_id, counts in action_counts_dict.items():
            row = [chronic_id] + list(counts.values())
            writer.writerow(row)

def write_steps_ol_to_csv(steps_overloaded, filename):
    # Extract thresholds dynamically by flattening all values in the nested dictionaries and using set to avoid duplicates.
    thresholds = sorted(set(threshold for chronic in steps_overloaded.values() for threshold in chronic))
    
    # Open the file for writing
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row with chronic IDs
        header = ['Threshold'] + [f"Chron_{chron}" for chron in steps_overloaded.keys()]
        writer.writerow(header)
        
        # Iterate over each threshold and write data rows
        for threshold in thresholds:
            row = [threshold]
            for chron_id in sorted(steps_overloaded.keys()):
                row.append(steps_overloaded[chron_id].get(threshold, 0))  # Use .get to avoid KeyError
            writer.writerow(row)
            # print(f'Writing row: {row}')  # Print statement to trace what is being written

def write_topologies_to_csv(topologies, output_file):
    # Collect all chronics and timesteps to prepare the header
    chronics_set = set((chron_id for (chron_id, timestep) in topologies.keys()))
    max_timestep = max((timestep for (chron_id, timestep) in topologies.keys()))
    
    # Prepare the header with chronics sorted for consistency
    chronics_sorted = sorted(chronics_set)
    header = ['timestep'] + [f"chron_{chron}" for chron in chronics_sorted]
    
    # Writing to CSV
    with open(output_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        # Each row corresponds to a timestep
        for timestep in range(max_timestep + 1):
            row = [timestep]
            for chron_id in chronics_sorted:
                key = (chron_id, timestep)
                # If the topology exists for this timestep and chronic, write it; otherwise, write an empty value
                row.append(topologies.get(key, ''))
            writer.writerow(row)

def write_is_safe_to_csv(is_safe, output_file):
    # Extract unique chronics and sort them
    chronics = sorted(set(key[0] for key in is_safe.keys()))
    
    # Find the maximum timestep to ensure all timesteps are covered
    max_timestep = max(key[1] for key in is_safe.keys())
    
    # Write to CSV
    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header row with chronix IDs
        writer.writerow(["Timestep"] + [f"Chron_{chron_id}" for chron_id in chronics])
        
        # Write is_safe data for each timestep
        for timestep in range(max_timestep + 1):
            row = [timestep]
            for chron_id in chronics:
                if (chron_id, timestep) in is_safe:
                    row.append(is_safe[(chron_id, timestep)])
                else:
                    row.append('')  # Placeholder if data does not exist for this chronix at this timestep
            writer.writerow(row)

def write_unique_topos_total_to_csv(unique_topos_total, filepath):
    with open(filepath, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['Topo_ID', 'Count', 'Topology_Vector'])
        for topo_vec, data in unique_topos_total.items():
            # Convert string representation of list back to list of integers
            topo_vect = list(map(int, topo_vec.strip('[]').split()))
            writer.writerow([data['topo_id'], data['count'], ' '.join(map(str, topo_vect))])

def write_substation_configs_to_csv(substation_configs, filepath):
    with open(filepath, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['Substation_ID', 'Configuration', 'Count'])
        for sub_id, configs in substation_configs.items():
            for config, count in configs.items():
                if not np.array_equal(config, np.ones(len(config), dtype=int)):
                    writer.writerow([sub_id, ' '.join(map(str, config)), count])

def load_depth_data(res_dir, cases, depth_type, eval=True, nb_step=None):
    depth_data = []
    for case in cases:
        if eval:
            file_path = os.path.join(res_dir, case, f'evaluation_measures/{depth_type}_depths.csv')
        else:
            file_path = os.path.join(res_dir, case, f'train_measures/{depth_type}_depths{f"_{nb_step}" if nb_step else ""}.csv')
        depth_data.append(pd.read_csv(file_path))
    return depth_data

def load_safety_data(res_dir, case, eval=True, nb_step=None):
    if eval:
        file_path = os.path.join(res_dir, case, 'evaluation_measures/is_safe.csv')
    else:
        file_path = os.path.join(res_dir, case, f'train_measures/is_safe{f"_{nb_step}" if nb_step else ""}.csv')
    return pd.read_csv(file_path)

def compare_depths(res_dir, cases, chronic_idx=0, depth_type="elem", style="default", nb_steps=None, ymin=None, ymax=None, ylabel="Depth", title="Topo Depth Comparison", show_danger=True, eval=True, nb_step=None):
    """
    Compare and plot the depth of topologies for multiple agents over time.

    Parameters
    ----------
    res_dir: str
        The directory where the results of the trained agents are saved.
    cases: list of str
        A list containing the names of the agents to compare.
    chronic_idx: int, list of int, or str, optional (default=0)
        The index or indices of the chronic(s) to analyze. Can be "all" to analyze all chronics.
    depth_type: str, optional (default="elem")
        The type of depth to analyze ("elem" or "sub").
    style: str, optional (default="default")
        The style to use for the plot.
    nb_steps: int or None, optional (default=None)
        The number of timesteps to display on the x-axis. If None, it is automatically determined based on the data.
    ymin: int or None, optional (default=None)
        The minimum value for the y-axis. If None, it is automatically determined based on the data.
    ymax: int or None, optional (default=None)
        The maximum value for the y-axis. If None, it is automatically determined based on the data.
    ylabel: str, optional (default="Depth")
        The label for the y-axis.
    title: str, optional (default="Topo Depth Comparison")
        The title of the plot.
    show_danger: bool, optional (default=True)
        Whether to show red background for danger areas when only one agent is being analyzed.
    eval: bool, optional (default=True)
        Whether to load evaluation or training data.
    nb_step: int or None, optional (default=None)
        The specific training step to load the data for, if eval is False.

    Returns
    -------
    matplotlib.figure.Figure
        The generated plot figure.
    """
    plt.style.use(style)
    
    # Load the depth data for each agent
    depth_data = load_depth_data(res_dir, cases, depth_type, eval, nb_step)
    
    # Extract chronic names
    chronic_names = depth_data[0].columns[1:]  # Exclude the "Timestep" column
    
    # Determine the chronics to analyze
    if chronic_idx == "all":
        chronics = chronic_names
    elif isinstance(chronic_idx, int):
        chronics = [chronic_names[chronic_idx]]
    elif isinstance(chronic_idx, list):
        chronics = [chronic_names[i] for i in chronic_idx]
    else:
        chronics = [chronic_names[0]]
    
    # Initialize plot
    plt.figure(figsize=(15, 7))
    max_y = 0  # Initialize a variable to find the maximum y-value
    max_x = 0  # Initialize a variable to find the maximum x-value
    

    # Load safety data if only one agent and show_danger is True
    if len(cases) == 1 and len(chronics) == 1 and show_danger:
        safety_data = load_safety_data(res_dir, cases[0], eval, nb_step)
        safety_data = safety_data.applymap(lambda x: bool(x))  # Ensure all values are boolean
        plt.axvspan(0, 1, color='red', alpha=0.3, label="Network in Danger")
 
    color_idx = 0
    color_map = plt.get_cmap("tab10")
    
    for chronic in chronics:
        for agent_idx, depth_df in enumerate(depth_data):
            agent_name = cases[agent_idx]
            depth_col = chronic
            if depth_col not in depth_df.columns:
                print(f"Warning: {depth_col} not found in {agent_name}'s data.")
                continue
            depth_series = depth_df[depth_col]
            
            # Plot the depth data
            label = f"{agent_name}_{chronic}" if len(cases) > 1 else chronic
            color = color_map(color_idx % 10)
            plt.plot(depth_df["Timestep"], depth_series, label=label, color=color)
            
            # Update max_y and max_x with the maximum values from this data
            current_max_y = depth_series.max()
            current_max_x = depth_df["Timestep"].max()
            if current_max_y > max_y:
                max_y = current_max_y
            if current_max_x > max_x:
                max_x = current_max_x
            
            # Check for NaN values to determine if the agent failed
            if depth_series.isna().any():
                failure_timestep = depth_series[depth_series.isna()].index[0]
                plt.scatter(failure_timestep, depth_series[failure_timestep - 1], color=color, zorder=5)
            
            color_idx += 1
    
    # Customize the plot
    plt.title(label=title, fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(xmin=0, xmax=max_x if nb_steps is None else nb_steps)
    plt.xticks(range(0, max_x + 1 if nb_steps is None else nb_steps + 1, 100))  # Adjust x-axis ticks to occur every 100 steps
    if ymax is None:
        ymax = max_y + (max_y * 0.1)  # Add 10% buffer above the max_y
    plt.ylim(ymin=ymin if ymin is not None else 0, ymax=ymax)
    plt.xlabel("Timestep", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True)
    
    # Add red background for danger areas if applicable
    if len(cases) == 1 and len(chronics) == 1 and show_danger:
        for chronic in chronics:
            if chronic in safety_data.columns:
                unsafe_periods = ~safety_data[chronic].astype(bool)  # Ensure all values are boolean
                start_idx = None
                for idx, is_unsafe in unsafe_periods.items():
                    if is_unsafe and start_idx is None:
                        start_idx = idx
                    elif not is_unsafe and start_idx is not None:
                        plt.axvspan(start_idx, idx, color='red', alpha=0.3)
                        start_idx = None
                if start_idx is not None:
                    plt.axvspan(start_idx, len(unsafe_periods), color='red', alpha=0.3)

    plt.legend(fontsize=18, loc='best')   
    plt.show()

def compare_unique_topologies(cases, res_dir, style='science', title="Comparison of Unique Topologies", eval=True, nb_step=None):
    """
    Compares unique topologies across multiple agents using bar plots.

    Parameters:
        cases (list): A list of case names corresponding to different agents.
        res_dir (str): The directory path where each agent's results are stored.
        style (str or list): Plotting style for matplotlib.
        title (str): Title of the plot.
        eval (bool): Whether to load evaluation or training data.
        nb_step (int or None): The specific training step to load the data for, if eval is False.
    
    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
    """
    plt.style.use(style)
    fig, ax = plt.subplots()

    # Bar plot data
    bar_width = 0.35  # width of the bars
    indices = np.arange(len(cases))  # indices of the bars

    unique_counts = []
    for case in cases:
        print(f"Processing case: {case}")
        if eval:
            file_path = f"{res_dir}/{case}/evaluation_measures/unique_topologies_chron.csv"
        else:
            file_path = f"{res_dir}/{case}/train_measures/unique_topologies_chron{f'_{nb_step}' if nb_step else ''}.csv"
        
        try:
            print("File path: ", file_path)
            # Explicitly mention no headers in the file
            data = pd.read_csv(file_path, header=None)
            print("Data loaded: ", data)
            # Sum of unique counts across all columns
            unique_counts.append(data.iloc[0].sum())
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            unique_counts.append(0)
        except Exception as e:
            print(f"An error occurred: {e}")
            unique_counts.append(0)

    ax.bar(indices, unique_counts, width=bar_width, color='b', label='Unique Topologies')

    ax.set_xlabel('Agents')
    ax.set_ylabel('Count of Unique Topologies')
    ax.set_title(title)
    ax.set_xticks(indices)
    ax.set_xticklabels(cases, rotation=45)  # Rotate labels for better visibility
    ax.legend()

    fig.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    
    return fig

def plot_unique_topologies_boxplot(cases, res_dir, title="Unique Topologies per chronic", eval=True, nb_step=None):
    """
    Creates a boxplot of unique topologies for each agent.

    Parameters:
        cases (list of str): A list of agent case names.
        res_dir (str): The directory path where each agent's results are stored.
        title (str): Title of the plot.
        eval (bool): Whether to load evaluation or training data.
        nb_step (int or None): The specific training step to load the data for, if eval is False.
    """
    data = []  # This will store the unique topologies counts for each agent

    for case in cases:
        if eval:
            file_path = os.path.join(res_dir, f"{case}/evaluation_measures/unique_topologies_chron.csv")
        else:
            file_path = os.path.join(res_dir, f"{case}/train_measures/unique_topologies_chron{f'_{nb_step}' if nb_step else ''}.csv")
        
        try:
            # Load the data assuming no header and all data in one row
            case_data = pd.read_csv(file_path, header=None)
            # Append the entire row of data as one entry in the list
            data.append(case_data.iloc[0].values)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, patch_artist=True)
    plt.xticks(range(1, len(cases) + 1), cases, rotation=45)
    plt.title(title)
    plt.xlabel('Agents')
    plt.ylabel('Unique Topologies Count')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()

def convert_to_list(s):
    if isinstance(s, str):
        return list(map(int, s.strip('[]').replace('\n', '').split()))
    return []

def load_topologies(file_path):
    df = pd.read_csv(file_path)
    for column in df.columns[1:]:
        df[column] = df[column].apply(convert_to_list)
    return df

def compute_differences_plot(res_dir, cases, chronics_to_analyze="all", eval=True, nb_step=None):
    """
    This function computes and plots the differences in topologies between two agents over time for specified chronics.
    It also indicates the points where agents fail and prints the failure information.

    Parameters
    ----------
    res_dir: str
        The directory where the results of the trained agents are saved.
    cases: list of str
        A list containing the names of two agents to compare.
    chronics_to_analyze: str, int, or list of int, optional (default="all")
        The chronics to analyze. Can be "all" to analyze all chronics, an integer to analyze a specific chronic by index,
        or a list of integers to analyze multiple specific chronics.
    eval (bool): Whether to load evaluation or training data.
    nb_step (int or None): The specific training step to load the data for, if eval is False.

    Example
    ----------
    compute_differences_plot(res_dir, ["ppoMetrics_ppo_0", "raippoMetrics_raippo_0"], chronics_to_analyze="all")
    compute_differences_plot(res_dir, ["ppoMetrics_ppo_0", "raippoMetrics_raippo_0"], chronics_to_analyze=0)
    compute_differences_plot(res_dir, ["ppoMetrics_ppo_0", "raippoMetrics_raippo_0"], chronics_to_analyze=[0, 2, 4])

    """
    # Load the topologies for both cases
    if eval:
        topologies_case_1 = load_topologies(os.path.join(res_dir, cases[0], 'evaluation_measures/raw_topologies.csv'))
        topologies_case_2 = load_topologies(os.path.join(res_dir, cases[1], 'evaluation_measures/raw_topologies.csv'))
    else:
        topologies_case_1 = load_topologies(os.path.join(res_dir, cases[0], f'train_measures/raw_topologies{f"_{nb_step}" if nb_step else ""}.csv'))
        topologies_case_2 = load_topologies(os.path.join(res_dir, cases[1], f'train_measures/raw_topologies{f"_{nb_step}" if nb_step else ""}.csv'))
    
    # List of all chronics
    all_chronics = topologies_case_1.columns[1:]
    
    # Determine the chronics to analyze
    if chronics_to_analyze == "all":
        chronics = all_chronics
    elif isinstance(chronics_to_analyze, int):
        chronics = [all_chronics[chronics_to_analyze]]
    elif isinstance(chronics_to_analyze, list):
        chronics = [all_chronics[i] for i in chronics_to_analyze]
    else:
        chronics = [all_chronics[0]]
    
    # Initialize plots
    plt.figure(figsize=(12, 8))
    
    # Dictionary to keep track of which agent fails first for each chronic
    failure_info = {}
    
    # Compute and plot differences for each chronic
    for chronic in chronics:
        differences = []
        failure_timestep = None
        for t in range(len(topologies_case_1)):
            topology_1 = topologies_case_1[chronic][t]
            topology_2 = topologies_case_2[chronic][t]
            
            if not topology_1 or not topology_2:
                # Check which agent failed first
                if not topology_1 and not topology_2:
                    failure_info[chronic] = (t, 'Both')
                elif not topology_1:
                    failure_info[chronic] = (t, cases[0])
                elif not topology_2:
                    failure_info[chronic] = (t, cases[1])
                failure_timestep = t - 1
                break
            
            # Calculate the difference as the number of different elements
            difference = sum(1 for a, b in zip(topology_1, topology_2) if a != b)
            differences.append(difference)
        
        # Plot the differences
        plt.plot(differences, label=chronic)
        
        # Add marker to indicate where the comparison ends
        if failure_timestep is not None:
            if len(chronics) <= 2:
                plt.axvline(x=failure_timestep, color='red', linestyle='--')
            else:
                plt.scatter(failure_timestep, differences[-1], color=plt.gca().lines[-1].get_color(), zorder=5)
    
    # Customize the plot
    plt.xlabel('Timestep')
    plt.ylabel('Distance Between Topologies')
    plt.title('Distance in Topologies Between Two Agents Over Time')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    # Print failure information
    for chronic in chronics:
        if chronic in failure_info:
            timestep, agent = failure_info[chronic]
            if agent == 'Both':
                print(f"For {chronic}, both agents failed at timestep {timestep}.")
            else:
                print(f"For {chronic}, {agent} failed first at timestep {timestep}.")
        else:
            print(f"For {chronic}, both agents completed the episode successfully.")

def plot_action_proportions(res_dir, cases, eval=True, nb_step=None):
    """
    Plots the proportion of actions affecting each substation for multiple agents.
    
    Parameters:
        res_dir (str): Directory where the CSV files are stored.
        cases (list): List of strings representing the case names (different agents).
        eval (bool): Whether to load evaluation or training data.
        nb_step (int or None): The specific training step to load the data for, if eval is False.
    """
    plt.figure(figsize=(10, 6))
    width = 0.2  # Width of the bars
    n = len(cases)  # Number of cases
    color_order = ['blue', 'green', 'orange', 'red']
    colors = [mcolors.TABLEAU_COLORS[f'tab:{color}'] for color in color_order]

    # Create subplot
    ax = plt.subplot(111)
    
    case_labels = {
        "case_14_ppo_ppo_7": "PPO",
        "case_14_ippo_ippo_0": "IPPO CAPA",
        "IPPO_I_2": "IPPO I",
        "case_14_2k_hac_ac_prob_raippo_0": "RAIPPO",
    }

    for i, case in enumerate(cases):
        # Construct file path
        if eval:
            file_path = os.path.join(res_dir, f"{case}/evaluation_measures/action_counts.csv")
        else:
            file_path = os.path.join(res_dir, f"{case}/train_measures/action_counts{f'_{nb_step}' if nb_step else ''}.csv")
        
        # Load data
        data = pd.read_csv(file_path)
        
        # Drop 'Chron' column and convert counts to proportions
        data = data.drop('Chron', axis=1)
        proportions = data.div(data.sum(axis=1), axis=0)
        means = proportions.mean()  # Mean proportion of each substation
        
        # Drop 'sub2' if it exists
        if 'sub_2' in means.index:
            means = means.drop(['sub_2'])
        
        # Sort indices correctly, ensuring 'sub12' is last
        means = means.reindex(sorted(means.index, key=lambda x: (int(re.sub(r'\D', '', x)) if x != 'sub12' else 12)))
        
        # Determine bar positions
        indices = np.arange(len(means)) + i * width
        
        # Plot
        case_label = case_labels.get(case, case)
        ax.bar(indices, means, width=width, label=case_label, color=colors[i])
    plt.style.use(['science', 'grid', 'no-latex'])
    
    # Set chart title and labels
    plt.xlabel('Affected Substation', fontsize=24)
    plt.ylabel('Proportion of Actions', fontsize=24)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
    f = plt.gcf()
    f.set_size_inches(8, 6)
    
    # Set x-ticks
    ax.set_xticks(np.arange(len(means)) + width * (n / 2 - 0.5))
    new_labels = [re.sub(r'sub_(\d+)', r'sub \1', label) for label in means.index]  # Replace "sub_X" with "sub X"
    ax.set_xticklabels(new_labels)
    
    # Adjust subplot parameters to reduce padding
    plt.tight_layout(pad=0.0)


    # Add legend
    plt.legend(fontsize=18)
    
    # Show plot
    plt.show()

def load_substation_configs(res_dir, case, eval=True, nb_step=None):
    """
    Load substation configurations from CSV file for a given agent.

    Parameters:
        res_dir (str): Directory where the CSV files are stored.
        case (str): Agent case name.
        eval (bool): Whether to load evaluation or training data.
        nb_step (int or None): The specific training step to load the data for, if eval is False.

    Returns:
        pd.DataFrame: DataFrame containing the substation configurations.
    """
    if eval:
        file_path = os.path.join(res_dir, f"{case}/evaluation_measures/unique_substation_configurations.csv")
    else:
        file_path = os.path.join(res_dir, f"{case}/train_measures/unique_substation_configurations{f'_{nb_step}' if nb_step else ''}.csv")
    data = pd.read_csv(file_path)
    return data

def calculate_proportions(data):
    """
    Calculate the proportions of actions for each substation configuration.

    Parameters:
        data (pd.DataFrame): DataFrame containing substation configurations.

    Returns:
        dict: Dictionary with substation IDs as keys and configuration proportions as values.
    """
    total_actions = data['Count'].sum()
    data['Proportion'] = data['Count'] / total_actions
    proportions = {}
    for sub_id, group in data.groupby('Substation_ID'):
        proportions[sub_id] = group.set_index('Configuration')['Proportion'].to_dict()
    return proportions

def plot_substation_configs(res_dir, cases, eval=True, nb_step=None):
    """
    Plot the proportion of actions for each substation configuration for multiple agents.

    Parameters:
        res_dir (str): Directory where the CSV files are stored.
        cases (list of str): List of agent case names.
        eval (bool): Whether to load evaluation or training data.
        nb_step (int or None): The specific training step to load the data for, if eval is False.
    """
    plt.figure(figsize=(15, 10))
    width = 0.2  # Width of the bars
    all_proportions = {}
    
    for case in cases:
        data = load_substation_configs(res_dir, case, eval, nb_step)
        proportions = calculate_proportions(data)
        all_proportions[case] = proportions

    # Collect all unique substation IDs and configurations
    all_substations = set()
    all_configurations = set()
    for case_proportions in all_proportions.values():
        for sub_id, configs in case_proportions.items():
            all_substations.add(sub_id)
            for config in configs.keys():
                all_configurations.add((sub_id, config))

    all_configurations = sorted(all_configurations)
    
    # Create a dictionary to map each (sub_id, config) pair to its position
    config_positions = {config: i for i, config in enumerate(all_configurations)}
    
    # Calculate the positions for the bars
    n_cases = len(cases)
    
    for offset, case in enumerate(cases):
        case_positions = []
        case_values = []
        proportions = all_proportions[case]
        for sub_id, config in all_configurations:
            position = config_positions[(sub_id, config)] + offset * width
            case_positions.append(position)
            if sub_id in proportions and config in proportions[sub_id]:
                case_values.append(proportions[sub_id][config])
            else:
                case_values.append(0)
        
        plt.bar(case_positions, case_values, width=width, label=case)


    # Set x-ticks and labels
    plt.xticks(
        [config_positions[config] + width * (n_cases - 1) / 2 for config in all_configurations],
        [f"{sub_id}\n{config}" for sub_id, config in all_configurations],
        rotation=45, ha='right'
    )

    # Add legend, title, and labels
    plt.legend(title='Agent')
    plt.title('Substation Configurations')
    plt.xlabel('Configuration')
    plt.ylabel('Proportion of Actions')
    plt.tight_layout()
    
    # Show plot
    plt.show()

def load_topo_data(res_dir, cases, eval=True, nb_step=None):
    unique_topos_total = defaultdict(lambda: defaultdict(int))
    for case in cases:
        if eval:
            file_path = os.path.join(res_dir, f"{case}/evaluation_measures/unique_topos_total.csv")
        else:
            file_path = os.path.join(res_dir, f"{case}/train_measures/unique_topos_total{f'_{nb_step}' if nb_step else ''}.csv")
        
        data = pd.read_csv(file_path)
        for _, row in data.iterrows():
            topo_id = row['Topo_ID']
            count = row['Count']
            topo_vec = row['Topology_Vector']
            unique_topos_total[topo_vec][topo_id] += count
    return unique_topos_total

def calculate_proportions_topos(res_dir, cases, eval=True, nb_step=None):
    unique_topos_total = load_topo_data(res_dir, cases, eval, nb_step)
    proportions = defaultdict(dict)
    for topo_vec, topo_ids in unique_topos_total.items():
        total_actions = sum(topo_ids.values())
        for topo_id, count in topo_ids.items():
            proportions[topo_vec][topo_id] = count / total_actions
    return proportions

def plot_unique_topos(res_dir, cases, eval=True, nb_step=None):
    """
    Plots the proportions of unique topologies for multiple agents.

    Parameters:
        res_dir (str): Directory where the CSV files are stored.
        cases (list): List of strings representing the case names (different agents).
        eval (bool): Whether to load evaluation or training data.
        nb_step (int or None): The specific training step to load the data for, if eval is False.
    """
    plt.figure(figsize=(15, 10))
    width = 0.2  # Width of the bars
    colors = [color for _, color in zip(range(len(cases)), mcolors.TABLEAU_COLORS.values())]  # Fetch colors from TABLEAU_COLORS
    
    # Create subplot
    ax = plt.subplot(111)
    
    # Collect all unique topologies
    all_topologies = {}
    total_actions_per_case = []
    
    for case in cases:
        if eval:
            file_path = os.path.join(res_dir, f"{case}/evaluation_measures/unique_topologies_total.csv")
        else:
            file_path = os.path.join(res_dir, f"{case}/train_measures/unique_topologies_total{f'_{nb_step}' if nb_step else ''}.csv")
        
        # Load data
        data = pd.read_csv(file_path)
        
        total_actions = data['Count'].sum()
        total_actions_per_case.append(total_actions)
        
        for _, row in data.iterrows():
            topo_vec = tuple(map(int, row['Topology_Vector'].split()))
            if topo_vec not in all_topologies:
                all_topologies[topo_vec] = {case: row['Count']}
            else:
                all_topologies[topo_vec][case] = row['Count']
    
    # Relabel the topologies
    relabeled_topologies = {topo: f"topo_{i+1}" for i, topo in enumerate(all_topologies.keys())}
    
    # Prepare the data for plotting
    bar_positions = np.arange(len(relabeled_topologies))
    case_offsets = np.arange(len(cases)) * width
    
    for i, case in enumerate(cases):
        proportions = []
        for topo in all_topologies.keys():
            count = all_topologies[topo].get(case, 0)
            proportions.append(count / total_actions_per_case[i])
        
        ax.bar(bar_positions + case_offsets[i], proportions, width=width, label=case, color=colors[i])
    
    # Set x-ticks and labels
    plt.xticks(bar_positions + width * (len(cases) - 1) / 2, list(relabeled_topologies.values()), rotation=45, ha='right')
    
    # Add legend, title, and labels
    plt.legend(title='Agent')
    plt.title('Unique Topologies')
    plt.xlabel('Topology')
    plt.ylabel('Proportion of Actions')
    
    # Show plot
    plt.show()

def compute_across_chronic_measures(output_dir, evaluate=True, verbose=False, suffix=None):
    # Helper function to read csv with suffix
    def read_csv_with_suffix(file_path, suffix, header=True):
        if suffix is not None:
            file_path = file_path.replace(".csv", f"{suffix}.csv")
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            if header:
                headers = next(reader)
                data = [row for row in reader]
                return headers, data
            else:
                data = [row for row in reader]
                return data
    
    if evaluate:
        final_dir = output_dir
        score_data = read_csv_with_suffix(os.path.join(final_dir, "score.csv"), "", header=False)
    else:
        final_dir = os.path.join(output_dir, "train_measures")
        score_headers, score_data = read_csv_with_suffix(os.path.join(final_dir, "score.csv"), "")
    
    # Read all the relevant CSV files
    step_data = read_csv_with_suffix(os.path.join(final_dir, "step.csv"), "", header=False)
    sub_depths_headers, sub_depths_data = read_csv_with_suffix(os.path.join(final_dir, "sub_depths.csv"), "")
    elem_depths_headers, elem_depths_data = read_csv_with_suffix(os.path.join(final_dir, "elem_depths.csv"), "")
    unique_topos_total_headers, unique_topos_total_data = read_csv_with_suffix(os.path.join(final_dir, "unique_topologies_total.csv"), "")
    _, unique_substation_data = read_csv_with_suffix(os.path.join(final_dir, "unique_substation_configurations.csv"), "")
    is_safe_data = read_csv_with_suffix(os.path.join(final_dir, "is_safe.csv"), "", header=False)

    # List to store the output strings
    output_lines = []

    # 1. Mean episode length and standard deviation from step_data
    first_row_step = list(map(float, step_data[0]))
    mean_episode_length = round(np.mean(first_row_step), 2)
    std_episode_length = round(np.std(first_row_step), 2)

    output_lines.append(f"Mean episode length: {mean_episode_length:.2f}")
    output_lines.append(f"Standard deviation of episode length: {std_episode_length:.2f}")

    # 2. Mean reward and std from score_data
    first_row_score = list(map(float, score_data[0]))
    mean_reward = round(np.mean(first_row_score), 2)
    std_reward = round(np.std(first_row_score), 2)

    output_lines.append(f"Mean reward: {mean_reward:.2f}")
    output_lines.append(f"Standard deviation of reward: {std_reward:.2f}")

    # 3. Number of unsolved scenarios
    num_unsolved_scenarios = len([res for res in first_row_score if float(res) < 80.0])

    output_lines.append(f"Number of unsolved scenarios: {num_unsolved_scenarios}")

    # 4. Number of actions
    num_topo_changes = sum(int(row[2]) for row in unique_substation_data)

    output_lines.append(f"Number of actions: {num_topo_changes}")

    # 5. Number of topology changes
    total_topo_changes = 0
    for row in unique_substation_data:
        config = row[1].split()
        count = int(row[2])
        num_twos = config.count('2')
        total_topo_changes += num_twos * count

    output_lines.append(f"Number of topo changes: {total_topo_changes}")

    # Total number of steps survived
    total_steps_survived = sum(map(int, step_data[0]))
    # Total steps in danger
    is_safe_array = np.array(is_safe_data)
    steps_in_danger = np.sum(is_safe_array == 'False')
    
    # 6. Proportion of timesteps network in danger
    proportion_steps_in_danger = round(steps_in_danger / total_steps_survived, 2)

    output_lines.append(f"Proportion of timesteps network in danger: {proportion_steps_in_danger:.2f}")

    # 7. Activity while in danger
    activity_while_in_danger = round(num_topo_changes / steps_in_danger, 2)

    output_lines.append(f"Actions per step in danger: {activity_while_in_danger:.2f}")

    topo_changes_per_step_in_danger = round(total_topo_changes / steps_in_danger, 2)

    output_lines.append(f"Topo changes per timestep in danger: {topo_changes_per_step_in_danger:.2f}")

    # 8. Number of unique topologies
    num_unique_topologies = len(unique_topos_total_data)
    output_lines.append(f"Number of unique topologies: {num_unique_topologies}")
    
    # 9. Number of unique topology configurations
    substation_configs = {}
    for row in unique_substation_data:
        sub_id = row[0]
        config = row[1]
        count = row[2]
        if sub_id not in substation_configs:
            substation_configs[sub_id] = []
        substation_configs[sub_id].append((config, count))

    num_unique_topo_configs = len(unique_substation_data)
    output_lines.append(f"Number of unique substation configurations: {num_unique_topo_configs}")

    # 10. Mean and std of topo depth
    sub_depths_df = pd.DataFrame(sub_depths_data[1:], columns=sub_depths_data[0]).apply(pd.to_numeric, errors='coerce')
    sub_depths_df = sub_depths_df.iloc[:, 1:]

    # Flatten the DataFrame to a single array of elements
    all_elements = sub_depths_df.values.flatten()

    # Compute the mean and standard deviation, ignoring NaNs
    mean_topo_depth = round(np.nanmean(all_elements), 2)
    std_topo_depth = round(np.nanstd(all_elements), 2)

    output_lines.append(f"Mean topo depth: {mean_topo_depth:.2f}")
    output_lines.append(f"Standard deviation of topo depth: {std_topo_depth:.2f}")

    # 11. Mean and standard deviation of elem depth
    elem_depths_df = pd.DataFrame(elem_depths_data[1:], columns=elem_depths_data[0]).apply(pd.to_numeric, errors='coerce')
    elem_depths_df = elem_depths_df.iloc[:, 1:]
    all_elements_elem = elem_depths_df.values.flatten()

    # Compute the mean and standard deviation, ignoring NaNs
    mean_elem_depth = round(np.nanmean(all_elements_elem), 2)
    std_elem_depth = round(np.nanstd(all_elements_elem), 2)

    output_lines.append(f"Mean elem depth: {mean_elem_depth:.2f}")
    output_lines.append(f"Standard deviation of elem depth: {std_elem_depth:.2f}")

    # Extra*. Identifying Substation configurations
    output_lines.append("\nUnique Substation Configurations:")
    for sub_id, configs in substation_configs.items():
        output_lines.append(f"Sub id: {sub_id}")
        for config, count in configs:
            output_lines.append(f"  Config: {config}, Times chosen: {count}")

    # Write the output to a file
    output_file_path = os.path.join(final_dir, "summary_measures.txt")
    with open(output_file_path, 'w') as output_file:
        output_file.write("\n".join(output_lines))

    # Print the output
    if verbose:
        for line in output_lines:
            print(line)