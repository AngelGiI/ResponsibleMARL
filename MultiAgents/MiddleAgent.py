import numpy as np
import random


class FixedSubPicker(object):
    def __init__(self, masked_sorted_sub, **kwargs):
        self.masked_sorted_sub = masked_sorted_sub
        self.subs_2_act = []
        n_subs = len(masked_sorted_sub)
        self.count = np.zeros((n_subs, n_subs), int)
        self.previous_sub = -1

    def complete_reset(self):
        self.subs_2_act = []
        self.previous_sub = -1

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # randomize the order in which the substations are activated
            self.subs_2_act = list(self.masked_sorted_sub)
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act

    def count_transitions(self, next_sub):
        if self.previous_sub >= 0:
            prev = np.flatnonzero(self.masked_sorted_sub == self.previous_sub).squeeze()
            next = np.flatnonzero(self.masked_sorted_sub == next_sub).squeeze()
            # print(prev, next)
            self.count[prev, next] += 1

    @property
    def transition_probs(self):
        row_sums = self.count.sum(axis=1, keepdims=True)
        non_zero_rows = (row_sums != 0).squeeze()
        probs = np.zeros_like(self.count, float)
        probs[non_zero_rows] = self.count[non_zero_rows] / row_sums[non_zero_rows]
        return probs


class RandomOrderedSubPicker(FixedSubPicker):
    def __init__(self, masked_sorted_sub, **kwargs):
        super().__init__(masked_sorted_sub)

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # randomize the order in which the substations are activated
            self.subs_2_act = list(self.masked_sorted_sub)
            random.shuffle(self.subs_2_act)
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act


class RuleBasedSubPicker(FixedSubPicker):
    def __init__(self, masked_sorted_sub, action_space):
        super().__init__(masked_sorted_sub)
        self.sub_line_or = []
        self.sub_line_ex = []
        for sub in self.masked_sorted_sub:
            self.sub_line_or.append(
                np.flatnonzero(action_space.line_or_to_subid == sub)
            )
            self.sub_line_ex.append(
                np.flatnonzero(action_space.line_ex_to_subid == sub)
            )

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
            # which applies an action to substations that require urgent care.
            rhos = []
            for sub in self.masked_sorted_sub:
                sub_i = np.flatnonzero(self.masked_sorted_sub == sub).squeeze()
                rho = np.append(
                    obs.rho[self.sub_line_or[sub_i]].copy(),
                    obs.rho[self.sub_line_ex[sub_i]].copy(),
                )
                rho[rho == 0] = 3
                rho_max = rho.max()
                rho_mean = rho.mean()
                rhos.append((rho_max, rho_mean))
            order = sorted(
                zip(self.masked_sorted_sub, rhos), key=lambda x: (-x[1][0], -x[1][1])
            )
            self.subs_2_act = list(list(zip(*order))[0])
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act

# ReArPicker: Responsible Area Picker. Similar logic as CAPA ordering.
# Discarded in the final experiment but base for the other pickers.
class ReArPicker(object):
    def __init__(self, action_space, clusters):
        self.action_space = action_space
        self.clusters = clusters
        self.cluster_priority = []  # To store priority of each cluster
        self.activation_sequence = []  # Ordered list of clusters based on priority (equivalent to subs_2_act)
        self.previous_ra = None
        n_ra = len(clusters)
        self.count = np.zeros((n_ra, n_ra), int)

    def update_activation_sequence(self, obs):
        self.calculate_cluster_priority(obs)
        # Sort clusters based on priority
        self.activation_sequence = sorted(range(len(self.clusters)), key=lambda i: -self.cluster_priority[i])

    def calculate_cluster_priority(self, obs):
        self.cluster_priority = []
        for cluster in self.clusters:
            cluster_max_rho = max([max(obs.rho[self.action_space.line_or_to_subid == sub]) for sub in cluster if len(obs.rho[self.action_space.line_or_to_subid == sub]) > 0])
            self.cluster_priority.append(cluster_max_rho)
    
    def pick_cluster(self, obs, sample):
        if not self.activation_sequence:
            # If the activation sequence is empty, it means priorities need to be recalculated
            self.update_activation_sequence(obs)

        if sample:
            self.count_ra_transitions(self.activation_sequence[0])
            self.previous_ra = self.activation_sequence[0]
        self.current_ra = self.activation_sequence[0]
        return self.activation_sequence.pop(0)  # Pop the highest priority cluster
    
    def complete_reset(self):
        self.activation_sequence = []
        self.previous_ra = None # Used to avoid counting as transition initialization or reset.

    def count_ra_transitions(self, next_ra):
        if self.previous_ra != None:
            prev = self.previous_ra
            next = next_ra
            self.count[prev, next] += 1

# RandomPicker: randomly select a cluster
class RandomPicker(ReArPicker):
    def __init__(self, action_space, clusters):
        super(RandomPicker, self).__init__(action_space, clusters)

    def update_activation_sequence(self, obs):
        # Randomly shuffle the clusters
        self.activation_sequence = list(range(len(self.clusters)))
        random.shuffle(self.activation_sequence)

# UrgentPicker: only contains the most urgent cluster
class UrgentPicker(ReArPicker):
    def __init__(self, action_space, clusters):
        super(UrgentPicker, self).__init__(action_space, clusters) 

    def update_activation_sequence(self, obs):
        self.calculate_cluster_priority(obs)
        # Set activation sequence to only contain the most urgent cluster
        most_urgent_cluster_index = self.cluster_priority.index(max(self.cluster_priority))
        self.activation_sequence = [most_urgent_cluster_index]

# ProbabilisticPicker: pick cluster based on a probability distribution derived from danger values
class ProbabilisticPicker(ReArPicker):
    def __init__(self, action_space, clusters):
        super(ProbabilisticPicker, self).__init__(action_space, clusters)

    def update_activation_sequence(self, obs):
        self.calculate_cluster_priority(obs)

        # Convert cluster_priority to a numpy array
        cluster_priority_array = np.array(self.cluster_priority)

        # Shifted exponential transformation with non-negative values
        exp_priorities = np.maximum(np.exp(cluster_priority_array - 0.7) - 1, 0)
        # Using 0.7 means that e.g. if the max danger value for one cluster is 0.8 -> we get 0.11 exp_priority,
                #                                       and for the other it is 0.9 -> we get 0.22 exp_priority
                # then, in the after transformation, the second cluster has twice the probability of being picked compared to the first one.
        # Calculate probabilities based on transformed priorities
        total_priority = sum(exp_priorities)
        if total_priority > 0:
            probabilities = [priority / total_priority for priority in exp_priorities]
        else:
            probabilities = [1 / len(self.clusters) for _ in self.clusters]

        # Select a cluster based on the calculated probabilities
        chosen_cluster = np.random.choice(range(len(self.clusters)), p=probabilities)
        self.activation_sequence = [chosen_cluster]

# CyclicPicker: pick cluster based on a fixed cycle
class CyclicPicker(ReArPicker):
    def __init__(self, action_space, clusters):
        super(CyclicPicker, self).__init__(action_space, clusters)
        self.current_index = 0  # To keep track of the current index in the cycle

    def update_activation_sequence(self, obs):
        # No need to update activation sequence for CyclicPicker as it follows a fixed order
        pass

    def pick_cluster(self, obs, sample):
        # Pick the next cluster in the cycle
        picked_ra = self.current_index
        self.current_index = (self.current_index + 1) % len(self.clusters)

        if sample:
            self.count_ra_transitions(picked_ra)
            self.previous_ra = picked_ra
        
        self.current_ra = picked_ra
        return picked_ra

    def complete_reset(self):
        super().complete_reset()
        self.current_index = 0

# UrgentLimitedPicker: pick cluster based on priority, but decrease priority of the current cluster after each consecutive pick.
# Discarded in the final experiment.
class UrgentLimitedPicker(ReArPicker):
    def __init__(self, action_space, clusters, exploration_coeff=0.9):
        super(UrgentLimitedPicker, self).__init__(action_space, clusters)
        self.consecutive_picks = 0 # Count of consecutive picks for the current cluster
        self.current_cluster = None  # Track the currently active cluster
        self.exploration_coeff = exploration_coeff  # Coefficient for exploration term

    def pick_cluster(self, obs, sample):
        if not self.activation_sequence:
            # If the activation sequence is empty, it means priorities need to be recalculated
            self.update_activation_sequence(obs)

        picked_ra = self.activation_sequence[0]
        if sample:
            self.count_ra_transitions(picked_ra)
            self.previous_ra = picked_ra
        
        if picked_ra == self.current_cluster:
            self.consecutive_picks += 1
        else:
            self.current_cluster = picked_ra
            self.consecutive_picks = 1

        return self.activation_sequence.pop(0)  # Pop the highest priority cluster

    def update_activation_sequence(self, obs):
        self.calculate_cluster_priority(obs)
        
        if self.current_cluster is not None:
            # Update priority of the current cluster
            self.cluster_priority[self.current_cluster] = self.cluster_priority[self.current_cluster] * self.exploration_coeff ** self.consecutive_picks
            print("Updated priority of cluster", self.current_cluster, "to", self.cluster_priority[self.current_cluster])

        # Sort clusters based on priority
        self.activation_sequence = sorted(range(len(self.clusters)), key=lambda i: -self.cluster_priority[i])

    def complete_reset(self):
        super().complete_reset()
        self.consecutive_picks = 0
        self.current_cluster = None