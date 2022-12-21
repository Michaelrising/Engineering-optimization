from ActConstraints import Constraints, ReadInfo
import numpy as np
from gym import spaces
from copy import deepcopy



class ProgEnv(Constraints, ReadInfo):
    def __init__(self, filepath, T_target=16, price_renew=5, price_non=0.5, penalty0=3, penalty1=3, penalty_mode='all', acnet='mlp'):
        super().__init__(filepath)
        self.acnet = acnet
        self.steps = 0
        self.action_num = sum(self.Activity_mode_Num)
        self.action_space = spaces.Discrete(self.action_num)
        self.price_renewable_resource = price_renew * np.ones_like(self.Renewable_resource.reshape(-1))
        self.price_nonrenewable_resource = price_non * np.ones_like(self.Nonrenewable_resource.reshape(-1))
        self.actSeq = []
        self.modeSeq = []
        self.timeSeq = []
        self.actionSeq = []
        self.crtTime = 0
        self.activities = list(self.SsLimit.keys()) # string
        self.done = False
        self.stateGraph = np.zeros((self.action_space.n, self.action_space.n))# np.ones((self.action_space.n, self.action_space.n)) * (-np.inf) # non-direction graph
        self.return_stateGraph = self.stateGraph
        self.actGraph = np.zeros((self.action_space.n, self.action_space.n))
        self.logicGraph1 = np.zeros((len(self.Activity_mode_Num), len(self.Activity_mode_Num))) # 7 * 7
        self.logicGraph2 = np.zeros((len(self.Activity_mode_Num), len(self.Activity_mode_Num)))
        self.actStatus = np.zeros(self.action_space.n)
        self.actStatus[-1] = -1
        self.candidate = np.arange(self.action_space.n)
        self.pastMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        self.penalty_coeff0 = penalty0
        self.penalty_coeff1 = penalty1
        self.T_target = T_target
        self.timeFeasible = []
        self.nonRenewFeasible = []
        self.renewFeasible = []
        self.actFeasible = []
        self.not_feasible_history = 0
        self.past_feasible_acts = np.zeros(self.action_space.n)
        self.penalty_mode = penalty_mode
        self.max_potential = 0
        self.nonFeasibleActivities = np.array([])

    def actionDetermine(self, action):
        # decide the action represents which activity and which mode
        assert action < self.action_num
        temp = 0
        for activity in range(len(self.Activity_mode_Num)): # act starts from 0 mode starts from 1
            for mode in np.arange(1, self.Activity_mode_Num[activity] + 1):
                if action == temp:
                    return int(activity), int(mode)
                temp += 1

    def resetStateGraph(self):
        # initial graph for the whole state space is len(action) x len(action), so in the test is 13x13.
        # we replace with 1e-8 for 0 weight link, and for -inf we replace with 0
        for activity in self.activities:
            temp = self.SsLimit[activity]
            for key in list(temp.keys()):
                if (temp[key] > 0).all():
                    self.logicGraph1[int(key), int(activity)] = 1
                if (temp[key] < 0).all():
                    self.logicGraph2[int(key), int(activity)] = 1
                for mode_act in np.arange(1, self.Activity_mode_Num[int(activity)] + 1): # 1, 2, 3 ...
                    for mode_key in np.arange(1, self.Activity_mode_Num[int(key)] + 1):# 1, 2, 3 ...
                        pcol = sum(self.Activity_mode_Num[:(int(activity))]) + mode_act - 1
                        prow = sum(self.Activity_mode_Num[:(int(key))]) + mode_key - 1
                        self.stateGraph[prow, pcol] = temp[key][mode_act-1, mode_key-1] if abs(temp[key][mode_act-1, mode_key-1]) else 1e-8 # if 0 then we set it as 1e-8
                        self.actGraph[prow, pcol] = 0.5
        # EOF activity
        last_activity = self.activities[-1]
        eof_action = int(self.action_space.n - 1)
        for mode in np.arange(1, self.Activity_mode_Num[int(last_activity)] + 1):
            pcol = sum(self.Activity_mode_Num[:(int(last_activity))]) + mode - 1
            self.stateGraph[eof_action, pcol] = 1

    def BuildactGraph(self, action):
        # after one action is taken, the explicit activity and corresponding mode are determined,
        # then we have to modify te graph for current state: for each mode that will not be taken in the future,
        # replace the whole column and row with 0
        action_limit = sum(self.Activity_mode_Num[:self.crtAct])+ np.arange(0, self.Activity_mode_Num[self.crtAct])
        temp = self.SsLimit[str(self.crtAct)]
        for act_col in action_limit:
            if act_col != action:
                self.actGraph[:, act_col] = 0 # - np.inf #### set as -M ####
                self.actGraph[act_col] = 0 # - np.inf
        for k in list(temp.keys()):
            for mode_k in np.arange(1, self.Activity_mode_Num[int(k)] + 1):
                prow = sum(self.Activity_mode_Num[:(int(k))]) + mode_k - 1
                self.actGraph[prow, action] = 1

    def updateActStatus(self, action, action_limit):
        # update the activity status
        # update the activity status, for the done activity set to 1,
        self.actStatus[action] = 1

    def startTimeDetermine(self, graph): # fix !!!!!!
        assert len(self.timeSeq) + 1 == graph.shape[0]

        if len(self.timeSeq) != 0:
            OutDegree, InDegree = graph[:-1, -1], graph[-1, :-1]
            OutDegree[OutDegree == 0] = -50000
            InDegree[InDegree == 0] = -50000
            latestTime = np.array(self.timeSeq) - OutDegree
            earliestTime = np.array(self.timeSeq) + InDegree
            if int(earliestTime.max()) <= int(latestTime.min()):
                greedy_startTime = int(earliestTime.max()) if earliestTime.max() > - 1000 else self.crtTime # greedy choice of start time
                end_startTime = max(latestTime.min(), greedy_startTime)
                return greedy_startTime, end_startTime
            else:
                return max(int(earliestTime.max()), latestTime.min()), max(int(earliestTime.max()), latestTime.min())
        return 0, 0

    def potential(self, past_feasible_acts):
        # the percentage that the project is complicated is set as the potential function
        # which is to decide the reward
        actions_scores = sum(past_feasible_acts)
        projectStatus = actions_scores/len(self.activities) * 500 # 2000
        if projectStatus > self.max_potential:
            self.max_potential = projectStatus
        return self.max_potential

    def feasibleDetermine(self):
        act_feasibleMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        time_feasibleMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        time_urgency_index = np.full(shape=self.action_space.n, fill_value=0, dtype=float)
        if self.done:
            return act_feasibleMask
        feasibleActivity = []
        time_infeasibleActivity = []

        for act in self.actSeq:
            temp = np.where(self.logicGraph1[:, act] == 1)[0]
            for t in temp:
                temp1 = np.where(self.logicGraph1[t] == 1)[0]
                if not np.isin(t, self.actSeq) and np.isin(temp1, self.actSeq).all() and not np.isin(t, feasibleActivity):
                    feasibleActivity.append(t)
        for f_activity in feasibleActivity:
            for f_mode in np.arange(1, self.Activity_mode_Num[f_activity]+1):
                feasibleAction = sum(self.Activity_mode_Num[:(int(f_activity))]) + f_mode - 1
                act_feasibleMask[feasibleAction] = True

        for i, act_mask in enumerate(act_feasibleMask):
            if act_mask:
                time_feasibleMask[i] = True
                time_urgency_index[i] = 10000
                action_states_vec = self.stateGraph[:, i]
                back_time_check_actions = np.where(action_states_vec < 0)[0]
                valid_back_time_check_actions = back_time_check_actions[
                    np.isin(back_time_check_actions, self.actionSeq)]
                for back_action in valid_back_time_check_actions:
                    pos = np.where(back_action == self.actionSeq)[0]
                    pos = pos[0].item()
                    back_action_start_time = self.timeSeq[int(pos)]
                    if back_action_start_time - self.crtTime >= action_states_vec[int(back_action)]:
                        time_feasibleMask[i] = True
                    else:
                        time_feasibleMask[i] = False

                    time_urgency_index[i] = min(- self.crtTime + back_action_start_time - action_states_vec[int(back_action)], time_urgency_index[i])

        time_urgency_index[np.where(time_urgency_index == 10000)[0]] = max(time_urgency_index[np.where(time_urgency_index != 10000)[0]]) + 1
        for f_activity in feasibleActivity:
            feasibleAction = sum(self.Activity_mode_Num[:(int(f_activity))]) + np.arange(self.Activity_mode_Num[f_activity])
            time_fea_or_not = time_feasibleMask[feasibleAction]
            if (time_fea_or_not == False).all():
                time_infeasibleActivity.append(f_activity)

        self.action_feasibleMask = act_feasibleMask
        self.time_feasibleMask = time_feasibleMask
        return feasibleActivity, time_infeasibleActivity, time_urgency_index

    def renew_time_feasibility(self, PastActGraph, PastLogicVec2, startTime):
        action_states_vec = self.stateGraph[:, int(self.actionSeq[-1])]
        back_time_check_actions = np.where(action_states_vec < 0)[0]
        valid_back_time_check_actions = back_time_check_actions[np.isin(back_time_check_actions, self.actionSeq[:-1])]
        greedy_startTime, latest_startTime = self.startTimeDetermine(PastActGraph)
        start_Time = greedy_startTime # max(startTime, greedy_startTime)
        time_Feasible = False
        RenewFeasible = self.Is_Renewable_Resource_Feasible(self.modeSeq, self.actSeq, self.timeSeq, start_Time)
        while start_Time <= latest_startTime:
            flag = 0
            for back_action in valid_back_time_check_actions:
                if back_action in self.actionSeq:
                    pos = np.where(back_action == self.actionSeq)[0]
                    pos = pos[0].item()
                    back_action_start_time = self.timeSeq[int(pos)]
                    if back_action_start_time - start_Time >= action_states_vec[int(back_action)]:
                        flag += 1
                else:
                    flag += 1
            if flag == valid_back_time_check_actions.shape[0]:
                time_Feasible = True
            if time_Feasible and RenewFeasible:
                break
            start_Time += 1
            RenewFeasible = self.Is_Renewable_Resource_Feasible(self.modeSeq, self.actSeq, self.timeSeq, start_Time)
        start_Time = min(start_Time, latest_startTime)
        return RenewFeasible, time_Feasible, start_Time, greedy_startTime, latest_startTime

    def resource_time_penalty(self, g_time):
        renewR = self.get_current_mode_using_renewable_resource(self.crtMode, self.crtAct)
        least_renewR = self.get_current_act_using_least_renewable_resource(self.crtAct)
        diff_renewR = renewR - least_renewR
        NonrenewR = self.get_current_mode_using_nonrenewable_resource(self.crtMode, self.crtAct)
        least_NonrenewR = self.get_current_act_using_least_nonrenewable_resource(self.crtAct)
        diff_NonrenewR = NonrenewR - least_NonrenewR
        # the time lag penalty compared to the best mode
        crt_duration = self.get_current_duration(self.crtMode, self.crtAct)
        best_duration = self.get_current_least_duration(self.crtAct)
        diff_duration = crt_duration - best_duration  # duration diff: cost3
        # penalty for resource used
        renew_Penalty = np.dot(diff_renewR, self.price_renewable_resource.reshape(-1))  # renew penalty cost1
        nonrenew_Penalty = np.dot(diff_NonrenewR * crt_duration,
                                  self.price_nonrenewable_resource.reshape(-1)) # nonrenew penalty cost2
        duration_Penalty = diff_duration * 1
        time_Penalty = 1 * (self.crtTime - g_time)  # g_time means the earliest start time for each activity
        return renew_Penalty, nonrenew_Penalty, duration_Penalty, time_Penalty

    def step(self, action):
        startTime = self.crtTime
        potential0 = self.potential(self.past_feasible_acts)
        self.actionSeq.append(action)
        self.crtAct, self.crtMode = self.actionDetermine(action)
        self.actSeq.append(self.crtAct)
        self.modeSeq.append(self.crtMode)
        mask_limit = sum(self.Activity_mode_Num[:self.crtAct])+ np.arange(0, self.Activity_mode_Num[self.crtAct]) #self.BuildStateGraph(action)
        self.pastMask[mask_limit] = True
        PastActGraph = deepcopy(self.stateGraph[self.actionSeq][:, self.actionSeq])
        PastLogicVec1 = deepcopy(self.logicGraph1[self.crtAct])
        PastLogicVec2 = deepcopy(self.logicGraph2[self.crtAct])
        actionFeasible = self.JudgeFeasibility(PastLogicVec1, self.actSeq) # determine whether former act has been taken
        nonRenewFeasible = self.Is_NonRenewable_Resource_Feasible(self.crtMode, self.crtAct)

        RenewFeasible, timeFeasible, start_Time, greedy_time, latest_time = self.renew_time_feasibility(PastActGraph, PastLogicVec2, startTime)

        self.crtTime = int(start_Time)
        self.timeSeq.append(self.crtTime)
        self.BuildactGraph(action)
        self.updateActStatus(action, mask_limit)

        feasibleActivity, time_infeasibleActivity, time_urgency_index = self.feasibleDetermine()
        self.steps += 1
        return_mask = ~self.action_feasibleMask
        ######### use when train lot 1, else set as notes ###########
        # if action ==  1:
        #     return_mask = np.full(shape=self.action_space.n, fill_value=1, dtype=bool)
        #     return_mask[5] = False

        self.done = bool(self.steps == len(self.Activity_mode_Num))

        renew_Penalty, nonrenew_Penalty, duration_Penalty, time_Penalty = self.resource_time_penalty(greedy_time)

        # not feasible penalty
        self.timeFeasible.append(not timeFeasible)
        self.actFeasible.append(not actionFeasible)
        self.nonRenewFeasible.append(not nonRenewFeasible)
        self.renewFeasible.append(not RenewFeasible)
        self.not_feasible_history += (not timeFeasible) \
                               + (not actionFeasible) \
                               + (not nonRenewFeasible) \
                               + (not RenewFeasible)

        if timeFeasible and actionFeasible and nonRenewFeasible and RenewFeasible:
            self.past_feasible_acts[action] = 1
        else:
            self.past_feasible_acts[action] = -1
        self.nonFeasibleActivities = np.append(self.nonFeasibleActivities, time_infeasibleActivity)
        if action in  self.nonFeasibleActivities:
            inds = np.where(self.nonFeasibleActivities == action)
            self.nonFeasibleActivities = np.delete(self.nonFeasibleActivities, inds)

        reward = - self.nonFeasibleActivities.shape[0] * (500 / len(self.activities)) # \

        # after taking action, there may exist some activities that should tak earlier
        if self.crtTime > latest_time:
            reward -= (self.crtTime - latest_time)*2  # **2

        potential1 = self.potential(self.past_feasible_acts)
        reward += potential1 - potential0
        if self.penalty_mode == 'early':
            reward += - 1 * time_Penalty
        elif self.penalty_mode == 'resource0':
            reward += - 1 * renew_Penalty - 1 * nonrenew_Penalty
        elif self.penalty_mode == 'resource1':
            reward += - 1 * renew_Penalty
        elif self.penalty_mode == 'resource2':
            reward += - 1 * nonrenew_Penalty
        elif self.penalty_mode == 'each':
            reward += - 1 * duration_Penalty
        elif self.penalty_mode == 'early+each':
            reward += - 1 * duration_Penalty - 1 * time_Penalty
        else:
            reward += - 1 * time_Penalty - 1 * duration_Penalty - 1 * renew_Penalty - 1 * nonrenew_Penalty

        if self.done:
            T = self.crtTime
            diff_T = self.T_target - T
            if diff_T >= 0:
                reward += self.penalty_coeff0 * diff_T * 0.1
            else:
                reward += self.penalty_coeff1 * diff_T * 0.1
            reward -= self.not_feasible_history * 2 # 50

        if self.acnet == 'mlp':
            fea = np.concatenate((self.actStatus, self.action_feasibleMask, self.time_feasibleMask))
        elif self.acnet == 'gnn':
            fea = np.vstack((self.actStatus, self.action_feasibleMask,   self.time_feasibleMask)).T
        else:
            fea = np.vstack((self.actStatus, self.action_feasibleMask, self.time_feasibleMask)).T
            fea = fea[np.newaxis, np.newaxis, :, :]

        return self.return_stateGraph, fea, reward, self.done, self.candidate, return_mask,  self.crtTime, {'time': timeFeasible, 'activity':actionFeasible, 'renew': RenewFeasible, 'nonrenew': nonRenewFeasible}

    def reset(self):
        self.resetResource()
        self.steps = 0
        self.actSeq = []
        self.modeSeq = []
        self.timeSeq = []
        self.actionSeq = []
        self.nonFeasibleActivities = np.array([])
        self.crtTime = 0
        self.done = False
        self.stateGraph = np.zeros((self.action_space.n, self.action_space.n))   # non-direction graph
        self.actStatus = np.zeros(self.action_space.n)
        self.actStatus[-1] = -1
        self.resetStateGraph()
        self.logicGraph1[1, 0] = 1
        self.pastMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        self.action_feasibleMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        self.time_feasibleMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        self.action_feasibleMask[0] = True # time 0 only activity 0 is feasible
        self.time_feasibleMask[0] = True
        self.feasiblemask = self.action_feasibleMask * self.time_feasibleMask
        self.candidate = np.arange(self.action_space.n)
        self.timeFeasible = []
        self.nonRenewFeasible = []
        self.renewFeasible = []
        self.actFeasible = []
        self.not_feasible_history = 0
        self.past_feasible_acts = np.zeros(self.action_space.n)
        feasible_weights = np.full(shape=self.action_space.n, fill_value=0, dtype=int)
        feasible_weights[0] = 1
        return_mask = ~self.feasiblemask
        self.max_potential = 0
        if self.acnet == 'mlp':
            fea = np.concatenate((self.actStatus, self.action_feasibleMask, self.time_feasibleMask))
        elif self.acnet == 'gnn':
            fea = np.vstack((self.actStatus, self.action_feasibleMask,   self.time_feasibleMask)).T
        else:
            fea = np.vstack((self.actStatus, self.action_feasibleMask, self.time_feasibleMask)).T
            fea = fea[np.newaxis, np.newaxis, :, :]
        self.return_stateGraph = self.stateGraph + np.diag(np.ones(self.action_space.n))
        return self.return_stateGraph, fea, self.candidate, return_mask