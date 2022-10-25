import gym
import numpy as np
from gym.utils import EzPickle
from params import configs
from tool import permissibleLeftShift


def override(fn):
    return fn


def getActionNbghs(action, opIDsOnMchs): # get action neighbourhound
    coordAction = np.where(opIDsOnMchs == action)
    precd = opIDsOnMchs[coordAction[0], coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]].item() # if the action is not the first action, then has pre action
    succdTemp = opIDsOnMchs[coordAction[0], coordAction[1] + 1 if coordAction[1].item() + 1 < opIDsOnMchs.shape[-1] else coordAction[1]].item()
    succd = action if succdTemp < 0 else succdTemp
    return precd, succd


def lastNonZero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)
    xAxis = np.arange(arr.shape[0], dtype=np.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def calEndTimeLB(temp1, dur_cp): # calculate the end time for the future jobs
    x, y = lastNonZero(temp1, 1, invalid_val=-1)
    dur_cp[np.where(temp1 != 0)] = 0
    dur_cp[x, y] = temp1[x, y]
    temp2 = np.cumsum(dur_cp, axis=1)
    temp2[np.where(temp1 != 0)] = 0
    ret = temp1+temp2
    return ret


class Env(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_modes = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_modes
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    def done(self): # incorrect cuz when jobs are done the whole episode done
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        if action not in self.partial_sol_sequeence:
            row = action // self.number_of_modes
            col = action % self.number_of_modes
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            if action not in self.last_col:
                self.omega[action // self.number_of_modes] += 1
            else:
                self.mask[action // self.number_of_modes] = 1

            self.temp1[row, col] = startTime_a + dur_a
            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1

        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        reward = - (self.LBs.max() - self.max_endTime) # reward is defined by the difference between the estimated time and the maximal time
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    @override
    def reset(self, data):

        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0
        # define the adj matrix of the network
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        conj_nei_up_stream[self.first_col] = 0
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        self.omega = self.first_col.astype(np.int64)
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)
        self.temp1 = np.zeros_like(self.dur, dtype=np.single)
        return self.adj, fea, self.omega, self.mask

if __name__ == '__main__':
    filepath = r'./test.sch'
    env = Env(15, 15)
    s, a, c, m = env.reset()
    env.step(0)
    env.step(1)