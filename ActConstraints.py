# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:29:52 2021

@author: Administrator
"""
import numpy as np
import tarjan
import random
from scipy.sparse.csgraph import johnson
from copy import deepcopy
MaxSearchPerIteration = 10000


class ReadInfo:
    def __init__(self, filename, para='none'):
        self.filename = filename
        self.para = para
        self.n = 0
        self.K0 = 0
        self.K1 = 0
        self.Renewable_resource = []
        self.Nonrenewable_resource = []
        self.activities = []
        self.Activity_mode_Num = []
        self.Resources = dict()
        self.SsLimit = dict()
        self.graph = dict()
        self.strong_connect_components = []

    def info(self):
        # 读取.sch中的数据并转化为相对应的数据结构
        # dlg = win32ui.CreateFileDialog(1)
        # dlg.SetOFNInitialDir('C:\\Users\\Administrator\\Desktop\\help\\Archive\\testset_mm100')
        # dlg.DoModal()
        # filename = dlg.GetPathName()
        filename = self.filename
        if (len(filename) > 0):
            print(filename)
            with open(filename, "r") as f:
                line = f.readline()
                line = line[:-1]
                data = line.split('\t')
                self.n = int(data[0]) + 2
                self.K0 = int(data[1])
                self.K1 = int(data[2])
                self.Renewable_resource = np.zeros([1, self.K0], dtype=int)
                self.Nonrenewable_resource = np.zeros([1, self.K1], dtype=int)
                self.activities = []
                self.Activity_mode_Num = []
                successor_num = []
                self.Resources = dict()
                self.SsLimit = dict()
                successor_act = []
                graph_data = []
                self.graph = dict()
                i = 0
                num = 0
                temp = ''
                while line:
                    line = f.readline()
                    line = line[:-1]
                    data = line.split('\t')
                    if i < self.n:
                        self.activities.append(data[0])
                        self.Activity_mode_Num.append(int(data[1]))
                        successor_num.append(int(data[2]))
                        successor_act.append(data[3:int(data[2]) + 3])
                        self.graph[data[0]] = data[3:int(data[2]) + 3]
                        graph_data.append(data[int(data[2]) + 3:])
                    if i == self.n:
                        num = sum(self.Activity_mode_Num)
                        assert data[0] != ''
                    if i >= self.n and i < self.n + num:
                        resource = dict()
                        if data[0] != '':
                            temp = data[0]
                        key = temp + '-' + data[1]
                        resource['duration'] = int(data[2])
                        using_renewable_resource = []
                        for k in range(self.K0):
                            using_renewable_resource.append(int(data[3 + k]))
                        resource['using_renewable_resource'] = using_renewable_resource
                        using_nonrenewable_resource = []
                        for k in range(self.K1):
                            using_nonrenewable_resource.append(int(data[3 + self.K0 + k]))
                        resource['using_nonrenewable_resource'] = using_nonrenewable_resource
                        resource['wi'] = 0
                        self.Resources[key] = resource
                    if i == self.n + num:
                        for j in range(len(data)):
                            if j < self.K0:
                                self.Renewable_resource[0, j] = int(data[j])
                            else:
                                self.Nonrenewable_resource[0, j - self.K0] = int(data[j])
                    i = i + 1
                f.close()
            self.strong_connect_components = tarjan.tarjan(self.graph)
            for i in range(len(graph_data)):
                ni = self.Activity_mode_Num[i]
                i_data = dict()
                ikey = str(i)
                for j in range(len(graph_data[i])):
                    nj = self.Activity_mode_Num[int(successor_act[i][j])]
                    key = successor_act[i][j]
                    t = graph_data[i][j].replace('[', '').replace(']', '').split(' ')
                    for k in range(len(t)):
                        t[k] = int(t[k])
                    arr = np.array(t).reshape([ni, nj])
                    i_data[key] = arr
                self.SsLimit[ikey] = i_data


class Constraints(ReadInfo):

    def __init__(self, filename):
        super().__init__(filename)
        self.info() # read the infos into the class
        self.initNonrenewable_resource = np.copy(self.Nonrenewable_resource.reshape(-1))
        self.initRenewable_resource = np.copy(self.Renewable_resource.reshape(-1))

    def resetResource(self):
        self.Nonrenewable_resource = np.copy(self.initNonrenewable_resource)
        self.Renewable_resource = np.copy(self.initRenewable_resource)

    def JudgeFeasibility(self, graph, actSeq) -> object:
        # assert actSeq.shape == modeSeq.shape[1]
        # graph = self.buildGraph(modeSeq, actSeq)
        graph = deepcopy(graph)
        # graph[graph == 0] = -50000
        # if len(actSeq) == 1:
        #     tempM = np.ones(len(self.activities)) * (-5000)
        #     for activity in self.activities:
        #         temp = self.SsLimit[activity]  # dict
        #         if str(act0) in list(temp.keys()):
        #             tempM[int(activity)] = temp[str(act0)][:, int(mode0) - 1].min().item()
        #     if (tempM > 0).any():
        #         isvalid = False
        #     else:
        #         isvalid = True
        # else:
        #     logic_former_act = np.where(graph == 1)[0]
        #     isvalid = (np.isin(logic_former_act, np.array(actSeq))==True).all()
        #     # M = self.ShortestPath(graph)
        #     # isvalid = self.Isvalid(M)
        logic_former_act = np.where(graph == 1)[0]
        isvalid = np.isin(logic_former_act, np.array(actSeq)).all()
        return isvalid

    def get_activity_least_time(self):
        # 获取每一个活动的所有mode的最短持续时间，用于最后cost的计算，返回np.array，shape=[1,n]
        activity_least_time = []
        for i in range(len(self.Activity_mode_Num)):
            temptime = []
            for j in range(self.Activity_mode_Num[i]):
                key = self.activities[i] + '-' + str(j + 1)
                temptime.append(self.Resources[key]['duration'])
            activity_least_time.append(np.min(np.array(temptime)))
        return np.array(activity_least_time).reshape([1, len(self.Activity_mode_Num)])

    def get_wi(self, modeSeq):
        # 获取参数wi，用于cost计算，输入模式向量np.array，shape=[1,n]，返回np.array，shape=[1,n]
        #    wi = []
        #    for i in range(modesequece.shape[1]):
        #        key = p.activities[i] + '-' + str(modesequece[0,i])
        #        wi.append(p.Resources[key]['wi'])
        #    return np.array(wi).reshape(modesequece.shape)
        return np.ones_like(modeSeq)

    # after the whole project is finished
    def getProjectTime(self, timeSeq, modeSeq, actSeq):
        # 获取在给定的mode以及开始时间，整个项目完成所持续的时间，输入每个模式的开始时间数组np.array，shape=[1,n]以及模式向量np.array，shape=[1,n]，用于最后cost的计算，返回int
        # give the past and current mode seq and the start time for each act
        assert len(timeSeq) == len(modeSeq) == len(actSeq)
        T = 0
        duration = []
        for act, mode in zip(actSeq, modeSeq):
            duration.append(self.get_current_duration(mode, act))
        # duration = np.array(duration, dtype = np.int).reshape(len(modeSeq))
        for i in range(len(timeSeq)):
            if timeSeq[i] + duration[i] > T:
                T = timeSeq[i] + duration[i]
        endTimeSeq = np.array(timeSeq) + np.array(duration)
        return T, endTimeSeq, np.array(duration)

    def get_current_duration(self, currentMode, currentAct):
        # 获取给定模式的持续时间，input current activity and current mode chose
        key = str(currentAct) + '-' + str(currentMode)
        duration = self.Resources[key]['duration']
        return duration

    def get_current_least_duration(self, currentAct):
        temptime = 10000
        for j in range(self.Activity_mode_Num[currentAct]):
            key = self.activities[currentAct] + '-' + str(j + 1)
            d = self.Resources[key]['duration']
            if d < temptime:
                temptime = d
        return temptime

    # 返回特定模式下消耗的可更新资源矩阵,输入current mode and activity output the resource used
    def get_current_mode_using_renewable_resource(self, currentMode, currentAct):
        # give current mode and current act give the resource used for this activity
        key = str(currentAct) + '-' + str(currentMode)
        mode_using_renewable_resource = self.Resources[key]['using_renewable_resource']
        return np.array(mode_using_renewable_resource)

    def get_current_act_using_least_renewable_resource(self, currentAct):
        temp = np.ones_like(self.Renewable_resource.reshape(-1)) * 10000
        for j in range(self.Activity_mode_Num[currentAct]):
            key = self.activities[currentAct] + '-' + str(j + 1)
            mode_using_renewable_resource = self.Resources[key]['using_renewable_resource']
            if (temp > np.array(mode_using_renewable_resource)).all():
                temp = np.array(mode_using_renewable_resource)
        return temp

    # 返回特定模式下消耗的不可更新资源矩阵,输入current mode and activity output the resource used
    def get_current_mode_using_nonrenewable_resource(self, currentMode, currentAct):
        key = str(currentAct) + '-' + str(currentMode)
        mode_using_nonrenewable_resource = self.Resources[key]['using_nonrenewable_resource']
        return np.array(mode_using_nonrenewable_resource)

    def get_current_act_using_least_nonrenewable_resource(self, currentAct):
        temp = np.ones_like(self.Nonrenewable_resource.reshape(-1)) * 10000
        for j in range(self.Activity_mode_Num[currentAct]):
            key = self.activities[currentAct] + '-' + str(j + 1)
            mode_using_nonrenewable_resource = self.Resources[key]['using_nonrenewable_resource']
            if (temp > np.array(mode_using_nonrenewable_resource)).all():
                temp = np.array(mode_using_nonrenewable_resource)
        return temp

    # def Is_Renewable_Resource_Feasible(self,modesequece , start_timesequence, T):
    #     # 判断给定的模式向量及各个模式开始时间是否满足可更新资源限制
    #     # modesequece:模式数组，存储每个工序所使用的模式序号,类型为np.array,shape=[1,n]
    #     # start_timesequence:每个模式的开始时间数组,类型为np.array,shape=[1,n]
    #     # duration:每个模式的持续时间，类型为np.array,shape=[1,n]
    #     # T:整个工序的周期，int
    #     # renewable_resource:可更新资源数组，存储每一种可更新资源数量,类型为np.array,shape=[1,K0]
    #     # mode_using_renewable_resource:当前模式队列下，每个模式需要消耗的可更新资源,类型为np.array,shape=[n,K0]
    #     duration = self.get_duration(modesequece)
    #     mode_using_renewable_resource = self.get_mode_using_renewable_resource(modesequece)
    #     end_timesequence = start_timesequence + duration
    #     temp = np.zeros(modesequece.shape, dtype=int)
    #     for i in range(T):
    #         bool1 = i > start_timesequence
    #         bool2 = i <= end_timesequence
    #         bool_temp = bool1 * bool2 + temp
    #         bool_temp = self.Renewable_resource - np.dot(bool_temp, mode_using_renewable_resource)
    #         if (bool_temp >= 0).all():
    #             continue
    #         else:
    #             return False
    #     return True

    def Is_Renewable_Resource_Feasible(self, modeSeq, actSeq, timeSeq, crtTime):
        # 判断给定的模式向量及各个模式开始时间是否满足可更新资源限制
        # give sequence of mode and activity, determine whether satisfy the renewable resource constraints
        assert len(timeSeq) + 1 == len(modeSeq) == len(actSeq)
        if (crtTime < np.array(timeSeq)).any():
            return False
        pastModeSeq = np.array(modeSeq[:-1], dtype = np.int)
        pastActSeq = np.array(actSeq[:-1], dtype = np.int)
        currentMode = modeSeq[-1]
        currentAct = actSeq[-1]
        durSeq = []
        for mode, act in zip(pastModeSeq, pastActSeq):
            durSeq.append(self.get_current_duration(mode, act))
        maskfinished = (np.array(timeSeq) + np.array(durSeq)) <= crtTime
        unfinishedActs = pastActSeq[~maskfinished]
        unfinishedModes = pastModeSeq[~maskfinished]
        renewR = np.zeros_like(self.Renewable_resource)
        for mode, act in zip(unfinishedModes, unfinishedActs):
            renewR += self.get_current_mode_using_renewable_resource(mode, act)
        AvaRenewR = self.Renewable_resource - renewR
        current_mode_using_renewable_resource = self.get_current_mode_using_renewable_resource(currentMode, currentAct)

        # endTime = crtTime + current_duration
        if (current_mode_using_renewable_resource <= AvaRenewR).all():
            return True
        return False

    def Is_NonRenewable_Resource_Feasible(self, currentMode, currentAct):
        # 判断给定的模式Activity是否满足不可更新资源限制
        current_mode_using_nonrenewable_resource = self.get_current_mode_using_nonrenewable_resource(currentMode, currentAct)
        if (current_mode_using_nonrenewable_resource <= self.Nonrenewable_resource).all():
            self.Nonrenewable_resource -= current_mode_using_nonrenewable_resource
            return True
        return False

######!!!!!!!!!!!!!###########
    def buildGraph(self, modeSeq, actSeq):
        # 根据给定的活动列表以及该活动选择的模式，构建各个mode之间的时间限制矩阵 actSeq.shape = modeSeq.shape[1]
        # 构建sslimit矩阵
        n = len(self.activities) # modeSeq.shape[1]
        sslimit = np.ones([n, n], dtype=int) * (-50000)
        for i in range(n):
            t = self.SsLimit[str(actSeq[i])] # dict
            ni = modeSeq[0, i]
            for key in t.keys():
                if key in actSeq:
                    p = actSeq[int(key)]
                    nj = modeSeq[0, p]
                    sslimit[i, p] = t[key][ni, nj]
        return sslimit # adj-matrix graph for the past and current activities

    def Isvalid(self, M):
        # 判断sslimit矩阵中是否存在正环，输入为ShortestPath函数输出的M矩阵
        if isinstance(M, bool):
            return M
        else:
            for i in range(M.shape[0]):
                if M[i, i] > 0:
                    return False
        return True

    def ShortestPath(self, graph):
        # 计算sslimit中，各个活动之间的最短距离，输入为buildGraph函数输出的sslimit矩阵
        # 输出矩阵M[i,j]代表第i个活动到第j个活动的最短距离
        # M = graph.copy()
        try:
            dist_matrix = johnson(graph)
        except:
            n = graph.shape[0]
            dist_matrix = graph.copy()
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        dist_matrix[i, j] = max(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])
        return dist_matrix

    def decodeTimeSequece(self, time_chromosome, M, d_ba):
        # 根据最短距离矩阵，进行时间序列解码，设置最后一个序列的最晚开始时间为2*（整个项目的最短时间）
        INF = - (np.max(M) + d_ba)
        start_timesequence = np.zeros([1, self.n], dtype=int)
        M[np.where(M < INF)] = INF
        N = self.ShortestPath(M)
        if not self.Isvalid(N):
            return False, start_timesequence, N

        finished = ['0']
        activities = self.activities.copy()
        activities.remove('0')
        while len(activities) != 0:
            uncode_key = activities.pop(0)
            uncodeindex = int(uncode_key)
            es = []
            ls = []
            for coded_key in finished:
                codedindex = int(coded_key)
                es.append(start_timesequence[0, codedindex] + N[codedindex, uncodeindex])
                ls.append(start_timesequence[0, codedindex] - N[uncodeindex, codedindex])
            start_timesequence[0, uncodeindex] = (min(ls) - max(es)) * time_chromosome[0, uncodeindex] + max(es)
        return True, start_timesequence, N



# if __name__ == '__main__':
#    p = readData()
#    p.ReadData()
#    population = []
#    graphlist = []
#    for i in range(1):
#        mode_chromosome = p.randomMode()
#        #psp2 Valid mode
#        #mode_chromosome = np.array([[1,2,1,1,3,1,2,1,3,2,1,3,3,3,3,1,2,1,3,2,1,1,1,3,3,3,2,3,2,1,1,1,3,1,1,2,1,2,1,1,1,2,1,3,2,2,2,1,2,3,1,3,2,3,3,2,2,3,1,2,2,3,1,2,2,2,1,3,1,2,3,3,2,2,1,3,3,1,3,1,2,3,1,1,3,3,2,1,1,1,1,3,2,2,2,2,3,1,2,3,1,1]])
#        sslimit = p.buildGraph(mode_chromosome)
#        M = p.ShortestPath(sslimit)
#        print(i)
#        if p.IsModeValid(M):
#            print(mode_chromosome)
#            for j in range(1):
#                time_chromosome = p.randomTime()
#                time = p.decodeTimeSequece(time_chromosome, M)
#                print(time)
