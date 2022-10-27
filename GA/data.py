# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:29:52 2021

@author: Administrator
"""
import numpy as np
import tarjan
import random

MaxSearchPerIteration = 10000

class readData():
    def __init__(self,filename,para = 'none'):
        self.filename = filename
        self.para = para
        
    def get_activity_least_time(self):
        activity_least_time = []
        for i in range(len(self.Activity_mode_Num)):
            temptime = []
            for j in range(self.Activity_mode_Num[i]):
                key = self.activities[i] + '-' + str(j + 1)
                temptime.append(self.Resources[key]['duration'])
            activity_least_time.append(np.min(np.array(temptime)))
        return np.array(activity_least_time).reshape([1,len(self.Activity_mode_Num)])

    def get_wi(self,modesequece):
    #    wi = []
    #    for i in range(modesequece.shape[1]):
    #        key = p.activities[i] + '-' + str(modesequece[0,i])
    #        wi.append(p.Resources[key]['wi'])
    #    return np.array(wi).reshape(modesequece.shape)
        return np.ones_like(modesequece)
    
    def getProjectTime(self,timesequece,modesequece):
        T = 0
        duration = self.get_duration(modesequece)
        for i in range(timesequece.shape[1]):
            if timesequece[0,i] + duration[0,i] > T:
                T = timesequece[0,i] + duration[0,i]
        return T
    
    def get_duration(self,modesequece):
        duration = []
        for i in range(modesequece.shape[1]):
            key = self.activities[i] + '-' + str(modesequece[0,i])
            duration.append(self.Resources[key]['duration'])
        return np.array(duration).reshape(modesequece.shape)
    
    #返回特定模式下消耗的可更新资源矩阵,输入modesequece = np.array,shape = [1,n],返回mode_using_renewable_resource = np.array,shape = [n,K0]
    def get_mode_using_renewable_resource(self,modesequece): 
        mode_using_renewable_resource = []
        for i in range(modesequece.shape[1]):
            key = self.activities[i] + '-' + str(modesequece[0,i])
            mode_using_renewable_resource.append(self.Resources[key]['using_renewable_resource'])
        return np.array(mode_using_renewable_resource).reshape([modesequece.shape[1],self.K0])
    
    #返回特定模式下消耗的不可更新资源矩阵,输入modesequece = np.array,shape = [1,n],返回mode_using_nonrenewable_resource = np.array,shape = [n,K1]
    def get_mode_using_nonrenewable_resource(self,modesequece):
        mode_using_nonrenewable_resource = []
        for i in range(modesequece.shape[1]):
            key = self.activities[i] + '-' + str(modesequece[0,i])
            mode_using_nonrenewable_resource.append(self.Resources[key]['using_nonrenewable_resource'])
        return np.array(mode_using_nonrenewable_resource).reshape([modesequece.shape[1],self.K1])
    
    def Is_Renewable_Resource_Feasible(self,modesequece,start_timesequence,T):
    #modesequece:模式数组，存储每个工序所使用的模式序号,类型为np.array,shape=[1,n]
    #start_timesequence:每个模式的开始时间数组,类型为np.array,shape=[1,n]
    #duration:每个模式的持续时间，类型为np.array,shape=[1,n]
    #T:整个工序的周期，int
    #renewable_resource:可更新资源数组，存储每一种可更新资源数量,类型为np.array,shape=[1,K0]
    #mode_using_renewable_resource:当前模式队列下，每个模式需要消耗的可更新资源,类型为np.array,shape=[n,K0]
        duration = self.get_duration(modesequece)
        mode_using_renewable_resource = self.get_mode_using_renewable_resource(modesequece)
        end_timesequence = start_timesequence + duration
        temp = np.zeros(modesequece.shape,dtype = int)
        for i in range(T):
            bool1 = i > start_timesequence
            bool2 = i <= end_timesequence
            bool_temp =  bool1 * bool2 + temp
            bool_temp = self.Renewable_resource - np.dot(bool_temp,mode_using_renewable_resource)
            if (bool_temp >= 0).all():
                continue
            else:
                return False
        return True
    
    def Is_NonRenewable_Resource_Feasible(self,modesequece):
    #modesequece:模式数组，存储每个工序所使用的模式序号,类型为np.array,shape=[1,n]
    #nonrenewable_resource:不可更新资源数组，存储每一种不可更新资源数量,类型为np.array,shape=[1,K1]
        mode_using_nonrenewable_resource = self.get_mode_using_nonrenewable_resource(modesequece)
        temp = self.Nonrenewable_resource - np.sum(mode_using_nonrenewable_resource,axis = 0,keepdims=True)
        #print('不可更新资源在该modesequece下的富余量{0}'.format(temp))
        return (temp >= 0).all()

    def buildGraph(self,modesequece,act_list):
    #构建sslimit矩阵
        n = modesequece.shape[1]
        sslimit = np.ones([n,n],dtype = int)*(-50000)
        for i in range(modesequece.shape[1]):
            t = self.SsLimit[act_list[i]]
            ni = modesequece[0,i]
            for key in t.keys():
                if key in act_list:
                    p = act_list.index(key)
                    nj = modesequece[0,p]
                    sslimit[i,p] = t[key][ni-1,nj-1]
        return sslimit
    
    def Isvalid(self,M):
    #判断sslimit矩阵中是否存在正环
        for i in range(M.shape[0]):
            if M[i,i] > 0:
                return False
        return True
    
    def select(self,population,probability):
    #根据概率选择相应的元素
        sum_ = 0
        ran = random.random()
        for chromosome, r in zip(population, probability):
            sum_ += r
            if ran < sum_ :break
        return chromosome
    
    def chooseMode(self,act,para = 'min'):
    #根据不可更新资源的数目来确定不同的mode被选中的概率，然后根据概率选择mode。
    #使用资源比例越高的mode被选中的概率越低
        index = self.activities.index(act)
        if self.Activity_mode_Num[index] == 1:
            return 1
        usingResource = np.zeros([self.Activity_mode_Num[index],self.K0])
        for i in range(self.Activity_mode_Num[index]):
            usingResource[[i],:] = self.Resources[act+'-'+str(i+1)]['using_nonrenewable_resource']        
        if para == 'min':
            minresource = np.argmin(self.Nonrenewable_resource)
            pro = usingResource[:,[minresource]].reshape(-1)
        if para == 'sum':
            pro = np.sum(usingResource/self.Nonrenewable_resource,axis = 1)
        if para == 'none':
            return np.random.randint(1,self.Activity_mode_Num[index] + 1)
        mode = np.arange(1,self.Activity_mode_Num[index] + 1).tolist()
        sumpro = np.max(pro) + np.min(pro)
        probability = ((sumpro-pro)/np.sum(sumpro-pro)).tolist()
        return self.select(mode,probability)
        
    def randomMode(self):
    #随机生成modesequece，根据graph中的强连接分量进行一一选择，保证所有的强连接分量均不含正环
    #同时以更高的概率选择资源占比少的mode
        mode_chromosome = np.zeros([1,self.n],dtype = int)
        for i in range(len(self.strong_connect_components)):
            cycle = self.strong_connect_components[i]
            if len(cycle) == 1:
                p = self.activities.index(cycle[0])
                mode_chromosome[0,p] = self.chooseMode(cycle[0],self.para)
            else:
                mode_temp = np.zeros([1,len(cycle)],dtype = int)
                isvalid = False
                num = 0
                while(not isvalid):
                    num = num + 1
                    if(num > MaxSearchPerIteration):
                        print(cycle)
                        return False,mode_chromosome
                    for j in range(len(cycle)):
                        p = self.activities.index(cycle[j])
                        mode_chromosome[0,p] = self.chooseMode(cycle[j],self.para)
                        mode_temp[0,j] = mode_chromosome[0,p]
                        sslimit = self.buildGraph(mode_temp,cycle)
                        M = self.ShortestPath(sslimit)
                    isvalid = self.Isvalid(M)
        return True,mode_chromosome
    
    def randomTime(self):
    #随机生成时间序列，为0-1之间的小数，保留三位
        time_chromosome = np.random.randint(0,1000,[1,self.n])/1000
        time_chromosome[0,0] = 0.0
        return time_chromosome

    def ShortestPath(self,graph):
    #计算sslimit中，各个活动之间的最短距离
        M = graph.copy()
        n = graph.shape[0]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    M[i,j] = max(M[i,j], M[i,k]+M[k,j])
        return M
    
    
    def decodeTimeSequece(self,time_chromosome,M,d_ba):
    #根据最短距离矩阵，进行时间序列解码，设置最后一个序列的最晚开始时间为2*（整个项目的最短时间）
        INF = - (np.max(M) + d_ba)
        start_timesequence = np.zeros([1,self.n],dtype = int)
        M[np.where(M<INF)] = INF
        N = self.ShortestPath(M)
        if not self.Isvalid(N):
            return False,start_timesequence,N
        
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
                es.append(start_timesequence[0,codedindex] + N[codedindex,uncodeindex])
                ls.append(start_timesequence[0,codedindex] - N[uncodeindex,codedindex])
            start_timesequence[0,uncodeindex] = (min(ls) - max(es))*time_chromosome[0,uncodeindex] + max(es)
            finished.append(uncode_key)
        return True,start_timesequence,N
    
    
    def ReadData(self):
    #读取.sch中的数据并转化为相对应的数据结构
        # dlg = win32ui.CreateFileDialog(1)
        # dlg.SetOFNInitialDir('C:\\Users\\Administrator\\Desktop\\help\\Archive\\testset_mm100') 
        # dlg.DoModal()
        # filename = dlg.GetPathName()
        filename = self.filename
        if(len(filename) > 0):    
            print(filename)
            with open(filename, "r") as f:
                line = f.readline()
                line = line[:-1]
                data = line.split('\t')
                self.n = int(data[0]) + 2
                self.K0 = int(data[1])
                self.K1 = int(data[2])
                self.Renewable_resource = np.zeros([1,self.K0],dtype = int)
                self.Nonrenewable_resource = np.zeros([1,self.K1],dtype = int)
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
                        successor_act.append(data[3:int(data[2])+3]) 
                        self.graph[data[0]] = data[3:int(data[2])+3]
                        graph_data.append(data[int(data[2])+3:])
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
                                self.Renewable_resource[0,j] = int(data[j])
                            else:
                                self.Nonrenewable_resource[0,j-self.K0] = int(data[j])
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
                    arr = np.array(t).reshape([ni,nj])
                    i_data[key] = arr
                self.SsLimit[ikey] = i_data

#if __name__ == '__main__':
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
            