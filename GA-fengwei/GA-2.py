# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 18:05:48 2021

@author: fengwei
"""

from data import readData
import numpy as np
import random
import matplotlib.pyplot as plt
import time

T_target = 20
penalty_coeff1 = 5
penalty_coeff2 = 6
d_ba = 30
#待分析文件路径
filepath = r'../test005.sch'
#最紧缺资源优先选择（'min'）or综合资源优先选择（'sum'）or完全随机选择（'none'）
para = 'sum'
#构造readData类的实例个体，同时可以使用类中的各类方法以及数据
p = readData(filepath, para)
#读取数据
p.ReadData()

#price_renewable_resource = 100 * np.ones_like(p.Renewable_resource)
price_renewable_resource = np.array([[2,3]])
#price_nonrenewable_resource = 10 * np.ones_like(p.Nonrenewable_resource)
price_nonrenewable_resource = np.array([[4,5]])

activity_least_time = p.get_activity_least_time()

#返回特定模式序列下的代价
def Cost(modesequece,T):
#modesequece:模式数组，存储每个工序所使用的模式序号,类型为np.array,shape=[1,n]
#T:整个工序的周期，int

    duration = p.get_duration(modesequece)
    mode_using_renewable_resource = p.get_mode_using_renewable_resource(modesequece)
    mode_using_nonrenewable_resource = p.get_mode_using_nonrenewable_resource(modesequece)
    wi = p.get_wi(modesequece)
    
    cost = 0
    penalty_coeff = 0
    if T > T_target:#超期
        penalty_coeff = penalty_coeff2
    else:
        penalty_coeff = penalty_coeff1
    cost0 = penalty_coeff * (T-T_target)
    cost1 = np.sum(np.dot(duration, mode_using_renewable_resource) * price_renewable_resource)
    cost2 = np.sum(np.sum(mode_using_nonrenewable_resource, axis=0, keepdims=True) * price_nonrenewable_resource)
    cost3 = np.sum(wi * (duration-activity_least_time))
    cost = cost0 + cost1 + cost2 + cost3
    return cost
    
def initializeMode(mode_num = 10):
    initial_mode = []
    initial_cost = []
    num = 0
    print('start initialize mode population..............')
    while len(initial_mode) < mode_num:
        #生成一条mode染色体
        state, mode_chromosome = p.randomMode()
        num = num + 1
#        print('run {0} times mode initialize'.format(num))
        #这里发现生成的染色体一定满足时间限制，但大大的不满足资源限制导致多数染色体无效，因此在生成染色的时候资源使用占比小的有较高概率被选中
        #资源选择规则由para参数确定
        if not state: 
#            print('第{0}次随机生成mode失败'.format(num))
            continue
        #生成成功   
        #判断是否满足不可更新资源
        if not p.Is_NonRenewable_Resource_Feasible(mode_chromosome):
#            print('第{0}次生成的个体不满足不可更新资源限制'.format(num))
            continue
        sslimit = p.buildGraph(mode_chromosome, p.activities)
        M = p.ShortestPath(sslimit)
        T = M[0, p.n-1]
        #满足条件的mode加入种群
        chromosome = dict()
        chromosome['mode'] = mode_chromosome
        chromosome['T'] = T
        chromosome['cost'] = 0
        insertInModePopulation(initial_cost, initial_mode, chromosome)

#        print('run {0} times,find {1} modesequece meet the requirements'.format(num,len(initial_mode)))
    return initial_mode, initial_cost

#父母模式交叉，交叉点作为所有环的分割点，分割点前的所有环取父代染色体，分割点后的所有环取母代染色体
#模式变异，变异点在环中则需要验证变异之后的环是否满足条件，否则重新变异
def modecrossoverAndmutate(father, mother, mutate_num):
    child_mode = np.zeros([1, p.n], dtype=int)
    #crossover
    num = len(p.strong_connect_components)
    mode_crossover_index = random.randint(0, num - 1)
    for i in range(num):
        for j in range(len(p.strong_connect_components[i])):              
            index = p.activities.index(p.strong_connect_components[i][j])
            if i < mode_crossover_index:
                child_mode[0, index] = father[0, index]
            else:
                child_mode[0, index] = mother[0, index]
    #mutate    
    isvalid = False
    for i in range(mutate_num):
        while(not isvalid):    
            child_mode_temp = child_mode.copy()
            mode_mutate_index = random.randint(0,p.n - 1)
            child_mode_temp[0,mode_mutate_index] = random.randint(1,p.Activity_mode_Num[mode_mutate_index])
            if child_mode_temp[0,mode_mutate_index] == child_mode[0,mode_mutate_index]:
                break
            act = p.activities[mode_mutate_index]
            for i in range(num):
                if act in p.strong_connect_components[i]:
                    cycle = p.strong_connect_components[i]
                    mode_temp = np.zeros([1,len(cycle)],dtype = int)
                    for j in range(len(cycle)):
                        index = p.activities.index(cycle[j])
                        mode_temp[0,j] = child_mode_temp[0,index]
                    sslimit = p.buildGraph(mode_temp,cycle)
                    M = p.ShortestPath(sslimit)
                    isvalid = p.Isvalid(M)
                    if isvalid:
                        child_mode = child_mode_temp
                    break
    return child_mode
    
def timecrossoverAndmutate(father,mother,mutate_num):
    time_crossover_index = random.randint(0,p.n-1)
    child_time = np.zeros([1,p.n],dtype = float)
    child_time[[0],0:time_crossover_index] = father[[0],0:time_crossover_index]
    child_time[[0],time_crossover_index:] = mother[[0],time_crossover_index:]
    for i in range(mutate_num):
        time_mutate_index = random.randint(1,p.n-1)
        child_time[[0],time_mutate_index] = random.randint(0, 99)/1000
    return child_time

def GA_Mode(initial_cost, initial_mode, population_num = 10, number_iter = 10, stop = -1, show = False, mutate_num = 1):
    if(stop < 0):
        stop = number_iter
    parent_num = int(0.2 * population_num)
    costlist_now = initial_cost
    population_now = initial_mode   
    print('-------------------------------------------')
    print('start mode crossover and mutate..............')    
    diffcost = []
    cost = []
    iters = []
    if show:
        plt.figure(1)
    for i in range(number_iter):
        print('{0}th generation'.format(i))
        cost.append(costlist_now[0])
        if(i>0):
            diffcost.append(cost[i] - cost[i-1])
        if(i>stop):
            if len(set(diffcost[-stop:])) == 1:
                print('{0}次迭代未更新最优值，提前终止迭代'.format(stop))
                break
        iters.append(i)
        if show:
            plt.clf()             
            plt.plot(iters,cost,'*-')  
            plt.title('min cost with mode iterations')
            plt.xlabel('iterations')
            plt.ylabel('min cost')
            plt.pause(0.1)        
            plt.ioff() 
        if (max(costlist_now)-costlist_now == 0).all():
            probability = list(1/len(costlist_now)*np.ones(len(costlist_now),dtype = float))
        else:
            probability = (max(costlist_now)-costlist_now)/sum(max(costlist_now)-costlist_now)

        #亲代中表现较好的parent_num个染色体直接进入下一代
        costlist_new = costlist_now[0:parent_num-1]
        population_new = population_now[0:parent_num-1]        
        num = 0
        #交叉变异
        while len(population_new) < population_num:
            #亲代样本中选择父母
            father = select(population_now, probability)
            mother = select(population_now, probability)

            father_mode = father['mode']
            mother_mode = mother['mode']
            
            child_mode = modecrossoverAndmutate(father_mode,mother_mode,mutate_num) 
            num = num + 1 
#            print('crossover and mutate {0} times'.format(num))                       
            sslimit = p.buildGraph(child_mode,p.activities)
            M = p.ShortestPath(sslimit)
            if not p.Isvalid(M):
#                print('交叉变异后的个体不满足时间限制')
                continue
                #判断模式是否不可更新资源满足
            if not p.Is_NonRenewable_Resource_Feasible(child_mode):
#                print('第{0}次遗传生成的子代不满足不可更新资源限制'.format(num))
                continue
            #满足条件的mode加入种群
            chromosome = dict()
            chromosome['mode'] = child_mode
            chromosome['T'] = M[0, p.n-1]
            chromosome['cost'] = 0
            insertInModePopulation(costlist_new,population_new,chromosome)            
#            print('{0} generation, crossover {1} times and get {2} mode for next generation'.format(i,num,len(population_new)))
        print('-------------------------------------------')
        #完成一次迭代后，用当前种群替代亲代种群
        costlist_now = costlist_new
        population_now = population_new
    return costlist_now,population_now
            
def insertInModePopulation(costlist,initial_population,chromosome):
    mode_chromosome = chromosome['mode']
    T = chromosome['T']
    cost = Cost(mode_chromosome, T)
    chromosome['cost'] = cost
    insertFlag = False
    for j in range(len(initial_population)):
        if cost > initial_population[j]['cost']:
            continue
        else:
            if cost == initial_population[j]['cost']:
                if (mode_chromosome - initial_population[j]['mode'] == 0).all():
                    print('该modesequece已经存在种群中')
                    return
            insertFlag = True
            initial_population.insert(j, chromosome)
            costlist.insert(j, cost)
            break
    if not insertFlag:
        initial_population.append(chromosome)
        costlist.append(cost) 

def insertInTimePopulation(bestMode, costlist, initial_population, chromosome):
    T = chromosome['T']   
    cost = Cost(bestMode, T)
    chromosome['cost'] = cost
    insertFlag = False
    for j in range(len(initial_population)):
        if cost > initial_population[j]['cost']:
            continue
        else:
            insertFlag = True
            initial_population.insert(j, chromosome)
            costlist.insert(j, cost)
            break
    if not insertFlag:
        initial_population.append(chromosome)
        costlist.append(cost)  

def searchBestModeAndTime(modepopulation, modecost, initialnum = 10, generationnum = 10, number_iter = 10, stop = -1, show = False, mutate_num = 1):
    bestCost = float("inf")
    searchNum = 0
    while len(modepopulation) > 0:
        searchNum += 1
        print('可供搜索的Mode个数为:', len(modepopulation))
        currentMode = modepopulation.pop(0)
        modecost.pop(0)
        print('当前搜索的mode:', currentMode)
        initial_cost, initial_time = initializeTime(currentMode, initialnum)
        if len(initial_time) < initialnum:
            print('当前搜索的mode:'+str(currentMode)+'无法找到符合要求的开始时间序列')
            continue
        _, costlist, population_time = GA_Time(currentMode, initial_cost, initial_time, population_num=generationnum,
                                             number_iter=number_iter, mutate_num=mutate_num, stop=stop, show=show)
        if costlist[0] < bestCost:
            bestCost = costlist[0]
            bestMode = currentMode
            bestTime = population_time[0]
            deleteAllBadMode(modepopulation, modecost, bestCost)
    if bestCost < float("inf"):
        print('搜索', searchNum, '个mode即得到最优解')
    else:
        print('搜索所有序列未找到满足条件的开始时间序列，请调整参数重试')
        bestTime = 0
        bestCost = 0
        bestMode = 0
    return bestMode, bestTime, bestCost


def deleteAllBadMode(modepopulation, modecost, bestCost):
    for i, cost in enumerate(modecost):
        if cost >= bestCost:
            break
    del modepopulation[i:]
    del modecost[i:]


def initializeTime(bestMode,population_num = 10):
#初始化模式种群，输入population_num为种群数量,返回为染色体列表以及代价列表
    print('start initialize time population..............')
    initial_time = []
    initial_cost = []
    sslimit = p.buildGraph(bestMode['mode'],p.activities)
    M = p.ShortestPath(sslimit)
    num = 0
    while len(initial_time) < population_num:
        #生成一条开始时间染色体，范围为[0-1)，保留两位小数
        time_chromosome = p.randomTime()
        state, start_timesequence, ES_LS_Matrix = p.decodeTimeSequece(time_chromosome, M, d_ba)
        num = num + 1
        if num > maxsearch_factor * population_num:
            break
#        print('run {0} times initialize'.format(num))
        if not state:
            print('项目的最晚结束时间设置不合理')
            continue
        T = p.getProjectTime(start_timesequence,bestMode['mode'])
        #判断是否满足可更新资源限制
        if not p.Is_Renewable_Resource_Feasible(bestMode['mode'], start_timesequence, T):
#            print('第{0}次生成的时间序列不满足可更新资源限制'.format(num))
            continue
        chromosome = dict()
        chromosome['time_code'] = time_chromosome
        chromosome['start_time'] = start_timesequence
        chromosome['T'] = T
        chromosome['cost'] = 0
        #插入种群中，并保证种群列表有序性及各异性
        insertInTimePopulation(bestMode['mode'], initial_cost,initial_time,chromosome)
#        print('run {0} times,find {1} timesequece'.format(num,len(initial_time)))
    return initial_cost, initial_time
    

def GA_Time(bestMode, initial_cost, initial_time, population_num=10, number_iter=10, stop=-1, show=False, mutate_num=1):
    if(stop < 0):
        stop = number_iter
    print('-------------------------------------------')
    print('start time crossover and mutate..............')   
    costlist_now = initial_cost
    population_now = initial_time
    parent_num = int(0.2*population_num)
    sslimit = p.buildGraph(bestMode['mode'], p.activities)
    M = p.ShortestPath(sslimit)
    diffcost = []
    cost = []
    iters = []
    if show:
        plt.figure(2) 
    for i in range(number_iter):
        print('{0}th generation'.format(i))
        cost.append(costlist_now[0])
        if(i>0):
            diffcost.append(cost[i] - cost[i-1])
        if(i>stop):
            if len(set(diffcost[-stop:])) == 1:
                print('{0}次迭代未更新最优值，提前终止迭代'.format(stop))
                break
        iters.append(i)
        if show:
            plt.clf()             
            plt.plot(iters,cost,'*-')  
            plt.title('min cost with Time iterations')
            plt.xlabel('iterations')
            plt.ylabel('min cost')
            plt.pause(0.1)        
            plt.ioff() 
        num = 0
        #计算亲代中个体被选中的概率
        if (max(costlist_now)-costlist_now == 0).all():
            probability = list(1/len(costlist_now)*np.ones(len(costlist_now),dtype = float))
        else:
            probability = (max(costlist_now)-costlist_now)/sum(max(costlist_now)-costlist_now)
        #亲代中表现较好的parent_num个染色体直接进入下一代
        costlist_new = costlist_now[0:parent_num-1]
        population_new = population_now[0:parent_num-1]  

        #交叉变异
        while len(population_new) < population_num:
            #亲代样本中选择父母
            father = select(population_now, probability)
            mother = select(population_now, probability)

            father_time = father['time_code']
            mother_time = mother['time_code']

            child_time = timecrossoverAndmutate(father_time, mother_time, mutate_num)
            state,start_timesequence, ES_LS_Matrix = p.decodeTimeSequece(child_time, M, d_ba)
            num = num + 1
#            print('crossover and mutate {0} times'.format(num))
            if not state:
                print('项目的最晚结束时间设置不合理')
                continue
            T = p.getProjectTime(start_timesequence,bestMode['mode'])              
            #判断是否满足可更新资源限制
            if not p.Is_Renewable_Resource_Feasible(bestMode['mode'], start_timesequence, T):
#                print('第{0}次遗传生成的子代不满足可更新资源限制'.format(num))
                continue
            chromosome = dict()
            chromosome['time_code'] = child_time
            chromosome['start_time'] = start_timesequence
            chromosome['T'] = T
            chromosome['cost'] = 0
            #插入种群中，并保证种群列表有序性及各异性
            insertInTimePopulation(bestMode['mode'],costlist_new,population_new,chromosome)
            
#            print('{0}th generation, crossover {1} times and get {2} timesequece for next generation'.format(i,num,len(population_new)))
        print('-------------------------------------------')
        #完成一次迭代后，用当前种群替代亲代种群
        costlist_now = costlist_new
        population_now = population_new
    return bestMode, costlist_now, population_now
                

#根据概率来选择样本，两者列表的长度需要一致，且概率列表和为1
def select(population, probability):
    sum_ = 0
    ran = random.random()
    for chromosome, r in zip(population, probability):
        sum_ += r
        if ran < sum_ :break
    return chromosome

if __name__ == '__main__':
    maxsearch_factor = 100
    start = time.process_time()
    initial_mode, initial_cost = initializeMode(30)
    costmode, mode = GA_Mode(initial_cost, initial_mode, 30, 50, mutate_num=1, stop=10)
    bestMode, bestTime, bestCost = searchBestModeAndTime(mode, costmode, initialnum=200, generationnum=200,number_iter=50,
                                                       mutate_num=1, stop=20)
    end = time.process_time()
    if bestCost == 0:
        print('未找到合适的解')
    else:       
        print('bestMode:', bestMode['mode'])
        print('bestTime:', bestTime['start_time'])
        print('Activity_Time:', bestTime['T'], '-----cost:', bestCost)
        print('Running time: %s Seconds' % (end - start))
#    bestMode,initial_cost,initial_time = initializeTime(mode,30)
#    _,costlist,population_time = GA_Time(bestMode,initial_cost, initial_time,30,50,mutate_num=3,stop=10)
#    print('bestMode:',bestMode['mode'])
#    print('bestTime:',population_time[0]['start_time'])
#    print('Activity_Time:',population_time[0]['T'],'-----cost:',costlist[0])

