# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:57:13 2021

@author: fengwei
"""
from data import readData
import numpy as np
import random
import matplotlib.pyplot as plt

T_target = 27
d_ba = 30
penalty_coeff1 = -20000
penalty_coeff2 = 20000

#待分析文件路径
filepath = r'test005.sch'
#最紧缺资源优先选择（'min'）or综合资源优先选择（'sum'）or完全随机选择（'none'）
para = 'min'
#构造readData类的实例个体，同时可以使用类中的各类方法以及数据
p = readData(filepath,para)
#读取数据
p.ReadData()

price_renewable_resource = 100 * np.ones_like(p.Renewable_resource)
price_nonrenewable_resource = 10 * np.ones_like(p.Nonrenewable_resource)

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
    if T>T_target:#超期
        penalty_coeff = penalty_coeff2
    else:
        penalty_coeff = penalty_coeff1
    cost0 = penalty_coeff * (T-T_target)
    cost1 = np.sum(np.dot(duration,mode_using_renewable_resource) * price_renewable_resource)    
    cost2 = np.sum(np.sum(mode_using_nonrenewable_resource,axis = 0,keepdims=True) * price_nonrenewable_resource)
    cost3 = np.sum(wi * (duration-activity_least_time))
    cost = cost0 + cost1 + cost2 + cost3
    return cost
    
#将模式向量以及计算出的代价插入各自列表中，保证代价列表的有序性
def insertInCostList(costlist,initial_population,chromosome):
    mode_chromosome = chromosome['mode']
    T = chromosome['T']   
    cost = Cost(mode_chromosome, T)
    chromosome['cost'] = cost
    insertFlag = False
    for j in range(len(initial_population)):
        if cost > initial_population[j]['cost']:
                continue
        if cost == initial_population[j]['cost']:
            if (chromosome['mode'] == initial_population[j]['mode']).all() and (chromosome['time_code'] == initial_population[j]['time_code']).all():
                return
        else:
            insertFlag = True
            initial_population.insert(j, chromosome)
            costlist.insert(j, cost)
            break
    if not insertFlag:
        initial_population.append(chromosome)
        costlist.append(cost)

def initializePopulation(population_num = 10):
#初始化模式种群，输入population_num为种群数量,返回为染色体列表以及代价列表
    initial_population = []
    costlist = []
    num = 0
    print('start initialize population..............')
    while len(initial_population) < population_num:        
        #生成一条mode染色体       
        state,mode_chromosome = p.randomMode()
        #这里发现生成的染色体一定满足时间限制，但大大的不满足资源限制导致多数染色体无效，因此在生成染色的时候资源使用占比小的有较高概率被选中
        #资源选择规则由para参数确定
        num = num + 1
        print('{0} 次生成modesequece'.format(num))
        if not state:
            print('{0} 次生成modesequece失败'.format(num))
            continue
        sslimit = p.buildGraph(mode_chromosome,p.activities)
        M = p.ShortestPath(sslimit)
        #判断是否满足不可更新资源
        if not p.Is_NonRenewable_Resource_Feasible(mode_chromosome):
            print('{0} 次生成modesequece不满足不可更新资源限制'.format(num))
            continue
        time_chromosome = p.randomTime()
        #解码获取实际的活动开始时间,以及设定最后工序最晚开始时间d之后各个工序之间时间逻辑关系矩阵ES_LS_Matrix
        state,start_timesequence,ES_LS_Matrix = p.decodeTimeSequece(time_chromosome,M,d_ba)
        T = p.getProjectTime(start_timesequence,mode_chromosome)
        #判断是否满足可更新资源限制
        if not p.Is_Renewable_Resource_Feasible(mode_chromosome, start_timesequence, T):   
            print('{0} 次生成的mode和开始时间序列不满足可更新资源限制'.format(num))
            continue
        chromosome = dict()
        chromosome['mode'] = mode_chromosome
        chromosome['time_code'] = time_chromosome
        chromosome['start_time'] = start_timesequence
        chromosome['T'] = T
        chromosome['cost'] = 0
        #插入种群中，并保证种群列表有序性及各异性
        insertInCostList(costlist,initial_population,chromosome)
        print('run {0} times,find {1} modesequece meet the requirements'.format(num,len(initial_population)))
    return costlist,initial_population

def select(population,probability):
    sum_ = 0
    ran = random.random()
    for chromosome, r in zip(population, probability):
        sum_ += r
        if ran < sum_ :break
    return chromosome

#父母模式交叉，交叉点作为所有环的分割点，分割点前的所有环取父代染色体，分割点后的所有环取母代染色体
#模式变异，变异点在环中则需要验证变异之后的环是否满足条件，否则重新变异
def modecrossoverAndmutate(father,mother,mutate_num):
    child_mode = np.zeros([1,p.n],dtype = int)
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
    
def GA_OneStep(costlist, initial_population, population_num = 10, number_iter = 10, stop = -1, show = True, mutate_num = 1):
#遗传算法求解最优模式，输出最终的种群列表和代价列表，列表第一个值为最优值
#需要输入初始化种群列表和种群代价列表
#parent_num亲代中直接进入下一代的最优的数量
#population_num子代种群数量
#number_iter迭代数量
    parent_num = int(0.2*population_num)
    if(stop == -1):
        stop = number_iter
    costlist_now = costlist
    population_now = initial_population
    diffcost = []
    cost = []
    iters = []
    if show:
        plt.ion() 
    print('-------------------------------------------')
    print('start crossover and mutate..............')
    for i in range(number_iter):
        print('{0} generations'.format(i))
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
            plt.title('min cost with iterations')
            plt.xlabel('iterations')
            plt.ylabel('min cost')
            plt.pause(0.1)        
            plt.ioff()     
        #计算亲代中个体被选中的概率
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
            #模式交叉变异
            child_mode = modecrossoverAndmutate(father_mode,mother_mode, mutate_num)
            num = num + 1
            print('{0}次交叉变异后生成modesequece'.format(num))
            sslimit = p.buildGraph(child_mode,p.activities)
            M = p.ShortestPath(sslimit)
            if not p.Isvalid(M):
                print('{0}次交叉变异后生成modesequece不满足时间限制'.format(num))
                continue

            #判断子代是否满足不可更新资源
            if not p.Is_NonRenewable_Resource_Feasible(child_mode):
                print('{0}次交叉变异后生成modesequece不满足不可更新资源限制限制'.format(num))
                continue
            
            #时间交叉变异
            father_time = father['time_code']
            mother_time = mother['time_code']
            child_time = timecrossoverAndmutate(father_time,mother_time,mutate_num)            
            #解码获取实际的活动开始时间
            state,start_timesequence,ES_LS_Matrix = p.decodeTimeSequece(child_time,M,d_ba)
            if not state:
                print('项目的最晚结束时间设置不合理')
                continue
            T = p.getProjectTime(start_timesequence,child_mode)
            #判断子代是否满足可更新资源
            if not p.Is_Renewable_Resource_Feasible(child_mode, start_timesequence, T):
                print('{0}次交叉变异后生成mode和开始时间序列不满足可更新资源限制'.format(num))
                continue
            chromosome = dict()
            chromosome['mode'] = child_mode
            chromosome['time_code'] = child_time
            chromosome['start_time'] = start_timesequence
            chromosome['T'] = T
            #插入种群列表并保证列表的有序性和种群中个体的各异性
            insertInCostList(costlist_new,population_new,chromosome)
            print('{0} generation, crossover {1} times and get {2} for next generation'.format(i+1,num,len(population_new)))
        print('-------------------------------------------')
        #完成一次迭代后，用当前种群替代亲代种群
        costlist_now = costlist_new
        population_now = population_new        
    return costlist_now,population_now

if __name__ == '__main__':
    costlist,initial_population = initializePopulation(1000)
    costlist_now,population_now = GA_OneStep(costlist,initial_population,1000,50,mutate_num=1,stop=20)
    print('最优mode序列为{0}'.format(population_now[0]['mode']))
    print('最优开始时间序列为{0}'.format(population_now[0]['start_time']))
    print('最小代价值为{0}'.format(costlist_now[0]))

