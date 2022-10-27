总体思路：
在构造mode序列时，找出图中的所有强连接分量，对每一个强连接分量单独构造，需要满足不存在正环的要求，然后将所有的构造合并成一个mode序列，这样可以保证生成的mode序列一定不存在正环
同时在选择活动序列的每一个活动的mode时，每个mode并不是平等的概率被选中，具体参考para参数的解释

在进行mode交叉时，选择的交叉点并不是mode序列的交叉点，而是所有强连接分量列表的交叉点，交叉点之前的强连接分量选用父亲的mode值，交叉点之后的强连接分量选用母亲的mode值
这样可以绝对保存生成的子代不包含正环

在进行变异时，如果变异点在环中，那么需要重新考量这个环是不是正环，如果是负环则OK，是正环则抛弃，重新进行变异过程
这样可以保证在交叉变异之后生成的子代仍不存在正环

时间编码则根据矩阵ES_LS_Matrix进行，该矩阵根据sslimit矩阵以及给定的d_算出，d_目前选取的值为2*（最后一个活动的最早开始时间），想要查看该矩阵可以使用
M = p.ShortestPath(sslimit)
state,start_timesequence,ES_LS_Matrix = p.decodeTimeSequece(time_chromosome,M)
这两个指令查看，ES_LS_Matrix在state为True时有效

注意：如果想要检查算法的运行时间，请将GA函数中的show参数置为False之后，再使用time相关的函数才是有效的运行时间


GA-1为一步遗传算法求解最佳的mode和开始时间
		主要参数说明如下：
		filepath
		待分析的文件绝对路径

		para
		资源优先选择模式，由于在构造mode时，完全随机构造会造成大量的序列不满足资源要求，因此采取资源使用量来决定该种模式被选取的概率，资源占用量越小，被选中概率越高
		有三种参数可以选择：
		'min'参数下，模式被选中的概率取决于资源总量最小的那一类资源的使用量；
		'sum'参数下，模式被选中的概率取决于sum(使用第i类资源/i类资源总量,i=1,2...,K0)；
		'none'则完全随机选择模式

		initializePopulation(population_num = 10)
		初始化种群函数
		输入：population_num代表生成种群个体数，默认为10（不给数据的话）
		输出：costlist, initial_population代表生成种群中个体的代价列表，生成的初始种群列表

		GA_OneStep(costlist, initial_population, population_num = 10, number_iter = 10, stop = -1, show = True, mutate_num = 1)
		一步遗传算法迭代函数
		输入：costlist初始种群中个体代价列表，initial_population初始种群列表，这两个参数由initializePopulation函数输出
		population_num代表每一代种群总数
		number_iter代表遗传迭代的代数，迭代次数达到number_iter则终止迭代
		stop代表提前终止迭代的条件，即最近的stop次迭代均没有更新最优代价值则提前终止迭代，如果不赋值或者赋值为负，则不提前终止
		show代表是否则迭代过程中展示种群代价的变化曲线
		mutate_num代表父染色体和母染色体在生成子染色体时变异点的个数
		输出：costlist_now,population_now分别代表迭代完成后的种群中个体的代价列表，种群列表。
		种群列表的第0个元素即为最优值

示例：
if __name__ == '__main__':
    #对于GA1来讲，在一般的dba设置下搜索空间较大，建议种群数量大一点，最好在10^2，10^3或者更大量级
    costlist,initial_population = initializePopulation(1000) 
    #这里建议子代总群数量设置成和初始种群数量相同，迭代次数可以适当大一点，stop设置成20次不更新提前停止
    costlist_now,population_now = GA_OneStep(costlist,initial_population,1000,50,mutate_num=1,stop=20,show=True)
    print('最优mode序列为{0}'.format(population_now[0]['mode']))
    print('最优开始时间序列为{0}'.format(population_now[0]['start_time']))
    print('最小代价值为{0}'.format(costlist_now[0]))

GA-2为两步遗传算法，先搜素mode空间得到最优的mode序列，然后再最有mode序列的前提下搜索开始时间空间，最终输出最优解
		主要参数说明如下：
		filepath，para参数同GA-1

		initializeMode(mode_num = 10)
		初始化mode种群函数
		输入：mode_num代表生成mode种群个体数量，默认为10（不给数据的话）
		输出：initial_mode, initial_cost代表生成的初始mode种群，种群个体对应的代价

		GA_Mode(initial_cost, initial_mode, population_num = 10, number_iter = 10, stop = -1, show = True, mutate_num = 1)
		mode种群使用遗传算法交叉迭代计算最优mode函数
		输入：initial_cost, initial_mode代表初始mode种群个体代价，初始mode种群，该参数由initializeMode函数输出
		population_num代表每一代种群个体总数
		number_iter代表遗传迭代的代数迭，代次数达到number_iter则终止迭代
		stop代表提前终止迭代的条件，即最近的stop次迭代均没有更新最优代价值则提前终止迭代，如果不赋值或者赋值为负，则不提前终止
		show代表是否则迭代过程中展示种群代价的变化曲线
		mutate_num代表父染色体和母染色体在生成子染色体时变异点的个数
		输出：costlist_now,population_now代表迭代之后的mode种群个体代价，mode种群
		种群列表中的第0个元素即为最优mode序列

		initializeTime(mode_population,population_num = 10)
		初始化开始时间种群函数
		输入：mode_population代表迭代输出的mode种群，由GA_Mode函数输出
		population_num代表生成的开始时间种群数量
		输出：bestMode,initial_cost,initial_time分别代表最佳mode序列，初始开始时间种群个体的代价列表，初始开始时间种群列表

		GA_Time(bestMode,initial_cost, initial_time, population_num = 10, number_iter = 10, stop = 5, show = True, mutate_num = 1)
		遗传算法迭代搜索最优时间序列函数
		输入：bestMode代表最优模式，搜索在该mode序列的前提下进行，由initializeTime输出
		initial_cost, initial_time代表初始化开始时间种群个体代价列表，初始化开始时间种群列表，由initializeTime输出
		population_num代表每一代种群个体总数
		number_iter代表遗传迭代的代数，代次数达到number_iter则终止迭代
		stop代表提前终止迭代的条件，即最近的stop次迭代均没有更新最优代价值则提前终止迭代，如果不赋值或者赋值为负，则不提前终止
		show代表是否则迭代过程中展示种群代价的变化曲线
		mutate_num代表父染色体和母染色体在生成子染色体时变异点的个数
		输出：bestMode,costlist_now,population_now分别代表最优的模式序列，迭代后开始时间种群个体的代价，迭代后的开始时间种群
		种群列表中的第0个元素即为最优开始时间序列

		searchBestModeAndTime(modepopulation, modecost, initialnum = 10, generationnum = 10, number_iter = 10, stop = -1, show = True, mutate_num = 1):
		根据第一步搜索出来的modelist，进行第二步搜索获取最佳mode和开始时间序列
		输入：modepopulation代表第一步搜索输出的modelist，modecost代表对应的cost
		initialnum代表初始化时间总群数量
		generationnum 代表遗传后子代的总群数量
		number_iter 代表迭代次数
		stop 代表是否提前终止迭代
		show代表是否画图
		mutate_num代表变异点的个数
		输出：最佳模式，最佳开始时间以及对应的cost

示例：
if __name__ == '__main__':
    maxsearch_factor = 100#最大的初始化搜索倍率，搜索次数为倍率*种群数量
    start = time.process_time()
    #对于GA2来讲，初始化mode种群数量最好不要超过所有模式排列组合数量的0.2，否则可能存在凑不够种群数量的可能，这里所有可能的排列数量为1*1*2*2*4*2*1
    initial_mode, initial_cost = initializeMode(30)
    #子代总群数量设置成和初始种群数量相同，迭代次数可以适当大一点，stop设置成若干次不更新提前停止
    costmode, mode = GA_Mode(initial_cost, initial_mode, 30, 50, mutate_num=3, stop=10)
    #在一般的dba设置下搜索空间较大，建议种群数量大一点，最好在10^2，10^3或者更大量级，初始化种群数量和子代种群数量一致，迭代次数适当大一点，设置提前终止
    bestMode, bestTime, bestCost = searchBestModeAndTime(mode, costmode, initialnum=100, generationnum=100,number_iter=50,
                                                       mutate_num=3, stop=20)
    end = time.process_time()
    if bestCost == 0:
        print('未找到合适的解')
    else:       
        print('bestMode:', bestMode['mode'])
        print('bestTime:', bestTime['start_time'])
        print('Activity_Time:', bestTime['T'], '-----cost:', bestCost)
        print('Running time: %s Seconds' % (end - start))

data为数据读取、数据结构构造、常用方法实现等