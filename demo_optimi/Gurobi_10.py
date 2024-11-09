import pandas as pd
import gurobipy as gp
from geopy.distance import geodesic


def calculate_distance(coord1, coord2):
    distance = geodesic(coord1, coord2).kilometers
    return round(distance,2)

def trans_seconds(data_time):
    temp = data_time.strftime('%H:%M').split(':')
    return int(temp[0])*60+int(temp[1])

# 点类型，用以存储基本点的信息
class Node:
    def __init__(self, coords, demand, time_list, service_time, value):
        self.COORDS = coords
        self.DEMEND = demand
        self.TIME_WINDOW = time_list
        self.SERVICE_TIME = service_time
        self.VALUE = value


data = pd.read_excel('demo_10points.xlsx')
print(data)
node_dict = {0 : Node((data.loc[len(data)-1,'y'],
                       data.loc[len(data)-1,'x']),
                       data.loc[len(data)-1, '需求量'],
                      [trans_seconds(data.loc[len(data)-1,'硬时间窗1']),
                       trans_seconds(data.loc[len(data)-1,'硬时间窗2'])],
                      data.loc[len(data)-1, '服务时间'],
                      data.loc[len(data)-1, '单位价值量'])}

for i in range(len(data)):
    node_dict[i+1] = Node((data.loc[i,'y'],
                            data.loc[i,'x']),
                            data.loc[i, '需求量'],
                            [trans_seconds(data.loc[i, '硬时间窗1']),
                            trans_seconds(data.loc[i, '硬时间窗2'])],
                            data.loc[i, '服务时间'],
                            data.loc[i, '单位价值量'])

nodes = list(node_dict.keys())
nodes_0 = nodes[:-1]
nodes_1 = nodes[1:]
customers = list(node_dict.keys())[1:-1]
k_num = 3
C_depart = 300
p_oil = 8.25
rho_m = 3
rho_0 = 1
Q_M = 2500
b1 = 1
b2 = 2
speed = 60
e = 0.6
p_e = 1.5
p_w = 5
p_l = 10
R = 0.001
scalar_M = 10000

# 计算距离矩阵和时间矩阵
distance_martix = {}
time_martix = {}
for key1, value1 in node_dict.items():
    for key2, value2 in node_dict.items():
        distance_martix[key1, key2] = calculate_distance(value1.COORDS, value2.COORDS)
        time_martix[key1, key2] = distance_martix[key1, key2]/speed*60


# 创建模型
m = gp.Model()

# 添加变量
x = m.addVars(nodes_0, nodes_1, vtype=gp.GRB.BINARY, name='x')
time_delay1 = m.addVars(customers, lb=0, ub=300, vtype=gp.GRB.CONTINUOUS, name='time_delay1')
time_delay2 = m.addVars(customers, lb=0, ub=300, vtype=gp.GRB.CONTINUOUS, name='time_delay2')
arr_time = m.addVars(nodes_0, lb=240, ub=720, vtype=gp.GRB.CONTINUOUS, name='arr_time')
u = m.addVars(nodes_1, ub=Q_M, vtype=gp.GRB.CONTINUOUS, name='u') # 车辆剩余货物量


temp = m.addVars(nodes_0, nodes_1, vtype=gp.GRB.CONTINUOUS, name='x') # 为了线性化满载量
m.addConstrs(temp[i,j] <= scalar_M * x[i,j] for i in nodes_0 for j in nodes_1)
m.addConstrs(temp[i,j] >= ((rho_m-rho_0)/Q_M*u[j]+rho_0) - 
             scalar_M * (1-x[i,j]) for i in nodes_0 for j in nodes_1)
m.addConstrs(temp[i,j] <= ((rho_m-rho_0)/Q_M*u[j]+rho_0) + 
             scalar_M * (1-x[i,j]) for i in nodes_0 for j in nodes_1)

# 固定成本
C_fix = m.addVar(vtype=gp.GRB.CONTINUOUS, name='C_fix')
m.addConstr(C_fix == C_depart * gp.quicksum(x[0,j] for j in customers))

# 运输成本，剩余货物量算消耗，乘以距离
C_dis = m.addVar(vtype=gp.GRB.CONTINUOUS, name='C_dis')
m.addConstr(C_dis == gp.quicksum(temp[i,j] * 
     p_oil * distance_martix[i,j] for i in nodes_0 for j in nodes_1))

# 冷藏成本
C_refir = m.addVar(vtype=gp.GRB.CONTINUOUS, name='C_refir')
m.addConstr(C_refir == b1 * gp.quicksum(x[i,j]*time_martix[i,j] \
    for i in nodes_0 for j in nodes_1) + b2*sum(node_dict[i].SERVICE_TIME for i in customers))


# 碳排放成本
C_carbon = m.addVar(vtype=gp.GRB.CONTINUOUS, name='C_carbon')
C_1 = e * p_e * gp.quicksum(temp[i,j] *
     distance_martix[i,j] for i in nodes_0 for j in nodes_1)
C_2 = e * p_e * C_refir
m.addConstr(C_carbon == C_1 + C_2)


# 惩罚成本
C_penalty = m.addVar(vtype=gp.GRB.CONTINUOUS, name='C_penalty')
m.addConstr(C_penalty == gp.quicksum(node_dict[i].VALUE*(
            p_w*time_delay1[i] + p_l*time_delay2[i]) for i in customers))


# 货损
C_qual = m.addVar(vtype=gp.GRB.CONTINUOUS, name='C_qual')
m.addConstr(C_qual == gp.quicksum(R * node_dict[i].VALUE * node_dict[i].DEMEND * 
                (arr_time[i]-240) for i in customers))

# 建立目标函数
m.setObjective(C_fix + C_dis + C_refir + C_carbon + C_penalty + C_qual, gp.GRB.MINIMIZE)


# 文中基础约束
m.addConstrs(gp.quicksum(x[i,j] for j in nodes_1) == \
             gp.quicksum(x[j,i] for j in nodes_0) for i in customers)
m.addConstrs(1 == \
             gp.quicksum(x[i,j] for j in nodes_1) for i in customers)

# 计算到达时间及时间窗延误
m.addConstrs(time_delay1[i] >= node_dict[i].TIME_WINDOW[0] - arr_time[i] for i in customers)
m.addConstrs(time_delay2[i] >= arr_time[i] - node_dict[i].TIME_WINDOW[1] for i in customers)

# 所有车从0点开出，回到终点
m.addConstr(gp.quicksum(x[0,j] for j in nodes_1) <= k_num)
m.addConstr(gp.quicksum(x[i,nodes[-1]] for i in nodes_0) == gp.quicksum(x[0,j] for j in nodes_1))

# 时间接续约束
m.addConstrs(arr_time[j] >= arr_time[i]+time_martix[i,j]+node_dict[i].SERVICE_TIME - scalar_M*(1-x[i,j]) \
             for i in nodes_0 for j in customers)
m.addConstrs(arr_time[j] <= arr_time[i]+time_martix[i,j]+node_dict[i].SERVICE_TIME + scalar_M*(1-x[i,j]) \
             for i in nodes_0 for j in customers)



# 线路容量约束
m.addConstrs(u[i] >= u[j] + node_dict[i].DEMEND - (1-x[i,j])*scalar_M for i in customers for j in nodes_1)
m.addConstrs(u[i] <= u[j] + node_dict[i].DEMEND + (1-x[i,j])*scalar_M for i in customers for j in nodes_1)


# 其余合理性约束
m.addConstrs(x[i,i] == 0 for i in customers) # 不能同自己进行接续
m.addConstr(arr_time[0] == 240)




m.Params.MIPGap = 0.0005 # 设置求解混合整数规划的 Gap 
m.Params.MIPFocus = 3 # 以缩小Gap为导向
m.Params.MinRelNodes = 10000
m.optimize()
# m.computeIIS()
# m.write('iii.ilp')


solution_x = m.getAttr('X', x)
for k,v in solution_x.items():
    if v>0.9:
        print(k)
print('使用车辆数量', C_fix.X/C_depart)
print('固定成本', C_fix.X)
print('运输成本', C_dis.X)
print('冷藏成本', C_refir.X)
print('碳排放成本', C_carbon.X)
print('惩罚成本', C_penalty.X)
print('货损成本', C_qual.X)


p=0










