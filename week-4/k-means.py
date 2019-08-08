import numpy as np 
import matplotlib.pyplot as plt

flag = 0
# 欧式距离
def euclidDist(x, y):
    return np.sqrt(np.sum(np.square(np.array(x)-np.array(y))))

# 曼哈顿距离
def manhattanDist(x, y):
    return np.sum(np.abs(x,y))

# 随机选取中心
def randCenter(points, k):
    index = []
    while len(index)<k:
        temp = np.random.randint(0, len(points)-1)
        if temp not in index:
            index.append(temp)
    return np.array([points[i] for i in index])

# 选取前ｋ个数据作为中心
def orderCenter(points, k):
    return np.array(points[:k])

# 计算新的中心点
def clusterCenter(points):
    return sum(np.array(points))/len(points)

# 聚类算法
def kMeans(datasets, dist, center, k):
    global flag
    allKinds = []  # 用于存放所有结果的中间值
    for _ in range(k):
        temp = []
        allKinds.append(temp)
    # 将数据分成ｋ类
    # print(center, end='\n')  # center没问题
    for i in datasets:
        temp = []  # 用于存放每个点到每个中心点的距离
        for j in center:
            temp.append(dist(i, j))
            # print(dist(i, j), i, j, '\n', center, end='\n')
        allKinds[temp.index(min(temp))].append(i)
    # 打印中间结果
    for i in range(k): 
        print('第'+str(i)+'组:', allKinds[i], end='\n') 
    flag += 1 
    print('************************迭代'+str(flag)+'次***************************')
    # 更新中心点
    center_ = np.array([clusterCenter(i) for i in allKinds])
    print(center, '\n', '\n', center_, '\n', '\n')
    if (center_ == center).all():
        print("结束")
        for i in range(k):
            print('第'+str(i)+'组的聚类中心是:', center[i])
            plt.scatter([j[0] for j in allKinds[i]], [j[1] for j in allKinds[i]], marker='o')  
        plt.grid()
        plt.show()
    else:
        kMeans(datasets, dist, center_, k)


def main(k):
    x = [np.random.randint(0,50)for i in range(50)]
    y = [np.random.randint(0,50)for i in range(50)]
    points = [[i,j] for i,j in zip(x,y)]
    intialCenter = randCenter(points, k)
    kMeans(datasets=points, dist=euclidDist, center=intialCenter, k=k)

if __name__ == '__main__':
    main(4)