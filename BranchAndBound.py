import heapq


# 示例用的地图是一个邻接矩阵，表示城市之间的距离
graph = [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
]
start = 0

def tsp():
    n = len(graph)
    visited = [False] * n
    visited[start] = True
    pq = [(0, start, [start])]  # 优先队列，每个元素包括路径长度、当前节点和已访问的节点列表
    min_cost = float('inf')  # 当前最小路径长度
    min_path = []

    while pq:
        cost, node, path = heapq.heappop(pq)

        if len(path) == n:  # 所有节点都已访问，回到起始点
            if cost + graph[node][start] < min_cost:
                min_cost = cost + graph[node][start]
                min_path = path + [start]
        else:
            for i in range(n):
                if not visited[i]:
                    new_cost = cost + graph[node][i]
                    if new_cost < min_cost:  # 有希望找到更优解的节点加入优先队列
                        heapq.heappush(pq, (new_cost, i, path + [i]))

    return min_cost, min_path




min_cost, min_path = tsp()
print("最短路径长度:", min_cost)
print("最短路径:", min_path)
