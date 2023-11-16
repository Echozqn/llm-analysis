
"""
0      BackwardPass     0  1.188915
1      BackwardPass     1  0.838724
2      BackwardPass     2  0.976269
3       ForwardPass     0  0.362812
4       ForwardPass     1  0.249911
5       ForwardPass     2  0.328812
"""
# comm_time = 0.07
# bwd_time = [3*x for x in fwd_time]
# fwd_time = [0.244087,0.368211,0.324038]
# bwd_time = [1.057293,1.081277,0.962686]
#
# fwd_time = [0.261343,0.388382,0.366843]
# bwd_time = [0.718263,1.082689,1.005779]

fwd_time = [0.388382,0.366843,0.261343]
bwd_time = [1.082689,1.005779,0.718263]

def solve1(fwd_time,bwd_time):
    fwd_total_time = 0
    bwd_total_time = 0
    n = len(fwd_time)
    pre = 0
    for i in range(n):
        cur = pre + fwd_time[i]*4
        fwd_total_time = max(fwd_total_time,cur)
        pre += fwd_time[i]
    pre = 0
    for i in range(n-1,-1,-1):
        cur = pre + bwd_time[i]*4
        bwd_total_time = max(bwd_total_time,cur)
        pre += bwd_time[i]
    return fwd_total_time + bwd_total_time

def solve2(fwd_time,bwd_time):
    print(fwd_time,bwd_time)
    n = len(fwd_time)
    print(f"duration = {n * (fwd_time[-1]+bwd_time[-1])}")
    total = sum(fwd_time) +sum(bwd_time) + 3 * (fwd_time[-1]+bwd_time[-1])
    return total
def solve3(fwd_time,bwd_time):
    n = len(fwd_time)
    num_microbatch = 4
    from collections import defaultdict
    graph = defaultdict(list)
    edges = defaultdict(list)
    for i in range(n):
        for j in range(num_microbatch):
            key = f"{i}_{j}_F"
            graph[key] = [fwd_time[i],[]]
            edges[key] = [fwd_time[i], []]
            if i > 0:
                graph[key][1].append(f'{i-1}_{j}_F')
            if j > 0:
                graph[key][1].append(f'{i}_{j-1}_F')
    for i in range(n-1,-1,-1):
        for j in range(num_microbatch):
            key = f"{i}_{j}_B"
            graph[key] = [bwd_time[i], []]
            edges[key] = [bwd_time[i], []]
            graph[key][1].append(f'{i}_{j}_F')
            if i < n - 1:
                graph[key][1].append(f'{i+1}_{j}_B')
            if j > 0:
                graph[key][1].append(f'{i}_{j-1}_B')

    for node in graph:
        for v in graph[node][1]:
            edges[v][1].append(node)
    # Step 1: Forward pass to calculate ES and EF
    ins = defaultdict(int)
    import queue
    dist = {}
    que = queue.Queue()
    for node in graph:
        if len(graph[node][1]) == 0:  # If no prerequisites
            dist[node] = [0,graph[node][0]]
            que.put(node)
    while not que.empty():
        u = que.get()
        for v in edges[u][1]:
            ins[v] += 1
            if ins[v] == len(graph[v][1]):
                if v not in dist or dist[u][1] + graph[v][0] > dist[v][1]:
                    dist[v] = [dist[u][1],dist[u][1] + graph[v][0]]
                que.put(v)
    for i in range(num_microbatch):
        print(dist[f"1_{i}_F"])
        print(dist[f"1_{i}_B"])
    print(dist)
    total_time = max([times[1] for times in dist.values()])
    print(f"total_time = {total_time}")
    return total_time



solve3([0.5,1],[1,2])
# solve3(fwd_time,bwd_time)
# print(f"time = {solve2(fwd_time,bwd_time)}")
# print(f"time = {solve1(fwd_time,bwd_time)}")
# print(f"fwd_total_time = {fwd_total_time} ms;bwd_total_time = {bwd_total_time} ms; total = {(fwd_total_time + bwd_total_time)/1000 + (n - 1) * comm_time} ms")