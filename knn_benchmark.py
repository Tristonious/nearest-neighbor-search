# Tristan Jones
# knn-comparison: Algorithmic Approaches to Efficient Nearest Neighbor Search
# in High-Dimensional Data
#
# Implements and benchmarks five k-NN search algorithms:
#   1. Brute Force
#   2. k-d Tree (exact)
#   3. Randomized k-d Tree Forest ANN (bt=0, bt=1, bt=2)
#
# AI usage: Claude assisted with test scaffolding, visualizations, and debugging.
# All algorithm designs and implementations are original.
# See docs/knn_comparison_paper.pdf for full write-up.
#
# Future work:
#   - Modular refactor (see src/)
#   - Implement ANNOY and HNSW
#   - Animated visualizations (GIFs)
#   - Additional distance metrics (Manhattan, Hamming)
#   - Systematic hyperparameter sweep (n_trees, candidate_axes, k)

import math
import random 
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt


#4/26/26
# I am adding this here this is at the top but its actually ironiclly probably the last thign im adding 
# this is just to make sure that when the code runs that it prints and logs it down  to a file. 
# this is pretty helpful so its really slow with the backtracking because it needs to backtrack so maany times that it effectively does almost brute force search 




def print_and_log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **{k: v for k, v in kwargs.items() if k != 'file'}, file=log_file, flush=True)

# open a log file to save results as they print
log_file = open("results_log.txt", "a")
print_and_log(f"\n{'='*85}\nRun started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*85}\n")


# This is a helper function for other things 
def euclidean_distance(point1, point2): 
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
# can also implement the manhattan distance however I dont think it would be signifigantly 
# different in this case? I did look into it and it is not exactly a cut and dry answer
# so the euclidian distance is a less stable algorithm but better for smaller datasets. 
# whereas the manhattan distance is better for higher dimensionality datasets like over 10-50ish
# so I think for the purposes of what we're trying to do it does make sense to implement a
# manhattan distance calculation 


######################################
# 1) Brute Force Approach 
######################################

# Psuedocode 

# function brute_force_knn(dataset, query, k):
#     distances = []
#     for each point in dataset:
#         d = euclidean_distance(query, point)
#         distances.append((d, point))
#     sort distances by d
#     return the first k entries


# have to consider how the data is stored and how to sort it efficiently as well as, how its retrieved
# and additionally there a sorting that needs to happen as well so need to tie all of those things into this

# okay so after doing a lot of research and what not I emailed the professor/TA

# I think that theres really 2-3 ways to implement this 

# 1. Utilizing insertion sort to just update the list every time we add a new point to the dataset, 
# this way we can keep the list sorted at all times and just pop the last element when we need to remove one. 
# This is a good option if we have a lot of queries and not a lot of updates to the dataset.

# 2. Just store them all into the list and the sort it after the fact, this is a good option if we have a lot of updates to the 
# dataset and not a lot of queries.

# This option 2 is kind of a open ended option as well right where we can either use whatever is the best sorting algorithm 
# for the dataset size or we can just use python's built in sort which is a timsort and is pretty efficient for most cases.


# they responded and said that I could just implement the python sorting method and I review the documentation for it
# as far as I understand it implements something like Timsort which is a hybrid sorting algorithm derived from merge sort 
# and insertion sort, designed to perform well on many kinds of real-world data. It is stable and adaptive, 
# meaning it can take advantage of existing order in the data. The biggest thing here is the stabiltiy whihc is what we really want 
# in this case. 

def model1_bf_knn(dataset, query,k):
    distances = []
    for point in dataset:
        d = euclidean_distance(query, point)
        distances.append((d,point))
    # Sort here best sorting function for entering a new value into the dataset? 
    distances.sort(key=lambda x: x[0])
    return [point for _, point in distances[:k]] # so need to test this but I think this 
    # returns the k nearest neighbors to the query point from the dataset using brute force





######################################
# 2) k-D tree Approach
######################################

# Psuedocode 

# Build Phase Psuedocode 

# function build(points):
#     if points is empty: return null
    
#     axis = dimension with largest spread (or cycle 0,1,2...)
#     median = median value of points along that axis
    
#     node.split_axis = axis
#     node.split_value = median
#     node.point = the median point itself
    
#     left_points = points where axis-value < median
#     right_points = points where axis-value >= median
    
#     node.left = build(left_points)
#     node.right = build(right_points)
#     return node

# this is a class that justs holds and stores the nodes themselves. 
class KDNode:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point        # the median point stored at this node
        self.axis = axis          # which dimension we split on
        self.left = left          # subtree with points < median on this axis
        self.right = right        # subtree with points >= median on this axis



def build_kdtree(points, depth=0):
    # Base case: if there are no points left, return None (empty subtree)
    if not points:
        return None

    # Figure out how many dimensions our data has 2d 3d etc. 
    ndim = len(points[0])

    # Here we are chosing the dimension to split on 
    # here we are splitting along the largest spread which ends up putting the most discriminating splits
    # near the root notde. 
    axis = max(range(ndim), key=lambda a: max(p[a] for p in points) - min(p[a] for p in points))

    # Sort along that axis and pick the median as the split point
    points_sorted = sorted(points, key=lambda p: p[axis])
    median_idx = len(points_sorted) // 2

    # this builds the node for this particular median point 
    # we then do recursive calls for building the left and right sides of the KDtree
    # essentially the left subtree gets all the points before the median
    # whereas the right subtree gets all of the pooints after the median (larger values)
    return KDNode(
        point=points_sorted[median_idx],
        axis=axis,
        left=build_kdtree(points_sorted[:median_idx], depth + 1),
        right=build_kdtree(points_sorted[median_idx + 1:], depth + 1)
    )





# Query Phase Psuedocode

# function query(node, query_point, k, best_k):
#     if node is null: return
    
#     d = euclidean_distance(query_point, node.point)
#     update best_k with (d, node.point) if better
    
#     # decide which side to visit first
#     diff = query_point[node.axis] - node.split_value
#     if diff < 0:
#         first, second = node.left, node.right
#     else:
#         first, second = node.right, node.left
    
#     # always search the "near" side
#     query(first, query_point, k, best_k)
    
#     # only search the "far" side if it could contain something closer
#     if |diff| < distance of kth-best neighbor so far:
#         query(second, query_point, k, best_k)



# this query now searches the tree 
def kd_query(node, query, k, best_k):
    # best_k is a list of (distance, point) tuples, max length k
    # this just represents the best tuple that we've found thus far 
    if node is None:
        return

    # Check current node distance 
    d = euclidean_distance(query, node.point)

    # Update best k if this node is one of the k NN found so far 
    if len(best_k) < k:
        best_k.append((d, node.point))
        best_k.sort(key=lambda x: x[0])          # keep sorted so best_k[-1] is worst
    elif d < best_k[-1][0]:
        best_k[-1] = (d, node.point)
        best_k.sort(key=lambda x: x[0])          # keep sorted so best_k[-1] is worst



    # Decide which side of the split the query is on
    # if diff is > 0 means we go to the right else we go left 
    # The "near" side is whichever side the query point is on more likely to have close neighbors
    # this is much much easier to show with a graph than just by talking about it 
    # The "far" side is the opposite we may be able to skip it entirely.
    diff = query[node.axis] - node.point[node.axis]
    near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

    # Always search the near side first
    kd_query(near, query, k, best_k)

    # only search far side if it could possibly contain a closer neighbor
    # the distance to the splitting plane is abs(diff)
    # if that distance is greater than our current worst neighbor then theres no point on the
    # far side can possibly be closer than what we already have, so we skip it entirely.
    # if its less then the far side might contain something closer and we have to check it
    # we also check len(best_k) < k because if we dont have k neighbors yet we must keep searching
    if len(best_k) < k or abs(diff) < best_k[-1][0]:
        kd_query(far, query, k, best_k)



# this is essentially just a little wrapper function that we can use
# this builds our tree with build KDtree(), then we can query for the nearest points to the tree
# this does rebuild the tree everytime which is really not ideal but I think for this project its not a big deal
def model2_kdtree_knn(dataset, query, k):
    tree = build_kdtree(dataset)
    best_k = []
    kd_query(tree, query, k, best_k)
    return [point for _, point in best_k]




















######################################
# 3) Randomized kd Tree Forest (ANN)
######################################

# Inspired by the randomized nearest neighbor approach described in:
# Lu and Gweon [4] introduces randomness to reduce computational cost
# while maintaining a high probability of finding relevant neighbors.

# Core idea: instead of always splitting on the single best axis,
# we randomly sample a subset of axes and pick the best among those.
# By building multiple trees this way (a forest), each tree partitions
# the space differently. At query time we search all trees and combine
# results, which gives us better coverage than any single tree alone.
# This is the "approximate" part - we might miss the true nearest neighbor
# in one tree but another tree in the forest is likely to find it.
# This is also notably similar to another method called ANNOY which
# I think was created by spotifiy and is something I also ran into while researching
# https://www.youtube.com/watch?v=DRbjpuqOsjk
# https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html
# I will probably add these as citations as well for the paper. even if I dont actually cite them in the bpaper


# Psuedocode

# Build Phase - one randomized tree
# function build_randomized_kdtree(points, depth, candidate_axes):
#     if points is empty: return null
    
#     # Instead of always picking the single best axis,
#     # randomly sample 'candidate_axes' axes and pick the best among those
#     k = number of dimensions
#     sampled_axes = random sample of 'candidate_axes' indices from range(k)
#     axis = axis with largest spread among sampled_axes only
    
#     median = median point along chosen axis
#     left = build_randomized_kdtree(points < median, depth+1, candidate_axes)
#     right = build_randomized_kdtree(points >= median, depth+1, candidate_axes)
#     return node(median, axis, left, right)


# Build Phase - the forest
# function build_forest(points, n_trees, candidate_axes):
#     forest = []
#     for i in range(n_trees):
#         tree = build_randomized_kdtree(points, depth=0, candidate_axes)
#         forest.append(tree)
#     return forest


# Query Phase
# function forest_query(forest, query, k):
#     best_k = []
#     for each tree in forest:
#         kd_query(tree, query, k, best_k)  # same query function you already have
#     keep only the k closest results in best_k
#     return best_k

# we can just use the same query function as before in this case what we're doing is building our tree differnetly
# additionaly we are creating a bit more like a forest of trees which gives us like an approximate space for 
# the nearest neighbors. 





# so we have an extension/rebuild from our kdtree from before so instead we want to build a "randomized"
# KD tree where the splits are random 

def build_randomized_kdtree(points, depth=0, candidate_axes=2):
    if not points:
        return None
    
    # Figure out how many dimensions our data has like 2d 3d etc. 
    ndim = len(points[0])

    
    # this is the part thats different from before:
    # previouslty we had axis = max(range(k), key=lambda a: max(p[a] for p in points) - min(p[a] for p in points))
    # but here we want to randomly sample a candidate axis then pick the largest spread from amound them

    #okay so what does that mean exactly? essentially instead of splitting on the best median axis we just 
    # split along the best random axis from a random sample, essentially when we do this we get a different 
    # KD tree than normal. hence why we do it multople times for a KD tree forest. 
    # sample along random axis then pick the best split on that axis
    sampled = random.sample(range(ndim), min(candidate_axes, ndim))
    axis = max(sampled, key=lambda a: max(p[a] for p in points) - min(p[a] for p in points))



    # rest of the build is exaclt the same as the other tree 
    points_sorted = sorted(points, key=lambda p: p[axis])
    median_idx = len(points_sorted) // 2

    # recursive call basically same as bfore 
    return KDNode(
        point=points_sorted[median_idx],
        axis=axis,
        left=build_randomized_kdtree(points_sorted[:median_idx], depth + 1, candidate_axes),
        right=build_randomized_kdtree(points_sorted[median_idx + 1:], depth + 1, candidate_axes)
    )




# okay so now onto the forest that I was talking about this is not a random forrest model though so 
# dont start getting your machine learning models confused. I mean I guess it kind of is but its certainly 
# different. 

def build_forest(points, n_trees=3, candidate_axes=1):
    # this is basicaly a wrapper for the previous funciton but essentially all we're doing here is making multiple
    # trees here and the ammount of trees and how many axes we are chosing between are effectively hyperparameters
    # but I dont think I'll have time to really get into testing those very extensive.y 

    # essentially the idea here though is that each individual tree might give us better candidate neighbors
    # so together they give us better approximate coverage of the dataset than a single tree could do. 
    forest = []
    for _ in range(n_trees):
        tree = build_randomized_kdtree(points, candidate_axes=candidate_axes)
        forest.append(tree)
    return forest



###### 4/25/26

# final updates to the code: 
# I had a thought for my implementation that essentially instead of using the other query we could instead 
# just do basically a DFS and return whatever for that tree. So essentially no backtracking at all. 
# but also maybe introduce backtracking as a hyperparameter?

#So the idea here is to kind of just shoot all the way down to the closest leaf node and then just return that 
# so based off the other queery funciton I think I need to just change a small part 

# change the bottom part

# this top part just kind of like scales it or whatever so the real differentce we need is to the end 
def kd_query_dfs(node, query, k, best_k):
    if node is None:
        return
    d = euclidean_distance(query, node.point)
    if len(best_k) < k:
        best_k.append((d, node.point))
        best_k.sort(key=lambda x: x[0])          # keep sorted so best_k[-1] is worst
    elif d < best_k[-1][0]:
        best_k[-1] = (d, node.point)
        best_k.sort(key=lambda x: x[0])          # keep sorted so best_k[-1] is worst


    diff = query[node.axis] - node.point[node.axis]

    # dont need to change this but far is just an unused variable right now. 
    # however I also want to maybe try this with short backtracking but im going to test this as is first

    # like essentially short backtracking would be another hyperparameter to just say he check
    # x (new hyperparameter) nodes in backtracking on the tree and if on those splits we find a better option 
    # then take the better option 
    near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)


    kd_query_dfs(near, query, k, best_k)


    # this is basiclly just checking the far side and the pruning statement but lets just not checking 
    # so essentially it just prunes whatever isnt right there
    # if len(best_k) < k or abs(diff) < best_k[-1][0]:
    #     kd_query_dfs(far, query, k, best_k)



# now also want to try and implement this with a backtrackking hyperparameter
# essentially the same thing but can just set backtracking to 0 if wanting the same thing. 
# currently its super super fast, I think testing for a range of backtracks would be an interesting hyper 
# parameter optimizaiton to check. 


def kd_query_dfs_backtrack(node, query, k, best_k, backtracking=2):
    if node is None:
        return
    d = euclidean_distance(query, node.point)
    if len(best_k) < k:
        best_k.append((d, node.point))
        best_k.sort(key=lambda x: x[0])          # keep sorted so best_k[-1] is worst
    elif d < best_k[-1][0]:
        best_k[-1] = (d, node.point)
        best_k.sort(key=lambda x: x[0])          # keep sorted so best_k[-1] is worst


    diff = query[node.axis] - node.point[node.axis]

    # dont need to change this but far is just an unused variable right now. 
    # however I also want to maybe try this with short backtracking but im going to test this as is first

    # like essentially short backtracking would be another hyperparameter to just say he check
    # x (new hyperparameter) nodes in backtracking on the tree and if on those splits we find a better option 
    # then take the better option 
    near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)


    kd_query_dfs_backtrack(near, query, k, best_k, backtracking)


    # this is basiclly just checking the far side and the pruning statement but lets just not checking 
    # so essentially it just prunes whatever isnt right there

    # only differnece here is this backtracking getting added 
    if backtracking > 0 and (len(best_k) < k or abs(diff) < best_k[-1][0]):
        kd_query_dfs_backtrack(far, query, k, best_k, backtracking - 1)












def model3_forest_knn(dataset, query, k, n_trees=3, candidate_axes=1, max_backtracks=2):

    # again this is basically just a wrapper function for the other functions but essentialy this works to 
    # create the forrest and then query it. In all realistic senses if you had a huge dataset of like millions 
    # of variables theres probably better ways of setting this up but for my purposes this will be fine
    #
    # this fucntion though will return the best candidate neighbors giving us an approximation of the nearest 
    # neighbors, this should effectively be faster than KNN but loses some accuracy. 

    # notable things here is that changing the hyperparamters will change how accurate/innacurate this will be 
    # more tree's higher accuracy and higher candidate axes will lead to the same thing. Essentially in KD tree
    # we have candidate axes =max candidate axes so ironically you could probably just run this with that parameter
    # and build the same tree hypotheticlaly

    forest = build_forest(dataset, n_trees=n_trees, candidate_axes=candidate_axes)
    candidates = []
    for tree in forest:
        tree_best = []
        kd_query_dfs_backtrack(tree, query, k, tree_best, max_backtracks) # changed over to new kd query 
        candidates.extend(tree_best)


    # merge all candidates and take the best k
    candidates.sort(key=lambda x: x[0])
    # deduplicate by point value before trimming
    seen = set()
    deduped = []
    for d, p in candidates:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            deduped.append((d, p))
    
    return [point for _, point in deduped[:k]]






#########################################
#########################################

#visaulization stuffs

#########################################
#########################################

# again will be having claude help me to make this stuff 
# note from down below:
# I already outlined exactly what tests i needed to make so I just had claude generate me a way of testing 
# these things but just to make sure im in line with the actual LLM usage guidelines Ive added a lot of commentary
# to showcase that I know what is going on with these functions 


# I actually addded this part after the testing one but It has to go before because ofwhen its called
# I had claude help me to make these visualizations because I dont like doing all that but essentially I jjust out
# outlined exactly what I wanted it to make so that I could see the visualizations 




def generate_visualizations(dimensions, runtime, memory_build, memory_query):
    # This really just keeps the colors and markers consistent accross the three diff figs
    # that are created with this func
    # COLORS = {
    #     "Brute Force": "#2E86AB",
    #     "k-d Tree":    "#A23B72",
    #     "Forest ANN":  "#F18F01",
    # }
    # MARKERS = {
    #     "Brute Force": "o",
    #     "k-d Tree":    "s",
    #     "Forest ANN":  "^",
    # }

    # a few hours later after running all my shennaningand on 4/25 I need to update this visualizations stuff as wel
    # really should have just made it so that it saves all these outputs to a json or something lol
    # im just so tired and also working on capstone at same time 
    def get_color(name):
        if name == "Brute Force":
            return "#2E86AB"
        elif name == "k-d Tree":
            return "#A23B72"
        else:
            return "#F18F01"

    def get_marker(name):
        if name == "Brute Force": return "o"
        elif name == "k-d Tree": return "s"
        bt_markers = {"bt=0":"^","bt=1":"v","bt=2":"D","bt=5":"P","bt=10":"X","bt=50":"*","bt=100":"h","bt=500":"^","bt=1000":"v"}
        for key, marker in bt_markers.items():
            if key in name: return marker
        return "^"



    # ── Figure 1: Runtime ──
    # pretty key figure for our resutls want to really see how lond each one takes so we can 
    # kind of see the curse of dimensionality in action. 
    fig, ax = plt.subplots(figsize=(9, 5))

    # one line per algo
    for name, vals in runtime.items():
        ax.plot(dimensions, vals,
                color=get_color(name),
                marker=get_marker(name),
                linewidth=2.2 if name in ["Brute Force", "k-d Tree"] else 1.2,
                markersize=7,
                alpha=1.0 if name in ["Brute Force", "k-d Tree"] else 0.7,
                label=name)
        
    # adding this for better scalling easier to see the graphs
    # theres a lot of different things here now but I had to add this because it kept cutting off the 
    # axis on the plot after adding the log scale 
    ax.set_xscale('log')
    ax.set_xlim(1.5, 250)
    ax.set_xticks(dimensions)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xticklabels(dimensions)
        
    # this part specifically highlights the crossover where kd tree stops being faster than brute force
    # # I specifically wanted ot show this 
    # ax.axvspan(8, 12, alpha=0.08, color="#333333")
    # ax.text(10, max(runtime["k-d Tree"]) * 0.4,
    #         "k-d tree\ncrosses BF\n~dim 10",
    #         fontsize=7.5, color="#555555", ha="center")

    # 4/25/26 okay so after doing some testing here again It seems like this just doesnt hold up well
    # with the dataset being larger 

    
    #rest is kind of self explanatory its just formating stuff
    ax.set_xlabel("Dimensionality", fontsize=11)
    ax.set_ylabel("Avg Query Runtime (s)", fontsize=11)
    ax.set_title("Query Runtime vs Dimensionality", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig("fig1_runtime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_runtime.png")

    # ── Figure 2: Memory ──
    # this figure really just shows the memory cost of each approach. these are very very similar acorrs each one 
    # the reason for that is theres a limitation in how I implemented it. I think the major issue here is that 
    # theres two differnet memory costs theres a memory cost for holding the data stuct and then 
    # also for spanning or querying the structure and so Idk it might be best to just redo this to make sure that s
    # shown here but I'd have to change a lot of other parts of the code I think to fix this?

    # currently this basically just shows a line 

    # Okay it was kind of a pain in the ass but I ran throuhg and fixed this so now instead we should be able to see
    # both the build and run times whihc I actually think is more important

    #essentially though now its a stacked bar chart so it shows side by side the build/query cost for each algo
    # so we can compre the two of them 
    # x = np.arange(len(dimensions))
    # width = 0.12
    # names = list(memory_build.keys())

    # needing to add this as well to fix the visualiations 
    memory_build_filtered = {n: v for n, v in memory_build.items() if n in ["Brute Force", "k-d Tree"] or n == "Forest ANN (bt=0)"}
    memory_query_filtered = {n: v for n, v in memory_query.items() if n in ["Brute Force", "k-d Tree"] or n == "Forest ANN (bt=0)"}
    x     = np.arange(len(dimensions))
    width = 0.12
    names = list(memory_build_filtered.keys())

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, name in enumerate(names):
        #guves slight offset 
        offset = (i - len(names)/2) * width
        # shows the build memory
        ax.bar(x + offset, memory_build_filtered[name], width,
               label=f"{name} (build)", color=get_color(name), alpha=0.9)
        # shows the query memory
        ax.bar(x + offset + width, memory_query_filtered[name], width,
               label=f"{name} (query)", color=get_color(name), alpha=0.9,
               hatch='///')

    # rest is self explanatory
    ax.set_xlabel("Dimensionality", fontsize=11)
    ax.set_ylabel("Memory Usage (KB)", fontsize=11)
    ax.set_title("Memory Usage: Build vs Query", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig2_memory.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 3: k-d Tree Partition Diagram ──

    # I feel like this one is generally kind of self explanatory if you know what a k-d tree looks like. 
    # I might add more figures later for the presentation to look at the KD tree as a gif and also a 
    # similar one for the ANN because it would be cool for the presentation. 

    #well see if I have time for all that though

    # essentially though this one just has a fixed seed and we build a kd tree and show where all the splits
    # are happenign 
    # np.random.seed(7)
    pts = [list(np.random.rand(2) * 10) for _ in range(18)] # just creates a bunch of poitns for us to use
    # for the plotting ere

    # this is not the KD tree we use necessarily but we use the function 
    # to showcase what this kind of looks like 
    tree = build_kdtree(pts)  # uses your existing build_kdtree function

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    # differnt colours being used per depth level, also could instead just do differnt colours per dimension?
    # however I think htis comes out looking very clean 
    split_colors = ["#2E86AB", "#A23B72", "#F18F01", "#44BBA4"]


    # kind of self explanatory here but this just runs throughy and makes all the splits on the KD tree so that we 
    # can visually see how this kind of works 
    def draw_splits(node, xmin, xmax, ymin, ymax, depth=0):
        if node is None:
            return
        color = split_colors[depth % len(split_colors)]
        lw = max(2.5 - depth * 0.5, 0.8) # lines get thinner at deeper levels 

        #since this is only in 2d for the visualization then we are shoing x/y here but could be many many 
        # different dimensions 
        if node.axis == 0:
            x = node.point[0]
            ax.plot([x, x], [ymin, ymax], color=color, linewidth=lw, alpha=0.85)
            #recursive calls for left/right subtree left or right of the split basicaly
            draw_splits(node.left,  xmin, x,   ymin, ymax, depth+1)
            draw_splits(node.right, x,    xmax, ymin, ymax, depth+1)
        else:
            #same thing here but horizontal 
            y = node.point[1]
            ax.plot([xmin, xmax], [y, y], color=color, linewidth=lw, alpha=0.85)
            draw_splits(node.left,  xmin, xmax, ymin, y,    depth+1)
            draw_splits(node.right, xmin, xmax, y,    ymax,  depth+1)

    draw_splits(tree, 0, 10, 0, 10)
    #just plots all the data points as a scatter 
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(xs, ys, color="#333333", s=55, zorder=5,
               edgecolors="#ffffff", linewidths=0.8)
    
    # this just highlights the root node its the first split so kind of important 
    ax.scatter([tree.point[0]], [tree.point[1]], color="#F18F01", s=120,
               zorder=6, edgecolors="#333333", linewidths=1.5)
    
    #legend for clear explanations 
    legend_elements = [
        plt.Line2D([0],[0], color=split_colors[0], lw=2.5, label="Depth 0 split"),
        plt.Line2D([0],[0], color=split_colors[1], lw=2.0, label="Depth 1 split"),
        plt.Line2D([0],[0], color=split_colors[2], lw=1.5, label="Depth 2 split"),
        plt.Line2D([0],[0], color=split_colors[3], lw=1.0, label="Depth 3 split"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.set_xlabel("X axis", fontsize=10)
    ax.set_ylabel("Y axis", fontsize=10)
    ax.set_title("k-d Tree Space Partitioning (2D)", fontsize=13,
                 fontweight="bold", pad=14)
    ax.grid(True, alpha=0.15, linestyle="--", color="#aaaaaa")
    plt.tight_layout()
    plt.savefig("fig3_kdtree_partition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3_kdtree_partition.png")





    # ── Figure 4: Randomized Forest - multiple trees on same dataset ──
    # this one just shows how the different trees work together so for outse we are using just 3 trees because
    # as I was saying previously 
    # I was trying hard to break ANN and make it not be super accurate I think the large issue here is probably 
    # going to be spread of items and the curse of dimensionality with so many items at higher dimensions
    # its basically borderline searching the whole tree hence why its so so accurate 
    # np.random.seed(99) this seemed to be causing issues
    # this just creates points for plotting 
    pts_ann = [list(np.random.rand(2) * 10) for _ in range(18)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Randomized k-d Tree Forest: 3 Different Space Partitionings",
                 fontsize=13, fontweight="bold")

    for tree_idx, ax in enumerate(axes):
        # build a different randomized tree each time
        # candidate_axes=1 means each split picks from 1 random axis
        # so different trees make very different partition decisions
        # with this method we should build multiple trees since candidate axes is 1 
        rand_tree = build_randomized_kdtree(pts_ann, candidate_axes=1)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(f"Tree {tree_idx + 1}", fontsize=11)

        # similar to before 
        split_colors = ["#2E86AB", "#A23B72", "#F18F01", "#44BBA4"]

        def draw_splits_ann(node, xmin, xmax, ymin, ymax, depth=0):
            # ver similar kind of recursive logic as vis 3 so not too muhc to explain here
            if node is None:
                return
            color = split_colors[depth % len(split_colors)]
            lw = max(2.5 - depth * 0.5, 0.8)
            if node.axis == 0:
                # x axis splittiung 
                x = node.point[0]
                ax.plot([x, x], [ymin, ymax], color=color, linewidth=lw, alpha=0.85)
                draw_splits_ann(node.left,  xmin, x,    ymin, ymax, depth+1)
                draw_splits_ann(node.right, x,    xmax,  ymin, ymax, depth+1)
            else:
                y = node.point[1]
                # y axis splitting 
                ax.plot([xmin, xmax], [y, y], color=color, linewidth=lw, alpha=0.85)
                draw_splits_ann(node.left,  xmin, xmax, ymin, y,    depth+1)
                draw_splits_ann(node.right, xmin, xmax, y,    ymax,  depth+1)

        draw_splits_ann(rand_tree, 0, 10, 0, 10)

        # same data points for each set so same data but different splits
        xs = [p[0] for p in pts_ann]
        ys = [p[1] for p in pts_ann]
        ax.scatter(xs, ys, color="#333333", s=55, zorder=5,
               edgecolors="#ffffff", linewidths=0.8)
        
        # highlight each tree's root split point 
        ax.scatter([rand_tree.point[0]], [rand_tree.point[1]],
                   color="#F18F01", s=120, zorder=6,
                   edgecolors="#333333", linewidths=1.5)
        ax.grid(True, alpha=0.15, linestyle="--", color="#aaaaaa")

    plt.tight_layout()
    plt.savefig("fig4_forest_partitions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig4_forest_partitions.png")










#########################################
#########################################

#TESTING stufffs 

#########################################
#########################################


# yes I know that the file structures suck but also its lowk awesome to have one big file 


# I already outlined exactly what tests i needed to make so I just had claude generate me a way of testing 
# these things but just to make sure im in line with the actual LLM usage guidelines Ive added a lot of commentary
# to showcase that I know what is going on with these functions 
# anyways below is what claude made for running some tests on:
# 1. Query runtime
# 2. Memory usage
# 3. Nearest neighbor accuracy 

# I also might just as well make some plots for visualizations and if I do that I'll have claude help me 
# because I hate syntax for plotting functions 



# function for looking at the accruacy of the ANN algorithm 
def compute_accuracy(true_neighbors, approx_neighbors):
    true_set = set(tuple(p) for p in true_neighbors)
    approx_set = set(tuple(p) for p in approx_neighbors)
    if not true_set:
        return 1.0
    return len(true_set & approx_set) / len(true_set)


# okay after a bunch of testing I realized that claude is stupid because the points I was testing on
# were normally distributed lol so adding this in now as well

# basically just makes a clustered dataset instead of what we have whihc Ill comment out 
def generate_clustered_dataset(n_points, dim, n_clusters=10):
    # spread cluster centers across a wider range
    centers = np.random.rand(n_clusters, dim) * 10  # centers in [0,10] not [0,1]
    
    dataset = []
    for _ in range(n_points):
        center = centers[np.random.randint(n_clusters)]
        # larger noise so clusters are more spread out and less tight
        point = center + np.random.randn(dim) * 1.5
        dataset.append(list(point))
    
    return dataset



# I think this test function is relatively self explanatory and especially wiht the comments that I added
# essentially here we are running through each set of dimensions and then running though each algoirthm for 
# those dimensions and testing them out
# I think what we will see is deterioration as we get to these higher and higher dimensionality spaces 
# this is due to the curse of dimensionality that we get with these kinds of problems. 

# so likely what we'll see is the KD tree is really good in 2d much better than KNN then its terrible because of 
# the curse of dimensionality and the ANN is mid at best at low dimensions and really quite good in comparison 
# at high dimensions 

def run_tests():
    dimensions = [2, 5, 10, 20, 50, 100, 200] #this goes up to 50 dimesnions in each point which I think is a reasonable test
    n_points = 100000 # number of points in the set each point having x dimensions
    n_queries = 50 # number of points to find k nearest neighbors for 
    k = 10 # how many nearest neighbors 

    print_and_log(f"{'Dim':<6} {'Algorithm':<20} {'Avg Runtime (s)':<20} {'Build Mem (KB)':<15} {'Query Mem (KB)':<15} {'Accuracy'}")
    print_and_log("-" * 85)


    # storage for results to pass to visualizations at the end
    # results = {
    #     "Brute Force": {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []},
    #     "k-d Tree":    {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []},
    #     "Random Forest ANN": {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []},
    # }


    # 4/25/26 new update I want to try and look at multiple different back tracking variables

    # so inorder to do that we are going to build the backtracking tree space once and then run backtracking 
    # a few different times with values from [0, 1, 2, 5, 10], so lets try this out it will be an interesting
    # addition to the paper I think 

    # another update I just wanted to add more backtracks in to test so im extenteding the set of backtracks to be 
    # in addition 50, 100, 500, 1000 going to try this out and see what kind of results we can see 
    results = {
        "Brute Force":       {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []},
        "k-d Tree":          {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []},
    }   
    for bt in [0, 1, 2]: # 5 and 10 was too many 5 is already really quite a lot of backtracking but slows it down quite a bit will write about this
        results[f"Forest ANN (bt={bt})"] = {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []}

    # runs through each dimension one by one 
    for dim in dimensions:
        # Generate random dataset and query points for this dimensionality
        # dataset = [list(np.random.rand(dim)) for _ in range(n_points)]  # this is so silly 

        #this new method generates clustered data so the data is more 
        #speard out and additionally we queery the same way so its no longer normaly distributed
        # this should make it so that the ANN will probably miss the truly neighboring points at high dimensions 
        # but we will see with testing 
        dataset = generate_clustered_dataset(n_points, dim)
        queries = generate_clustered_dataset(n_queries, dim, n_clusters=10)


        # Get brute force ground truth for accuracy comparison
        bf_results = [model1_bf_knn(dataset, q, k) for q in queries]

        # okay so the way i made things before came back to bite me in the ass 
        # so what we need to do is essenially just build he model and then just run the query
        # this issue I had of course was that it would just build the whole thing every time 
        # which I should have maybe seen lol but its fine. Anyways after some debugging:

        # build trees ONCE outside the query loop so we only time the query itself
        # measure build memory once per algorithm per dimension
        tracemalloc.start()
        kd_tree = build_kdtree(dataset)
        _, kd_build_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        kd_build_mem = kd_build_peak / 1024

        tracemalloc.start()
        # forest = build_forest(dataset, n_trees=5, candidate_axes=2) # 4/25/26 updating to 5 and 2 again 
        # rather than the original paper 3 and 1 this should help the backtracking a lil bit 


        # maybe 2 hours later than this other comment? 
        # future or maybe past me again? Idk im still doing testing and I wanted to try and really ramp up
        # the whole forest shenanigans. So I wanted to also test this with now 20 trees instead of just 5 
        # I think really making this into a whole forest would be dope
        forest = build_forest(dataset, n_trees=20, candidate_axes=2) # Tristan adding a new thiingymabob here 

        _, forest_build_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        forest_build_mem = forest_build_peak / 1024





        # algorithms now just do the query, not the build
        def run_bf(q):
            return model1_bf_knn(dataset, q, k)

        def run_kd(q):
            best_k = []
            kd_query(kd_tree, q, k, best_k)
            return [p for _, p in best_k]

        # def run_forest(q):
        #     candidates = []
        #     for tree in forest:
        #         tree_best = []
        #         kd_query_dfs_backtrack(tree, q, k, tree_best, 2) 
        #         candidates.extend(tree_best)
        #     candidates.sort(key=lambda x: x[0])
        #     seen = set()
        #     deduped = []
        #     for d, p in candidates:
        #         key = tuple(p)
        #         if key not in seen:
        #             seen.add(key)
        #             deduped.append((d, p))
        #     return [p for _, p in deduped[:k]]
        # # it seems like the algorithm is too efficient? its constantly returning an accuracy of 1 even at high dim 
        # # which is great but I'm struggling to break it


        # named_algos = [
        #     ("Brute Force",      run_bf),
        #     ("k-d Tree",         run_kd),
        #     ("Random Forest ANN",run_forest),
        # ]


        # 4/25/26 new testing regime update here to the spagehetti code making more spaghetti
        # sorry that tyou have to read  all of this stuff

        # this is the update in order to run through all of the different backtracking methods we need
        # to make a lil wrapper funciton for it and then run through all of them. Then we just print I think
        # but essentialy this should pause for each dimension run the backtracking for 100,000 datapoints and 
        # do several layers of backtracking which should give us higher accuracy presumably 

        named_algos = [
            ("Brute Force",      run_bf),
            ("k-d Tree",         run_kd),
        ]

        # test multiple backtracking budgets for the forest
        for bt in [0, 1, 2]: # 10 was too many
            def make_run_forest(backtracks):
                def run_forest(q):
                    candidates = []
                    for tree in forest:
                        tree_best = []
                        kd_query_dfs_backtrack(tree, q, k, tree_best, backtracks)
                        candidates.extend(tree_best)
                    candidates.sort(key=lambda x: x[0])
                    seen = set()
                    deduped = []
                    for d, p in candidates:
                        key = tuple(p)
                        if key not in seen:
                            seen.add(key)
                            deduped.append((d, p))
                    return [p for _, p in deduped[:k]]
                return run_forest
            named_algos.append((f"Forest ANN (bt={bt})", make_run_forest(bt)))








        # runs through each algo one by one 
        for name, algo in named_algos:
            runtimes = []
            memories = []
            accuracies = []

            for i, query in enumerate(queries):
                # Measure runtime
                start = time.perf_counter()
                result = algo(query)
                end = time.perf_counter()
                runtimes.append(end - start)

                # Measure memory
                tracemalloc.start()
                _ = algo(query)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                query_mem = peak / 1024  # convert bytes to KB
                memories.append(query_mem) 


                # Measure accuracy vs brute force ground truth
                # Brute force is always 100% accurate by definition
                accuracies.append(compute_accuracy(bf_results[i], result))
                

            avg_runtime = np.mean(runtimes)
            avg_memory   = np.mean(memories)
            avg_accuracy = np.mean(accuracies)

            # store results for plotting
            results[name]["runtime"].append(avg_runtime)
            results[name]["accuracy"].append(avg_accuracy)
            results[name]["memory_query"].append(query_mem)

            # add build memory for tree-based methods
            # store build memory separately per algorithm this is an appplied fix 
            # store build memory once per dimension per algorithm, not per query
            if name == "k-d Tree":
                results[name]["memory_build"].append(kd_build_mem)
            elif "Forest ANN" in name:  # catches all bt= variants this is an update
                results[name]["memory_build"].append(forest_build_mem)
            else:
                results[name]["memory_build"].append(0)
                        
            # basically want to store query and build as seperate things for plotting 
            
            
            # updated this to catach all th new backtracking thingsy

            print_and_log(f"{dim:<6} {name:<20} {avg_runtime:<20.6f} {results[name]['memory_build'][-1]:<15.2f} {avg_memory:<15.2f} {avg_accuracy:.4f}")
        
        print_and_log()  # blank line between dimensionalities

    ## I added this to generate a few visualizations as well for my own sake and for the wrtie up 
    runtime_viz = {name: results[name]["runtime"] for name in results}
    memory_build_viz = {name: results[name]["memory_build"] for name in results}
    memory_query_viz = {name: results[name]["memory_query"] for name in results}

    # rename Forest ANN key to match what the function expects
    # for d in [runtime_viz, memory_build_viz, memory_query_viz]:
    #     d["Forest ANN"] = d.pop("Random Forest ANN")

    generate_visualizations(dimensions, runtime_viz, memory_build_viz, memory_query_viz)
    log_file.close()












if __name__ == "__main__":
    run_tests()





