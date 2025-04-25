# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np
import networkx as nx 

from pyttn.ttnpp import ntree, ntreeNode, ntreeBuilder

from scipy.cluster.hierarchy import linkage

def gen_graph(M):
   return nx.from_numpy_array(np.abs(M - np.diag(np.diag(M))))

def split_high_degree_nodes(spanning_tree, N, root_index, max_degree=None):
   if max_degree is None:
      #if max degree has not been specified we just return the current tree
      return spanning_tree
   else:
      #in this case iterate through the tree determine if any nodes are too large and if they are we partition 
      #the node into sets of the correct size.  To do this we get a DFS edges list and if there are any instances
      #of nodes with more than max_degree children we insert sufficiently many logical nodes so that we have the
      #correct degree of connectivity
      edges = sorted([edge for edge in nx.dfs_edges(spanning_tree, source=root_index)], key = lambda x:np.abs(x[0]-root_index))
      
      nchildren = {}
      nodes_to_split = {}

      curr_node = None
      sind = 0

      #iterate over the edges getting the number of children associated with each node and the location of the
      #first edge associated with a nodes children in the list
      for i, e in enumerate(edges):
         if curr_node != e[0]:
            curr_node = e[0]
            sind = i

         if e[0] not in nchildren.keys():
            nchildren[e[0]]=1
         else:
            nchildren[e[0]]+=1

         if nchildren[e[0]] > max_degree:
            if e[0] not in nodes_to_split.keys():
               nodes_to_split[e[0]]=sind
               
      #if none of the nodes are high degree we don't need to do anything
      if len(nodes_to_split) == 0:
         return spanning_tree
      
      #otherwise we iterate through the tree and split high degree nodes off
      else:
         #now we split any nodes that need to be split
         counter = N
         T = nx.Graph()
         #if we are at a node we need to split
         for node, ind in nodes_to_split.items():
            nchild = nchildren[node]

            #insert floor(nchild/max_degree) logical nodes that will be full connecting 
            for j in range(nchild//max_degree):
               T.add_edge(node, counter)
               for k in range(max_degree):
                  T.add_edge(counter, edges[ind+j*max_degree+k][1])

               counter = counter+1

            if nchild%max_degree == 1:
               for j in range( (nchild//max_degree)*max_degree, nchild):
                  T.add_edge(node,  edges[ind+j][1])

            elif nchild%max_degree > 1:
               T.add_edge(node, counter)

               for k in range((nchild//max_degree)*max_degree, nchild):
                  T.add_edge(counter, edges[ind+j][1])

               counter = counter+1

         for e in edges:
            if e[0] not in nodes_to_split.keys():
               T.add_edge(e[0], e[1])
         return T

def insert_physical_nodes(spanning_tree, N, root_ind):
   nindex = spanning_tree.number_of_nodes()

   mapping = {}
   for i in range(nindex):
      mapping[i] = N+i

   #iterate over the tree and add nodes to any n
   nx.relabel_nodes(spanning_tree, mapping=mapping, copy=False)

   for i in range(N):
      spanning_tree.add_edge(N+i, i)

   return spanning_tree, N+root_ind

def generate_spanning_tree(M, max_degree=None, root_index=0):
   if root_index > M.shape[0] or root_index < 0:
      raise RuntimeError("Failed to generate spanning tree from weight matrix.  User specified root index out of bounds.")
   G = gen_graph(M)

   spanning_tree = nx.maximum_spanning_tree(G)
   spanning_tree = split_high_degree_nodes(spanning_tree, M.shape[0], root_index, max_degree=max_degree)

   return insert_physical_nodes(spanning_tree, M.shape[0], root_index)

def condense_distance_matrix(M):
   N = M.shape[0]

   dist = np.zeros((N*(N-1))//2)
   c = 0
   for i in range(N):
      for j in range(i+1, N):
         dist[c] = M[i, j]
         c += 1
   return dist

def linkage_to_nxtree(Z):
   N = Z.shape[0]+1

   edges = []
   root_index = 0
   #now iterate over links and construct an edge list
   for i in range(Z.shape[0]):
      Z0 = int(Z[i, 0])
      Z1 = int(Z[i, 1])
      edges.append((i+N, Z0))
      edges.append((i+N, Z1))
      root_index = i+N

   T = nx.Graph()
   for e in edges:
      T.add_edge(e[1], e[0])
      T.add_edge(e[0], e[1])

   return T, root_index

def generate_hierarchical_clustering_tree(M):
   dist = condense_distance_matrix(M)
   Z = linkage(dist, method='ward')
   return linkage_to_nxtree(Z,)


def generate_tree_from_matrix(M, method, *args, **kwargs):
   """A function for generating a networkx graph representing a generic tree structure from a
   a matrix describing correlations between nodes.  This function provides several different
   implementations dependent on the choice of 
   :param M: _description_
   :type M: _type_
   :param method: _description_
   :type method: _type_
   """
   return
