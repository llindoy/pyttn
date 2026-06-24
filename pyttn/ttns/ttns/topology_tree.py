# This files is part of the pyttn package.
# (C) Copyright 2026 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import List, Dict, Optional

from .ntreeExt import ntree

class TopoTree: 
    """A lightweight wrapper around an ntree object that associates leaf nodes with composite site labels.

    This class provides a mapping between the leaves of a tree tensor network topology
    and user-defined site labels. The ordering of the labels determines the ordering
    of modes when constructing system_modes and relabelling operator objects.
    """

    def __init__(self, tree : ntree, leaf_labels : List[str]):
        """Initialise a labelled topology tree.

        This associates each leaf of the provided ntree structure with a corresponding
        site label. The ordering of the labels must match the ordering returned by
        the tree's leaf traversal.

        :param tree: The ntree structure defining the TTN topology
        :type tree: ntree
        :param leaf_labels: A list of labels corresponding to each leaf node in the tree
        :type leaf_labels: list[str]
        :raises ValueError: If the number of labels does not match the number of leaves in the tree
        """

        if tree.nleaves() != len(leaf_labels):
            raise ValueError("Number of labels must match number of leaves")
        
        self.tree = tree 
        self.leaf_labels = leaf_labels

    def leaf_order(self) -> List[str]:
        """Return the ordering of site labels defined by the tree structure.

        This ordering corresponds to the order of leaves in the tree and is used to
        define the ordering of modes in system_modes and operator relabelling.

        :return: The ordered list of site labels
        :rtype: list[str]
        """

        return self.leaf_labels
    
    def site_map(self) -> Dict[str, int]:
        """Construct a mapping from site labels to integer indices.

        The indices correspond to the position of each site label in the leaf ordering
        and define how operators should be relabelled to match the tree-induced ordering.

        :return: A mapping from site labels to integer mode indices
        :rtype: dict[str, int]
        """
        return {label: i for i, label in enumerate(self.leaf_labels)}

    def insert_subtree(self, 
                       node_path : List[int], 
                       subtree : Optional[ntree] = None, 
                       subtree_labels : Optional[List[str]] = None, 
                       TopoTree : Optional["TopoTree"] = None) -> None:
        """Insert a subtree into the topology and update leaf labels.

        This function allows insertion of either:
          - a raw subtree (`ntree`) with corresponding `subtree_labels`, or
          - an existing `TopoTree` object

        After insertion, the internal `leaf_labels` are updated such that
        their ordering remains consistent with the leaf traversal of the
        updated tree.

        :param node_path: Path to the node where the subtree is inserted
        :type node_path: list[int]
        :param subtree: The subtree to insert
        :type subtree: ntree, optional
        :param subtree_labels: Labels corresponding to the subtree leaves
        :type subtree_labels: list[str], optional
        :param TopoTree: A TopoTree object providing both tree and labels
        :type TopoTree: TopoTree, optional
        :raises ValueError: If inputs are inconsistent or insufficient
        """

        #validate inputs:
        if TopoTree is not None:
            if subtree is not None or subtree_labels is not None:
                raise ValueError("Provide either (subtree, subtree_labels) OR TopoTree, not both")
            subtree = TopoTree.tree
            subtree_labels = TopoTree.leaf_labels

        if subtree is None or subtree_labels is None:
            raise ValueError("Must provide either (subtree and subtree_labels) or TopoTree")

        if subtree.nleaves() != len(subtree_labels):
            raise ValueError("Number of subtree labels must match number of subtree leaves")

        #get the positions of all of the leaves
        old_leaf_paths = self.tree.leaf_indices()

        #get the node we are attaching the subtree to
        node = None
        try:
            node = self.tree.at(node_path)
        except Exception:
            raise RuntimeError(f"Failed to insert subtree.  Node path {node_path} is not valid for tree {self.tree}.") from None

        #now check if it is a leaf as in this case we will need to preserve the leaf structure
        is_leaf_target = node.is_leaf()

        #determine where we need to insert the new children in the site array
        insert_index = None
        for i, path in enumerate(old_leaf_paths):
            if(len(path) == len(node_path)):
                if(path == node_path):
                    insert_index=i
                    break

        #if we get to the end of the leaves without finding the current path then 
        if insert_index is None:
            insert_index = len(self.leaf_labels)

        #now handle the leaf case
        if is_leaf_target:
            original_value = node.value
            node.insert(original_value)
            node.insert(subtree.root())

            old_label = self.leaf_labels[insert_index]
            self.leaf_labels[insert_index:insert_index+1] = [old_label] + subtree_labels
        else:
            node.insert(subtree.root())
            self.leaf_labels[insert_index:insert_index] = subtree_labels