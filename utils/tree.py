"""Tree traversal util functions."""

import tskit
from collections import deque
import numpy as np

def build_ts_from_parents(parents, samples):
    # parents (N_samples), samples (N_samples)

    # Step 1: Initialize a TableCollection
    tables = tskit.TableCollection(sequence_length=1.0) # Assuming sequence length is 1.0

    # Step 2: Create a node map and set time as 0 for terminal nodes
    node_map = {node_id: None for node_id in np.unique(np.concatenate([parents, samples]))}
    node_time = {node_id: 0 for node_id in samples}  # Initialize time as 0 for samples

    for sample in samples:
        node_map[sample] = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)

    for child, parent in enumerate(parents[:-1]):
        # Add a new node for each child and set its time
        # max child time + 1: parent time
        t = node_time[child] + 1
        if node_map[parent] != None:
            existing_t = node_time[parent]
            if existing_t > t:
                t = existing_t

        node_map[parent] = tables.nodes.add_row(time=t)
        node_time[parent] = t

        # Add an edge from the parent to the child
        tables.edges.add_row(left=0, right=1, parent=node_map[parent], child=node_map[child])

    # Step 4: Sort and simplify the tables
    tables.sort()
    #tables.simplify(samples=[node_map[sample] for sample in samples])

    # Step 5: Generate the tree sequence
    ts = tables.tree_sequence()
    return ts

def build_ts_from_matrix(M, samples):
    # Step 1: Initialize a TableCollection
    tables = tskit.TableCollection(sequence_length=1.0) # Assuming sequence length is 1.0

    # assuming M is your numpy array
    # merging children columns
    children = np.concatenate((M[:, 1], M[:, 2]))

    # finding root node mask
    root = np.isin(M[:, 0], children, invert=True)

    t = M.shape[0] - len(samples)

    node_map = {node_id: None for node_id in np.unique(M)}

    # Step 2: Create a node map and set time as 0 for terminal nodes
    for sample in samples:
        node_map[sample] = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)

    # Step 3: Starting at the root BFS down adding times so that parents are always older than children

    # initialize a queue for BFS
    queue = deque()

    # starting node (or nodes) is/are the root(s), i.e., nodes that are not children of any other nodes

    node_map[M[root, 0][0]] = tables.nodes.add_row(time=t)
    queue.append((M[root, 0][0], t))

    # BFS traversal
    while queue:
        # Get the parent node and its time
        parent, parent_time = queue.popleft()

        # Decrease the time for each new generation
        t -= 1

        # Get the children of the parent node
        children = M[M[:, 0] == parent, 1:]

        # Go through each child
        for child in children[0]:

            # Add a new node for each child and set its time
            node_map[int(child)] = tables.nodes.add_row(time=t)

            # Add an edge from the parent to the child
            tables.edges.add_row(left=0, right=1, parent=node_map[parent], child=node_map[int(child)])

            # Enqueue the child node for further processing
            if child not in samples:
              queue.append((child, t))


    # Step 4: Sort and simplify the tables
    tables.sort()
    tables.simplify(samples=[node_map[sample] for sample in samples])

    # Step 5: Generate the tree sequence
    ts = tables.tree_sequence()
    return ts

def descendants_traversal(tree):
    """Get all descendants non-recursively, in traversal order."""
    n = len(list(tree.nodes()))
    root = n - 1

    traversal = []

    children = [list(tree.neighbors(node)) for node in range(n)]  # children remaining to process
    is_leaf = [len(children[node]) == 0 for node in range(n)]
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            assert node == stack.pop()
            if is_leaf[node]:
                traversal.append(node)

    return traversal[::-1]


def descendants_count(tree):
    """For every node, count its number of descendant leaves, and the number of leaves before it."""
    n = len(list(tree.nodes()))
    root = n - 1

    left = [0] * n
    desc = [0] * n
    leaf_idx = 0

    children = [list(tree.neighbors(node))[::-1] for node in range(n)]  # children remaining to process
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) == 0:
                desc[node] = 1
                left[node] = leaf_idx
                leaf_idx += 1
            else:
                desc[node] = sum([desc[c] for c in children_])
                left[node] = left[children_[0]]
            assert node == stack.pop()

    return desc, left
