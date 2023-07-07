"""Tree traversal util functions."""

import tskit
from collections import deque
import numpy as np


# Naive O(N^2) version, original algo 1 from 2020 paper
def build_ts_from_embeddings(leaves_embeddings, samples):

    # Assuming `leaves_embeddings` is a 2D tensor of shape [num_leaves, embedding_dim]

    # Get the indices of the upper triangular part excluding the diagonal
    triu_indices = torch.triu_indices(leaves_embeddings.size(0), leaves_embeddings.size(0), offset=1)

    # Extract the embeddings corresponding to these indices
    embeddings1 = leaves_embeddings[triu_indices[0]]
    embeddings2 = leaves_embeddings[triu_indices[1]]

    # Compute dot products
    dot_products = (embeddings1 * embeddings2).sum(dim=1)

    # Use argsort to get the indices in ascending order of dot products
    tri_indices = torch.argsort(dot_products)
    sorted_pairs = triu_indices.t()[tri_indices]

    # Step 1: Initialize a TableCollection
    tables = tskit.TableCollection(sequence_length=1.0) # Assuming sequence length is 1.0

    # Step 2: Create a node map and set time as 0 for terminal nodes
    node_map = {}
    for sample in samples:
        node_map[sample] = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)

    # tree_id -> root node of tree
    tree_roots = {sample: sample for sample in samples}

    # node_id -> tree_id
    node_to_tree = {sample: sample for sample in samples}

    node_counter = max(samples) + 1

    for pair in sorted_pairs:
        # get tree_id of each node
        tree_id1 = node_to_tree[int(pair[0])]
        tree_id2 = node_to_tree[int(pair[1])]

        if tree_id1 != tree_id2:
            root1 = tree_roots[tree_id1]
            root2 = tree_roots[tree_id2]

            # Add a new root node joining root1 and root2:
            # parent time: max child time + 1
            t = max(tables.nodes[root1].time, tables.nodes[root2].time) + 1

            node_map[node_counter] = tables.nodes.add_row(time=t)

            # Add an edge from the parent to the child
            tables.edges.add_row(left=0, right=1, parent=node_map[node_counter], child=node_map[root1])
            tables.edges.add_row(left=0, right=1, parent=node_map[node_counter], child=node_map[root2])

            node_to_tree[node_map[node_counter]] = tree_id1
            tree_roots[tree_id1] = node_map[node_counter]

            for key, value in node_to_tree.items():
                if value == tree_id2:
                    node_to_tree[key] = tree_id1

            node_counter += 1

    # Step 4: Sort and simplify the tables
    tables.sort()
    #tables.simplify(samples=[node_map[sample] for sample in samples])

    # Step 5: Generate the tree sequence
    return tables.tree_sequence()

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

def map_mutations(inferred_ts, ts_data):
    tree = inferred_ts.first()  # there's only one tree anyway
    tables = inferred_ts.dump_tables()

    for variant, site_pos in zip(ts_data["genotypes"].T, ts_data["sites"]):
        ancestral_state, mutations = tree.map_mutations(variant, ("0", "1"))
        site_id = tables.sites.add_row(position=site_pos, ancestral_state="0")
        parent_offset = len(tables.mutations)
        for mut in mutations:
            parent = mut.parent
            if parent != tskit.NULL:
                parent += parent_offset
            mut_id = tables.mutations.add_row(
                site_id, node=mut.node, parent=parent, derived_state=mut.derived_state)
    return tables.tree_sequence()
