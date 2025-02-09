import numpy as np
from collections import deque  # 确保导入 deque

def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        identifier = None
        seq = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if identifier:
                    sequences[identifier] = ''.join(seq)
                identifier = line[1:]  # Remove '>'
                seq = []
            else:
                seq.append(line)
        if identifier:
            sequences[identifier] = ''.join(seq)
    return sequences

def remove_gaps(sequences):
    sample_seq = next(iter(sequences.values()))
    length = len(sample_seq)
    valid_positions = [i for i in range(length) if all(seq[i] != '-' and seq[i] != '?' for seq in sequences.values())]
    new_sequences = {key: ''.join(seq[i] for i in valid_positions) for key, seq in sequences.items()}
    return new_sequences, len(valid_positions)

def convert_u_to_t(sequences):
    new_sequences = {key: seq.replace('U', 'T') for key, seq in sequences.items()}
    return new_sequences

def calculate_hamming_distance(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)

def create_hamming_distance_matrix(sequences):
    seq_names = list(sequences.keys())
    num_seqs = len(seq_names)
    distance_matrix = np.zeros((num_seqs, num_seqs))

    sequences = convert_u_to_t(sequences)
    sequences, valid_length = remove_gaps(sequences)

    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            distance = calculate_hamming_distance(sequences[seq_names[i]], sequences[seq_names[j]])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix, seq_names

def compute_ui(distance_matrix):
    n = len(distance_matrix)
    u = np.zeros(n)

    for i in range(n):
        sum_distances = sum(distance_matrix[i][j] for j in range(n) if i != j)
        u[i] = sum_distances / (n - 2) if (n - 2) > 0 else 0

    return u

def compute_diff(distance_matrix, u_values):
    n = len(distance_matrix)
    diff_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff_matrix[i][j] = distance_matrix[i][j] - u_values[i] - u_values[j]

    return diff_matrix

def find_min_diff_pair(diff_matrix):
    n = len(diff_matrix)
    min_value = float('inf')
    min_index = (-1, -1)

    for i in range(n):
        for j in range(i + 1, n):
            if diff_matrix[i][j] < min_value:
                min_value = diff_matrix[i][j]
                min_index = (i, j)

    return min_index

def update_distance_matrix(distance_matrix, i, j):
    n = len(distance_matrix)
    new_distances = []

    for k in range(n):
        if k != i and k != j:
            distance_to_k = (distance_matrix[i][k] + distance_matrix[j][k] - distance_matrix[i][j]) / 2
            new_distances.append(distance_to_k)

    new_matrix = []
    for k in range(n):
        if k != i and k != j:
            new_row = [distance_matrix[k][m] for m in range(n) if m != i and m != j]
            new_row.append(new_distances.pop(0))
            new_matrix.append(new_row)

    new_matrix.append([row[-1] for row in new_matrix] + [0])

    return np.array(new_matrix)

def phylogenetic_reduction(distance_matrix, labels):
    while len(distance_matrix) > 2:
        u_values = compute_ui(distance_matrix)
        diff_matrix = compute_diff(distance_matrix, u_values)
        min_pair = find_min_diff_pair(diff_matrix)
        i, j = min_pair

        distance_matrix = update_distance_matrix(distance_matrix, i, j)
        new_label = f"({labels[i]},{labels[j]})"
        labels = [labels[k] for k in range(len(labels)) if k != i and k != j] + [new_label]

    final_i, final_j = 0, 1
    final_label = f"({labels[final_i]},{labels[final_j]})"

    return final_label

class TreeNode:
    _id_counter = 0

    def __init__(self, name=''):
        self.name = name
        self.children = []
        self.id = TreeNode._id_counter
        TreeNode._id_counter += 1

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def find_all_nodes(self):
        """Recursively find all nodes in the tree."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.find_all_nodes())
        return nodes

    def get_nonterminals(self, order="postorder"):
        """Get all non-terminal (internal) nodes in the tree.

        :param order: Traversal order, "postorder" (default) or "level"
        """
        if order == "postorder":  # 默认：后序遍历
            nodes = []
            if not self.is_leaf():
                nodes.append(self)
            for child in self.children:
                nodes.extend(child.get_nonterminals(order="postorder"))
            return nodes

        elif order == "level":  # 新增：层次遍历（Level-order）
            nodes = []
            queue = deque([self])  # 使用 `deque` 进行广度优先搜索

            while queue:
                node = queue.popleft()
                if not node.is_leaf():
                    nodes.append(node)  # 只存储非叶节点
                queue.extend(node.children)  # 继续加入子节点

            return nodes

        else:
            raise ValueError("Invalid order type. Use 'postorder' or 'level'.")


def parse_newick(newick):
    def _parse_subtree(subtree):
        if '(' not in subtree:
            return TreeNode(subtree.strip())

        subtree = subtree.strip()[1:-1]
        level = 0
        split_index = None
        for i, char in enumerate(subtree):
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            elif char == ',' and level == 0:
                split_index = i
                break

        left = subtree[:split_index]
        right = subtree[split_index + 1:]
        node = TreeNode('')
        node.add_child(_parse_subtree(left))
        node.add_child(_parse_subtree(right))
        return node

    return _parse_subtree(newick)


def fitch_parsimony(tree, sequences):
    """Calculate the parsimony score using Fitch's algorithm for any binary tree structure."""

    # 获取所有叶节点
    def get_leaves(node):
        """递归获取所有叶子节点"""
        if not hasattr(node, "children") or not node.children:  # 没有 children 说明是叶子节点
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(get_leaves(child))
        return leaves

    leaf_nodes = get_leaves(tree)  # 获取所有叶子节点
    seq_length = len(next(iter(sequences.values())))
    total_score = 0

    for site in range(seq_length):
        site_score = 0
        site_states = {leaf: {sequences[leaf.name][site]} for leaf in leaf_nodes}

        # 后序遍历 (手动递归实现)
        def postorder_traverse(node):
            """递归进行后序遍历"""
            if not hasattr(node, "children") or not node.children:  # 叶节点
                return

            # 确保二叉树
            if len(node.children) != 2:
                raise ValueError("Tree must be strictly binary")

            left_child, right_child = node.children
            postorder_traverse(left_child)
            postorder_traverse(right_child)

            left_state = site_states[left_child]
            right_state = site_states[right_child]

            # 计算交集和并集
            intersection = left_state & right_state
            if intersection:
                site_states[node] = intersection
            else:
                site_states[node] = left_state | right_state
                nonlocal site_score
                site_score += 1  # 发生了变异

        postorder_traverse(tree)  # 从根节点开始后序遍历
        total_score += site_score

    return total_score


import copy


def get_nni_neighbors(tree):
    """Get all neighbor trees of the given tree using the NNI algorithm."""

    neighbors = []  # 用于存储邻居树

    # 递归构建父节点字典
    def build_parent_dict(node, parent=None):
        parent_dict = {node: parent}
        for child in node.children:
            parent_dict.update(build_parent_dict(child, node))
        return parent_dict

    parents = build_parent_dict(tree)  # 构建父节点字典

    # 使用层次遍历获取所有非叶子节点
    queue = deque([tree])
    nonterminals = []

    while queue:
        node = queue.popleft()
        if not node.is_leaf():
            nonterminals.append(node)
        queue.extend(node.children)

    for node in nonterminals:
        left, right = node.children  # 获取当前节点的左右孩子

        # **子树内部交换 (仅当左右孩子有子树时才进行)**
        if not left.is_leaf() and not right.is_leaf():
            for left_child, right_child in zip(left.children, right.children):
                # 交换左右孩子的子节点
                new_tree = copy.deepcopy(tree)  # 复制整个树

                new_node = find_node_by_id(new_tree, node.id)
                assert new_node, f"Error: Node {node.id} not found in new tree"

                new_left, new_right = new_node.children
                new_left_child = find_node_by_id(new_tree, left_child.id)
                new_right_child = find_node_by_id(new_tree, right_child.id)

                assert new_left_child and new_right_child, "Error: Swapped children not found in new tree"

                new_left.children.remove(new_left_child)
                new_right.children.remove(new_right_child)

                new_left.children.append(new_right_child)
                new_right.children.append(new_left_child)

                neighbors.append(new_tree)

        # **父节点-当前节点交换**
        parent = parents.get(node)
        if parent is not None:
            sibling = parent.children[0] if parent.children[1] == node else parent.children[1]
            for child in node.children:
                new_tree = copy.deepcopy(tree)

                new_parent = find_node_by_id(new_tree, parent.id)
                new_node = find_node_by_id(new_tree, node.id)
                new_sibling = find_node_by_id(new_tree, sibling.id)
                new_child = find_node_by_id(new_tree, child.id)

                assert new_parent and new_node and new_sibling and new_child, "Error: Parent, sibling, or child node not found in new tree"

                new_parent.children.remove(new_sibling)
                new_node.children.remove(new_child)

                new_parent.children.append(new_child)
                new_node.children.append(new_sibling)

                neighbors.append(new_tree)

    return neighbors  # 返回所有邻居树


# **修正的 `find_node_by_id`**
def find_node_by_id(tree, target_id):
    """在树中根据 ID 查找节点"""
    for node in tree.find_all_nodes():
        if node.id == target_id:
            return node
    return None  # 没找到返回 None


def find_max_parsimony_tree_with_nni(sequences, initial_tree):
    best_tree = initial_tree
    best_score = fitch_parsimony(best_tree, sequences)

    while True:
        improved = False
        neighbors = get_nni_neighbors(best_tree)

        best_candidate = None
        best_candidate_score = best_score

        for neighbor in neighbors:
            score = fitch_parsimony(neighbor, sequences)
            if score < best_candidate_score:
                best_candidate = neighbor
                best_candidate_score = score

        if best_candidate:
            best_tree = best_candidate
            best_score = best_candidate_score
            improved = True

        if not improved:
            break

    return best_tree, best_score

def tree_to_newick(node):
    if node.is_leaf():
        return node.name if node.name else "Unknown"

    return f"({','.join(tree_to_newick(child) for child in node.children)})"


# 假设 `read_fasta`、`create_hamming_distance_matrix` 和 `phylogenetic_reduction` 已经定义
fasta_file_path = "/mnt/data/*.fasta"
sequences = read_fasta(fasta_file_path)
distance_matrix, labels = create_hamming_distance_matrix(sequences)

nj_tree_newick = phylogenetic_reduction(distance_matrix, labels)
initial_tree = parse_newick(nj_tree_newick)
max_parsimony_tree, best_score = find_max_parsimony_tree_with_nni(sequences, initial_tree)

print("NJ Tree (Newick format without branch lengths):")
print(nj_tree_newick)
print("\nParsimony Tree (Newick format without branch lengths):")
print(tree_to_newick(max_parsimony_tree))
print(f"\nThe length of the best tree (parsimony score) is: {best_score}")
