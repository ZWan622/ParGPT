import numpy as np
import copy


def read_sequences(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        sequence_name = ""
        sequence_data = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_name:
                    sequences[sequence_name] = sequence_data
                sequence_name = line[1:]  # Remove the '>' character
                sequence_data = ""
            else:
                sequence_data += line
        if sequence_name:
            sequences[sequence_name] = sequence_data
    return sequences


def calculate_hamming_distance(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)


def create_hamming_distance_matrix(sequences):
    seq_names = list(sequences.keys())
    num_seqs = len(seq_names)
    distance_matrix = np.zeros((num_seqs, num_seqs))

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
    return new_matrix


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
    def process_site(node, site):
        if node.is_leaf():
            return {sequences[node.name][site]}, 0

        left_child, right_child = node.children[:2]
        left_state, left_score = process_site(left_child, site)
        right_state, right_score = process_site(right_child, site)

        intersection = left_state & right_state
        if intersection:
            return intersection, left_score + right_score
        else:
            return left_state | right_state, left_score + right_score + 1

    seq_length = len(next(iter(sequences.values())))
    total_score = 0
    for site in range(seq_length):
        _, site_score = process_site(tree, site)
        total_score += site_score
    return total_score


def get_nni_neighbors(tree):
    neighbors = []
    for node in tree.children:
        if not node.is_leaf() and len(node.children) == 2:
            left, right = node.children
            neighbor = copy.deepcopy(tree)
            node_to_modify = find_node_by_id(neighbor, node.id)
            if node_to_modify is None:
                continue
            node_to_modify.children = [right, left]
            neighbors.append(neighbor)
    return neighbors


def find_node_by_id(tree, target_id):
    if tree.id == target_id:
        return tree
    for child in tree.children:
        result = find_node_by_id(child, target_id)
        if result:
            return result
    return None


def find_max_parsimony_tree_with_nni(sequences, initial_tree):
    best_tree = initial_tree
    best_score = fitch_parsimony(best_tree, sequences)

    improved = True
    while improved:
        improved = False
        neighbors = get_nni_neighbors(best_tree)

        for neighbor in neighbors:
            score = fitch_parsimony(neighbor, sequences)
            if score < best_score:
                best_tree = neighbor
                best_score = score
                improved = True
                break

    return best_tree, best_score


def tree_to_newick(node):
    if node.is_leaf():
        return node.name
    return f"({','.join(tree_to_newick(child) for child in node.children)})"


fasta_file_path = "/Users/iphonebiubiubiu/Desktop/par-test/seq50-500.fasta"  # Replace with your actual file path
sequences = read_sequences(fasta_file_path)
distance_matrix, labels = create_hamming_distance_matrix(sequences)

nj_tree_newick = phylogenetic_reduction(distance_matrix, labels)
initial_tree = parse_newick(nj_tree_newick)
max_parsimony_tree, best_score = find_max_parsimony_tree_with_nni(sequences, initial_tree)

print("NJ Tree (Newick format without branch lengths):")
print(nj_tree_newick)
print("\nParsimony Tree (Newick format without branch lengths):")
print(tree_to_newick(max_parsimony_tree))
print(f"\nThe length of the best tree (parsimony score) is: {best_score}")