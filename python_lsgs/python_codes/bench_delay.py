import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Define gate parameter values based on the table.
GATE_PARAMETERS = {
    'inv': {'inputs': 1, 'a': 3, 'r': 0.333, 'c_in': 3, 'c_int': 3},
    'nand2': {'inputs': 2, 'a': 8, 'r': 0.333, 'c_in': 4, 'c_int': 6},
    'nor2': {'inputs': 2, 'a': 10, 'r': 0.333, 'c_in': 5, 'c_int': 6},
    'aoi21': {'inputs': 3, 'a': 17, 'r': 0.333, 'c_in': 6, 'c_int': 7},
    'oai21': {'inputs': 3, 'a': 16, 'r': 0.333, 'c_in': 6, 'c_int': 7}
}

def parse_bench(file_path):
    def extract_name(identifier):
        if '$' in identifier:
            parts = identifier.split('_')
            return parts[-1] if len(parts) > 1 else identifier
        return identifier

    with open(file_path, 'r') as file:
        lines = file.readlines()

    inputs = []
    outputs = []
    gates = []

    for line in lines:
        line = line.strip()
        if line.startswith('INPUT'):
            inputs.append(extract_name(re.search(r'INPUT\((\w+)\)', line).group(1)))
        elif line.startswith('OUTPUT'):
            outputs.append(extract_name(re.search(r'OUTPUT\((\w+)\)', line).group(1)))
        else:
            match = re.match(r"(\w+)\s*=\s*(\w+)\((.*)\)", line)
            if match:
                output, gate_type, gate_inputs = match.groups()
                gates.append((extract_name(output), gate_type.lower(), [extract_name(inp) for inp in gate_inputs.split(',')]))

    return inputs, outputs, gates

def build_dag(gate_list):
    nodes = set()
    edges = defaultdict(list)

    for gate_output, gate_type, gate_inputs in gate_list:
        nodes.add(gate_output)
        for gate_input in gate_inputs:
            nodes.add(gate_input)
            edges[gate_input].append(gate_output)

    return list(nodes), edges

def generate_fanout_matrix(nodes, edges):
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    fan_out_matrix = np.zeros((n, n), dtype=int)

    for src, dst_list in edges.items():
        src_idx = node_index[src]
        for dst in dst_list:
            dst_idx = node_index[dst]
            fan_out_matrix[src_idx, dst_idx] = 1

    return fan_out_matrix, node_index

def calculate_delays(nodes, fan_out_matrix, node_index, gate_list):
    n = len(nodes)
    delays = np.zeros(n)

    for gate_output, gate_type, gate_inputs in gate_list:
        if gate_type in GATE_PARAMETERS:
            params = GATE_PARAMETERS[gate_type]
            a = params['a']
            r = params['r']
            c_int = params['c_int']
            c_in = params['c_in']
        else:
            continue  # Skip if the gate type is not in our parameter table

        output_idx = node_index[gate_output]
        sum_c_in = sum(fan_out_matrix[output_idx, node_index[input_node]] * c_in for input_node in gate_inputs)
        delays[output_idx] = r * a * (c_int + sum_c_in)

    return delays

def visualize_dfg(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for src, dst_list in edges.items():
        for dst in dst_list:
            G.add_edge(src, dst)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=12, font_weight='bold', arrows=True)
    plt.title("Data Flow Graph (DFG)", fontsize=16)
    plt.show()

def main(bench_file_path):
    inputs, outputs, gate_list = parse_bench(bench_file_path)
    nodes, edges = build_dag(gate_list)
    fan_out_matrix, node_index = generate_fanout_matrix(nodes, edges)
    delays = calculate_delays(nodes, fan_out_matrix, node_index, gate_list)

    print("Nodes:", nodes)
    print("\nFan-out Matrix:")
    print(fan_out_matrix)

    print("\nDelays for each node:")
    for node, delay in zip(nodes, delays):
        print(f"Node {node}: Delay = {delay:.4f}")

    visualize_dfg(nodes, edges)

# Example usage with the .bench file
bench_file_path = '/home/ms2024007/Documents/dcmos_project/random_bench_circuits/random_circuit.bench'
if __name__ == "__main__":
    main(bench_file_path)
