import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Define gate parameter values based on the table.
GATE_PARAMETERS = {
    'inv': {'inputs': 1, 'a': 3, 'r': 0.333, 'c_in': 3, 'c_int': 3},
    'nand': {'inputs': 2, 'a': 8, 'r': 0.333, 'c_in': 4, 'c_int': 6},
    'nor': {'inputs': 2, 'a': 10, 'r': 0.333, 'c_in': 5, 'c_int': 6},
    'aoi21': {'inputs': 3, 'a': 17, 'r': 0.333, 'c_in': 6, 'c_int': 7},
    'oai21': {'inputs': 3, 'a': 16, 'r': 0.333, 'c_in': 6, 'c_int': 7}
}

def get_gate_parameters(gate_type, num_inputs):
    """Fetch gate parameters based on type and number of inputs."""
    return GATE_PARAMETERS.get(gate_type, {
        'a': 5 * num_inputs,
        'r': 0.333,
        'c_in': 2.3 * num_inputs,
        'c_int': 3 * num_inputs
    })

def read_netlist_from_file(file_path):
    """Read the netlist content from a specified file path."""
    try:
        with open(file_path, "r") as file:
            netlist_content = file.read()
        return netlist_content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

def replace_and_or_with_nand_nor_inverter(netlist):
    """Replace AND/OR gates with NAND/NOR followed by INV and replace BUFF gates with two consecutive INV gates."""
    and_pattern = r"(\w+)\s*=\s*AND\s*\(([\w\s,]+)\)"
    or_pattern = r"(\w+)\s*=\s*OR\s*\(([\w\s,]+)\)"
    buff_pattern = r"(\w+)\s*=\s*BUFF\s*\(([\w\s,]+)\)"

    modified_netlist = []
    gate_counter = 1

    for line in netlist.splitlines():
        match_and = re.match(and_pattern, line, re.IGNORECASE)
        match_or = re.match(or_pattern, line, re.IGNORECASE)
        match_buff = re.match(buff_pattern, line, re.IGNORECASE)

        if match_and:
            output, inputs = match_and.groups()
            inputs = [inp.strip() for inp in inputs.split(',')]
            intermediate_wire = f"{output}_{gate_counter}"
            gate_counter += 1
            # Replace AND with NAND
            modified_netlist.append(f"{intermediate_wire} = NAND({', '.join(inputs)})")
            modified_netlist.append(f"{output} = INV({intermediate_wire})")

        elif match_or:
            output, inputs = match_or.groups()
            inputs = [inp.strip() for inp in inputs.split(',')]
            intermediate_wire = f"{output}_{gate_counter}"
            gate_counter += 1
            # Replace OR with NOR
            modified_netlist.append(f"{intermediate_wire} = NOR({', '.join(inputs)})")
            modified_netlist.append(f"{output} = INV({intermediate_wire})")

        elif match_buff:
            output, input_signal = match_buff.groups()
            # Replace BUFF with two consecutive INV gates
            modified_netlist.append(f"{output}_temp = INV({input_signal})")
            modified_netlist.append(f"{output} = INV({output}_temp)")

        else:
            modified_netlist.append(line)

    return "\n".join(modified_netlist)


def extract_gate_outputs_in_order(gate_list):
    """Extract gate outputs from the gate list in the order they appear."""
    gate_outputs = [gate[0] for gate in gate_list]
    return gate_outputs

def parse_netlist(netlist):
    """Parse the netlist to identify inputs, outputs, and gates."""
    input_pattern = r"INPUT\((\w+)\)"
    output_pattern = r"OUTPUT\((\w+)\)"
    gate_pattern = r"(\w+)\s*=\s*(\w+)\(([\w,\s]+)\)"

    inputs = re.findall(input_pattern, netlist)
    outputs = re.findall(output_pattern, netlist)
    gates = []

    for output, gate_type, gate_inputs in re.findall(gate_pattern, netlist):
        gate_inputs = [inp.strip() for inp in gate_inputs.split(',')]
        gates.append((output, gate_type.lower(), gate_inputs))

    return inputs, outputs, gates

def build_dag(gate_list):
    """Construct a directed acyclic graph from the gate list."""
    edges = defaultdict(list)
    gate_outputs = extract_gate_outputs_in_order(gate_list)  # Use the custom order

    for gate_output, gate_type, gate_inputs in gate_list:
        for gate_input in gate_inputs:
            edges[gate_input].append(gate_output)

    return gate_outputs, edges

def generate_fanout_matrix(gate_outputs, edges, gate_list):
    """Generate a fan-out matrix for the gates."""
    node_index = {node: idx for idx, node in enumerate(gate_outputs)}
    n = len(gate_outputs)
    fan_out_matrix = np.zeros((n, n), dtype=int)

    for gate_output, gate_type, gate_inputs in gate_list:
        num_inputs = len(gate_inputs)
        params = get_gate_parameters(gate_type, num_inputs)
        a, r, c_in, c_int = params['a'], params['r'], params['c_in'], params['c_int']

    for src, dst_list in edges.items():
        if src in node_index:
            src_idx = node_index[src]
            for dst in dst_list:
                if dst in node_index:
                    dst_idx = node_index[dst]
                    
                    fan_out_matrix[src_idx, dst_idx] =  r * c_in     #1

    return fan_out_matrix, node_index

def calculate_delays(gate_outputs, fan_out_matrix, node_index, gate_list):
    """Calculate delays for each gate output."""
    delays = np.zeros(len(gate_outputs))
    g = np.zeros(len(gate_outputs))
    for gate_output, gate_type, gate_inputs in gate_list:
        num_inputs = len(gate_inputs)
        params = get_gate_parameters(gate_type, num_inputs)

        a, r, c_in, c_int = params['a'], params['r'], params['c_in'], params['c_int']
        output_idx = node_index[gate_output]

        sum_c_in = sum(fan_out_matrix[output_idx, node_index.get(inp, -1)] * c_in
                       for inp in gate_inputs if inp in node_index)
 #       delays[output_idx] = r * a * (c_int + sum_c_in)
        delays[output_idx] = r * c_int 
    

    return delays

def topological_sort(gate_outputs, edges):
    """Perform a topological sort on the gate outputs based on dependencies."""
    G = nx.DiGraph()
    G.add_nodes_from(gate_outputs)
    for src, dst_list in edges.items():
        for dst in dst_list:
            G.add_edge(src, dst)

    sorted_gate_outputs = list(nx.topological_sort(G))
    return sorted_gate_outputs

def visualize_dfg(gate_outputs, edges):
    """Visualize the Data Flow Graph (DFG)."""
    G = nx.DiGraph()
    G.add_nodes_from(gate_outputs)
    for src, dst_list in edges.items():
        for dst in dst_list:
            G.add_edge(src, dst)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=12, font_weight='bold', arrows=True)
    plt.title("Data Flow Graph (DFG)", fontsize=16)
    plt.show()

def print_delays_in_custom_order(sorted_gate_outputs, delays):
    """Print the delays in the custom order specified by the user."""
    intermediate_gates = []
    primary_gates = []
    remaining_gates = []

    # Separate intermediate and primary gates
    for gate in sorted_gate_outputs:
        if "_" in gate:
            intermediate_gates.append(gate)
        elif gate.isdigit() or re.match(r"\d+$", gate):
            primary_gates.append(gate)
        else:
            remaining_gates.append(gate)

    # Sort intermediate gates based on their primary gate
    intermediate_gates.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

    # Prepare the custom order
    custom_order = primary_gates + intermediate_gates + remaining_gates

    # Print delays in the desired order
    print("\nDelays for each gate output:")
    for gate_output in custom_order:
        delay_idx = sorted_gate_outputs.index(gate_output)
        print(f"Gate Output {gate_output}: Delay = {delays[delay_idx]:.4f}")

def generate_FI_FO(F):
    # Number of gates (size of the matrix)
    n = F.shape[0]

    # Initialize dictionaries to store FI(i) and FO(i) sets
    FI = {i: set() for i in range(n)}
    FO = {i: set() for i in range(n)}
    i  = 1
    j  = 1
    # Loop over each gate (row and column)
    for i in range(n):
        for j in range(i + 1, n):  # We only care about j > i (upper triangular)
            if F[i, j] == 1:
                # Gate i drives gate j, so add j to FO(i)
                FO[i].add(j+1)
                # Gate j is driven by gate i, so add i to FI(j)
                FI[j].add(i+1)
    
    return FI, FO

def format_FI_FO(fi_fo_dict):
    """Convert FI or FO dictionary to a list of sets with '{}' for empty sets."""
    formatted_list = []
    for i in range(len(fi_fo_dict)):
        # Replace empty set with {}
        formatted_list.append(fi_fo_dict[i] if fi_fo_dict[i] else {})
    return formatted_list

def main():
    file_path = "/home/ms2024007/Documents/dcmos_project/bench_files/net2.bench"
    netlist = read_netlist_from_file(file_path)

    if netlist:
        modified_netlist = replace_and_or_with_nand_nor_inverter(netlist)
        inputs, outputs, gate_list = parse_netlist(modified_netlist)
        
        # Build the DAG using the custom order of gate outputs
        gate_outputs, edges = build_dag(gate_list)

        # Print the Gate List
        print("Gate List:")
        for gate in gate_list:
            print(f"{gate[0]} = {gate[1].upper()}({', '.join(gate[2])})")

        # Perform topological sorting
        sorted_gate_outputs = topological_sort(gate_outputs, edges)
        
        # Exclude input nodes from delay calculation (inputs have no delay)
        sorted_gate_outputs = [gate for gate in sorted_gate_outputs if gate not in inputs]

        # Generate the fan-out matrix
        fan_out_matrix, node_index = generate_fanout_matrix(gate_outputs, edges, gate_list)
        fan_out_matrix = np.array(fan_out_matrix)
        formatted_fan_out_matrix = [list(item) for item in fan_out_matrix]

        FI, FO = generate_FI_FO(fan_out_matrix)

        FI = format_FI_FO(FI)
        formatted_FI = [list(item) for item in FI]
        FO = format_FI_FO(FO)
        formatted_FO = [list(item) for item in FO]

        # Calculate delays for each gate
        delays = calculate_delays(gate_outputs, fan_out_matrix, node_index, gate_list)
        print("\nFan-out Matrix:")
        print(formatted_fan_out_matrix)
        print(formatted_FI)
        print(formatted_FO)
        print (delays)

        # Visualize the Data Flow Graph (DFG)
        visualize_dfg(gate_outputs, edges)
        
        # Print the fan-out matrix
        

        # Print the delays in the custom order
        print_delays_in_custom_order(sorted_gate_outputs, delays)

if __name__ == "__main__":
    main()