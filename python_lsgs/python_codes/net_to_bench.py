import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_blif(file_path):
    def extract_name(identifier):
        if '$' in identifier:
            parts = identifier.split('_')
            return parts[-1] if len(parts) > 1 else identifier
        return identifier

    with open(file_path, 'r') as file:
        lines = file.readlines()

    mappings = []
    for line in lines:
        if line.startswith('.subckt'):
            parts = line.split()
            gate_type = parts[1]
            inputs = []
            output = None
            for part in parts[2:]:
                if '=' in part:
                    key, value = part.split('=')
                    if key == 'Y':
                        output = extract_name(value)
                    else:
                        inputs.append(extract_name(value))
            if output:
                mappings.append(f"{output} = {gate_type.upper()}({', '.join(inputs)})")

    return mappings

def save_as_bench(mappings, output_file_path):
    primary_inputs = set()
    primary_outputs = set()
    all_signals = set()
    gates = []
    gate_count = 0
    gate_types_count = defaultdict(int)

    for mapping in mappings:
        parts = re.match(r"(\w+)\s*=\s*(\w+)\((.*)\)", mapping)
        if parts:
            output, gate_type, gate_inputs = parts.groups()
            gate_inputs_list = gate_inputs.split(', ')
            primary_inputs.update(gate_inputs_list)
            primary_outputs.add(output)
            all_signals.update(gate_inputs_list)
            all_signals.add(output)
            gates.append(f"{output} = {gate_type.upper()}({', '.join(gate_inputs_list)})")
            gate_count += 1
            gate_types_count[gate_type.upper()] += 1

    # Correct input and output sets
    intermediate_signals = all_signals - primary_inputs - primary_outputs

    input_count = len(primary_inputs)
    output_count = len(primary_outputs)
    inv_count = gate_types_count.get('INV', 0)

    with open(output_file_path, 'w') as file:
        file.write(f"# blif_file_path \n")
        file.write(f"# {input_count} inputs\n")
        file.write(f"# {output_count} outputs\n")
        file.write(f"# {inv_count} inverter\n")
        file.write(f"# {gate_count} gates ({', '.join([f'{count} {gt}' for gt, count in gate_types_count.items()])})\n\n")

        for inp in primary_inputs:
            file.write(f"INPUT({inp})\n")
        file.write("\n")
        for out in primary_outputs:
            file.write(f"OUTPUT({out})\n")
        file.write("\n")
        for gate in gates:
            file.write(f"{gate}\n")
        file.write("\n")
        file.write("# Intermediate signals\n")
        for signal in intermediate_signals:
            file.write(f"{signal}\n")

# Example usage
blif_file_path = '/home/ms2024007/Documents/dcmos_project/netlist_yosys_blif/netlist_2.blif'
bench_file_path = '/home/ms2024007/Documents/dcmos_project/bench_files/net2.bench'
mappings = parse_blif(blif_file_path)
save_as_bench(mappings, bench_file_path)
print(f"Converted mappings saved to {bench_file_path}")
