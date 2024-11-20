import random

# Define gate types
gate_types = ['NAND', 'NOR', 'INV', 'AOI21', 'OAI21']

def generate_random_circuit(levels, gates_per_level, input_gates, output_gates):
    if len(gates_per_level) != levels:
        raise ValueError("The length of gates_per_level must match the number of levels.")

    gates = []
    total_gates = sum(gates_per_level)
    
    # Create a list of all gates with their levels
    for level in range(1, levels + 1):
        for gate_number in range(gates_per_level[level - 1]):
            gates.append((level, len(gates) + 1))
    
    circuit = {
        'inputs': [],
        'outputs': [],
        'gates': [],
        'fanout': [[] for _ in range(total_gates)]
    }
    
    input_gate_numbers = random.sample([gate[1] for gate in gates], input_gates)
    output_gate_numbers = random.sample([gate[1] for gate in gates], output_gates)
    
    for gate in input_gate_numbers:
        circuit['inputs'].append(gate)
    
    for gate in output_gate_numbers:
        circuit['outputs'].append(gate)
    
    # Assign inputs to each gate ensuring interconnection
    for gate in gates:
        level, gate_number = gate
        gate_type = random.choice(gate_types)
        available_gates = [g for g in range(1, gate_number) if g != gate_number]
        
        if gate_type == 'INV' and len(available_gates) >= 1:
            inputs = random.sample(available_gates, 1)
        elif gate_type in ['AOI21', 'OAI21'] and len(available_gates) >= 3:
            inputs = random.sample(available_gates, 3)
        elif len(available_gates) >= 2:
            inputs = random.sample(available_gates, 2)
        else:
            continue  # Skip gates that can't be formed due to lack of inputs
        
        circuit['gates'].append((gate_number, gate_type, inputs))
        for inp in inputs:
            circuit['fanout'][inp - 1].append(gate_number)
    
    return circuit

def output_bench_format(circuit):
    bench_output = []
    bench_output.append("# random circuit")
    bench_output.append(f"# {len(circuit['inputs'])} inputs")
    bench_output.append(f"# {len(circuit['outputs'])} outputs")
    bench_output.append(f"# 0 inverter")
    bench_output.append(f"# {len(circuit['gates'])} gates (various types)")

    for i in circuit['inputs']:
        bench_output.append(f"INPUT({i})")
    
    for i in circuit['outputs']:
        bench_output.append(f"OUTPUT({i})")
    
    for gate in circuit['gates']:
        bench_output.append(f"{gate[0]} = {gate[1]}({', '.join(map(str, gate[2]))})")
    
    return '\n'.join(bench_output)

def save_bench_file(content, filepath):
    with open(filepath, 'w') as file:
        file.write(content)

# Example usage:
levels = 100  # Increase the number of levels
gates_per_level = [1000] * 100  # Increase the number of gates per level
input_gates = 100
output_gates = 100

random_circuit = generate_random_circuit(levels, gates_per_level, input_gates, output_gates)
bench_format = output_bench_format(random_circuit)

# Save the output to a .bench file
filepath = "/home/ms2024007/Documents/dcmos_project/random_bench_circuits/random_circuit.bench"
save_bench_file(bench_format, filepath)

print(f"Bench file saved to: {filepath}")

# Displaying the fanout matrix
print("\nFanout Matrix:")
for idx, fo in enumerate(random_circuit['fanout']):
    print(f"Gate {idx + 1}: {fo}")
