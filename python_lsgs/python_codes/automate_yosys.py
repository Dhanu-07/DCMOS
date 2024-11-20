import os

# Define the paths
liberty_file = "/home/ms2024007/Documents/dcmos_project/custom.lib"
verilog_dir = "/home/ms2024007/Documents/dcmos_project/verilog_files/"
netlist_v_dir = "/home/ms2024007/Documents/dcmos_project/netlist_yosys_v/"
netlist_blif_dir = "/home/ms2024007/Documents/dcmos_project/netlist_yosys_blif/"

# List of Verilog files
verilog_files = ["1.v", "2.v", "3.v", "4.v", "5.v", "6.v", "7.v", "8.v", "9.v", "10.v", "11.v"]

# Ensure output directories exist
os.makedirs(netlist_v_dir, exist_ok=True)
os.makedirs(netlist_blif_dir, exist_ok=True)

# Generate .tcl files and run Yosys
for i, verilog_file in enumerate(verilog_files, start=1):
    tcl_content = f"""
# Read the liberty and Verilog files
yosys -import
read_liberty -lib {liberty_file}
read_verilog {os.path.join(verilog_dir, verilog_file)}
opt
clean
# Check if the top module is correctly specified
synth
opt
clean
# Perform logic synthesis (optional but useful)
abc -liberty {liberty_file}
opt
clean
# Write the netlist to a file
write_verilog -noattr {os.path.join(netlist_v_dir, f"netlist_{i}.v")}
write_blif {os.path.join(netlist_blif_dir, f"netlist_{i}.blif")}
exit
"""
    # Write the .tcl file
    tcl_file_path = f"yosys{i}.tcl"
    with open(tcl_file_path, "w") as tcl_file:
        tcl_file.write(tcl_content)
    
    # Run Yosys with the .tcl file
    os.system(f"yosys -c {tcl_file_path}")

print("Yosys processing complete for all Verilog files.")
