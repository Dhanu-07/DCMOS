import numpy as np

def nlbs_softmax(n, Ft, Fti, Ftj, g, p, u, x, umax):
    for i in range(n - 1, -1, -1):  # Process nodes in reverse order
        umax[i] = g[i]
        k = 0
        while Ftj[i + 1] - Ftj[i] - k > 0:
            j = Ftj[i] + k
            fog = Fti[j]
            umax[i] += Ft[j] * x[fog]
            k += 1
        
        xtemp = umax[i] / u[i]
        if xtemp <= 1:
            x[i] = (xtemp ** p + 1) ** (1 / p)
        else:
            x[i] = xtemp * (1 + xtemp ** -p) ** (1 / p)

    return x, umax

'''# Example usage:
n = 5  # Number of gates/nodes
Ft = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Example Ft values
Fti = np.array([1, 2, 3, 4])  # Row indices
Ftj = np.array([0, 1, 2, 3, 4, 4])  # Column index pointers
g = np.array([1.2, 1.3, 1.4, 1.1, 1.5])  # g values for each node
p = 2.0  # Softmax parameter
u = np.array([1.5, 2.1, 1.8, 2.3, 1.7])  # Maximum delays for each node
x = np.ones(n)  # Initialize x with ones
umax = np.zeros(n)  # Initialize umax array

x, umax = nlbs_softmax(n, Ft, Fti, Ftj, g, p, u, x, umax)
print("Updated x values:", x)
print("Updated umax values:", umax)'''

