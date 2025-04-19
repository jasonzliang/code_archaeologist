import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import time

# Set random seed for reproducibility
np.random.seed(42)

def print_with_timing(description, operation_func):
    """Execute an operation with timing and formatted output"""
    start_time = time.time()
    result = operation_func()
    elapsed = time.time() - start_time
    print(f"\n{'-'*50}")
    print(f"{description} (computed in {elapsed:.6f} seconds):")
    print(f"{'-'*50}")
    return result

# Create complex matrices
n = 5
A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
B = np.random.rand(n, n) + 1j * np.random.rand(n, n)
v = np.random.rand(n) + 1j * np.random.rand(n)

# Print the original matrices with formatting
print_with_timing("Matrix A (complex)", lambda: A)
print_with_timing("Matrix B (complex)", lambda: B)
print_with_timing("Vector v (complex)", lambda: v)

# Singular Value Decomposition
U, S, Vh = print_with_timing("SVD of A: Singular values",
                            lambda: linalg.svd(A))
print(f"Singular values: {S}")

# Calculate condition number
cond_num = print_with_timing("Condition number of A",
                            lambda: np.linalg.cond(A))
print(f"Condition number: {cond_num:.4e}")

# Compute matrix exponential
exp_A = print_with_timing("Matrix exponential e^A",
                        lambda: linalg.expm(A))

# Solve linear system Ax = v
x = print_with_timing("Solution to linear system Ax = v",
                    lambda: linalg.solve(A, v))
print(f"Solution vector x:\n{x}")

# Verify solution
residual = print_with_timing("Residual ||Ax - v||",
                            lambda: np.linalg.norm(A @ x - v))
print(f"Residual norm: {residual:.4e}")

# Find eigendecomposition
eigenvalues, eigenvectors = print_with_timing("Eigendecomposition of A",
                                            lambda: linalg.eig(A))
print(f"Eigenvalues:\n{eigenvalues}")

# Calculate trace and determinant
trace_A = print_with_timing("Trace of A", lambda: np.trace(A))
print(f"Trace: {trace_A}")

det_A = print_with_timing("Determinant of A", lambda: np.linalg.det(A))
print(f"Determinant: {det_A}")

# Calculate Schur decomposition
T, Z = print_with_timing("Schur decomposition", lambda: linalg.schur(A))
print(f"Schur form T:\n{T}")

# Calculate QR decomposition
Q, R = print_with_timing("QR decomposition", lambda: linalg.qr(A))
print(f"Q is unitary: {np.allclose(Q @ Q.conj().T, np.eye(n), atol=1e-10)}")
print(f"A = QR verification error: {np.linalg.norm(A - Q @ R):.4e}")

# Compute matrix logarithm
log_A = print_with_timing("Matrix logarithm log(A)", lambda: linalg.logm(A))

# Calculate matrix power
power_A = print_with_timing("Matrix power A^3", lambda: linalg.inv(A) @ A @ A)
