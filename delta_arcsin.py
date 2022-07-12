import random
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def make_delta_matrix(n):
  """Returns an nxn diagonal matrix where 50% of the entries are 1.0 and 50% are -1.0."""
  mat = np.zeros([n, n], dtype=complex)
  for i in range(n):
    mat[i][i] = random.choice([1.0 + 0.0j, -1.0 + 0.0j])
  return mat


def model_delta_addition(n):
  d = make_delta_matrix(n)
  return np.linalg.eigvals(d + scramble_with_unitary(d))


def scramble_with_unitary(mat: np.ndarray) -> np.ndarray:
  u = stats.unitary_group.rvs(dim=mat.shape[0])
  return np.matmul(u, np.matmul(mat, u.conj().T))


def main():
  eigenvals = list(map(lambda c: c.real, model_delta_addition(1000)))
  fig, ax = plt.subplots(figsize=(10, 7))
  ax.hist(eigenvals, bins=[-2.5 + (0.05 * i) for i in range(int(5 / 0.05))])
  plt.savefig('eigenvalues.png')


if __name__ == '__main__':
  main()