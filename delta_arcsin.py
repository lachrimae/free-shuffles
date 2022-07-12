import random
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


class BlockMatrix:
  def __init__(self, block_dim: int, entries: np.ndarray):
    if len(entries) != 2:
      raise ValueError('Needs to be a matrix')
    elif entries.shape[0] != entries.shape[1]:
      raise ValueError('Needs to be a square matrix')
    elif entries.shape[0] % block_dim != 0:
      raise ValueError('Underlying dimension needs to be divisible by block dimension')
    self.entries = entries
    self.block_dim = block_dim

  def _translate_index(self, i: int) -> tuple[int]:
    return (i * self.block_dim, (i + 1) * self.block_dim)
  
  def __len__(self) -> int:
    return self.entries.shape[0] // self.block_dim
  
  def __getitem__(self, indices: tuple[int]) -> np.ndarray:
    i, j = indices
    i_start, i_end = self._translate_index(i)
    j_start, j_end = self._translate_index(j)
    return self.entries[i_start:i_end, j_start:j_end]
  
  def __setitem__(self, indices: tuple[int], matrix: np.ndarray):
    i, j = indices
    i_start, i_end = self._translate_index(i)
    j_start, j_end = self._translate_index(j)
    for k in range(i_start, i_end):
      for l in range(j_start, j_end):
        self.entries[k][l] = matrix[k - i_start][l - j_start]

  def trace(self) -> np.ndarray:
    res = np.zeros(self.block_dim, dtype=complex)
    for i in range(len(self)):
      res += self[i][i]


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