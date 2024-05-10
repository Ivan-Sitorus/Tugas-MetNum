import numpy as np

def metode_matriks_balikan(A, b):
  """Mencari solusi sistem persamaan linear dengan metode matriks balikan.

  Args:
    A: Matriks koefisien sistem persamaan linear.
    b: Vektor konstanta sistem persamaan linear.

  Returns:
    Vektor solusi x.
  """
  x = np.linalg.inv(A) @ b
  return x

def metode_dekomposisi_LU_Gauss(A, b):
  """Mencari solusi sistem persamaan linear dengan metode dekomposisi LU Gauss.

  Args:
    A: Matriks koefisien sistem persamaan linear.
    b: Vektor konstanta sistem persamaan linear.

  Returns:
    Vektor solusi x.
  """
  P, L, U = np.linalg.lu(A)
  y = np.linalg.solve(L, P @ b)
  x = np.linalg.solve(U, y)
  return x

def metode_dekomposisi_Crout(A, b):
  """Mencari solusi sistem persamaan linear dengan metode dekomposisi Crout.

  Args:
    A: Matriks koefisien sistem persamaan linear.
    b: Vektor konstanta sistem persamaan linear.

  Returns:
    Vektor solusi x.
  """
  n = A.shape[0]
  L = np.eye(n)
  U = np.copy(A)

  for i in range(n):
    for j in range(i + 1, n):
      L[j, i] = U[j, i] / U[i, i]
      U[j, i] = 0

  y = np.linalg.solve(L, b)
  x = np.linalg.solve(U, y)
  return x

def main():
  """Meminta input dari pengguna dan menampilkan solusi."""
  # Meminta input dari pengguna
  n = int(input("Masukkan jumlah variabel: "))
  A = np.zeros((n, n))
  b = np.zeros(n)

  for i in range(n):
    for j in range(n):
      A[i, j] = float(input(f"Masukkan elemen A[{i+1},{j+1}]: "))
    b[i] = float(input(f"Masukkan nilai b[{i+1}]: "))

  # Menampilkan solusi dengan metode matriks balikan
  x_balikan = metode_matriks_balikan(A, b)
  print("\nSolusi dengan metode matriks balikan:")
  print(x_balikan)

  # Menampilkan solusi dengan metode dekomposisi LU Gauss
  x_LU_Gauss = metode_dekomposisi_LU_Gauss(A, b)
  print("\nSolusi dengan metode dekomposisi LU Gauss:")
  print(x_LU_Gauss)

  # Menampilkan solusi dengan metode dekomposisi Crout
  x_Crout = metode_dekomposisi_Crout(A, b)
  print("\nSolusi dengan metode dekomposisi Crout:")
  print(x_Crout)

if __name__ == "__main__":
  main()