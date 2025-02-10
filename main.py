import numpy as np
from random import uniform

norm = np.linalg.norm


def find_dk(Bk: np.ndarray, Fxk: np.ndarray):
    dk = np.linalg.solve(Bk, -Fxk)
    return dk


def get_new_Bk(Bk: np.ndarray, sk: np.ndarray, yk: np.ndarray):
    return Bk + (np.outer((yk - (Bk @ sk)), sk) / (sk @ sk))


def F(x: np.ndarray):
    return np.array([x[0] ** 2 - x[1], x[0] + x[1]])


def eta(k: int):
    return 1 / ((k + 1) ** 2)


def solver():
    xk = np.array([-0.5, 0.5])
    Bk = np.array([[1, 0], [0, 1]])
    Fxk = F(xk)
    dk = find_dk(Bk, Fxk)
    k = 0
    th = 0.5
    s = 0.5
    t_min = 1 / 3
    t_max = 2 / 3
    a = 1
    while True:
        print(k)
        if norm(F(xk + a * dk)) <= (1 + a * s * (th - 1)) * norm(Fxk) + eta(k):
            k += 1
            new_xk = xk + a * dk
            new_Fxk = F(new_xk)
            if norm(new_Fxk) < 10e-4:
                return new_xk
            sk = new_xk - xk
            yk = new_Fxk - Fxk
            Bk = get_new_Bk(Bk, sk, yk)
            dk = find_dk(Bk, new_Fxk)
            xk = new_xk
            Fxk = new_Fxk
            a = 1
        else:
            a = uniform(a * t_min, a * t_max)


def main():
    x = solver()
    print("Solution found:")
    print(x)


# x0 = (2,3) -> xk = (-0.99949852  0.99949766), k = 9
# x0 = (0.2,0.1) -> xk = (0.0001273 -0.0001305), k = 4
# x0 = (-0.5,0.5) -> xk = (-1.00006108  1.00006108), k = 5


if __name__ == "__main__":
    main()
