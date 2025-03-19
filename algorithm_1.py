import numpy as np
from random import uniform
from numpy.linalg import norm
from typing import Callable
from numpy.typing import NDArray
from numbers import Number


def find_dk(Bk: NDArray, Fxk: NDArray) -> NDArray:
    dk = np.linalg.solve(Bk, -Fxk)
    return dk


def get_new_Bk(Bk: NDArray, sk: NDArray, yk: NDArray) -> NDArray:
    return Bk + (np.outer((yk - (Bk @ sk)), sk) / (sk @ sk))


def _eta(k: int):
    return 1 / ((k + 1) ** 2)


def solver(
    x0: NDArray,
    B0: NDArray,
    F: Callable[[NDArray], NDArray],
    th: float = 0.5,
    s: float = 0.5,
    tol: float = 10e-4,
    t_min: float = 1 / 3,
    t_max: float = 2 / 3,
    eta: Callable[[Number], float] = _eta,
):
    xk = x0
    Bk = B0
    Fxk = F(xk)
    dk = find_dk(Bk, Fxk)
    a = 1

    log: dict[str, list[NDArray]] = {"xk": [xk], "Fxk": [Fxk]}
    k = 0
    while True:
        if norm(F(xk + a * dk)) <= (1 + a * s * (th - 1)) * norm(Fxk) + eta(k):
            k += 1
            new_xk = xk + a * dk
            new_Fxk = F(new_xk)
            log["xk"].append(new_xk)
            log["Fxk"].append(new_Fxk)
            if norm(new_Fxk) < tol:
                return new_xk, log
            sk = new_xk - xk
            yk = new_Fxk - Fxk
            Bk = get_new_Bk(Bk, sk, yk)
            dk = find_dk(Bk, new_Fxk)
            xk = new_xk
            Fxk = new_Fxk
            a = 1
        else:
            a = uniform(a * t_min, a * t_max)


def flog(log) -> str:
    return np.array2string(log, suppress_small=True)


def main():
    def F1(x: NDArray) -> NDArray:
        return np.array([x[0] ** 2 - x[1], x[0] + x[1]])

    x0 = np.array([-0.5, 0.5])
    B0 = np.array([[1, 0], [0, 1]])

    x, log = solver(x0, B0, F1)
    print("x0:")
    print(np.array2string(x0))
    print("B0:")
    print(np.array2string(B0))
    print("Solution found:")
    print(x)
    print("Log:")
    for i in range(len(log["xk"])):
        print(
            f"x{i} = {flog(log['xk'][i])}, F{i} = {flog(log['Fxk'][i])}, ||F{i}|| = {flog(norm(log['Fxk'][i]))}"
        )


# x0 = (2,3) -> xk = (-0.99949852  0.99949766), k = 9
# x0 = (0.2,0.1) -> xk = (0.0001273 -0.0001305), k = 4
# x0 = (-0.5,0.5) -> xk = (-1.00006108  1.00006108), k = 5


if __name__ == "__main__":
    main()
