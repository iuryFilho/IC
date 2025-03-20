import numpy as np
from random import uniform
from numpy.linalg import norm
from typing import Callable
from numpy.typing import NDArray
from numbers import Number


def find_dk(Bk: NDArray, fgxk: NDArray, sec: NDArray) -> NDArray:
    dk = np.linalg.solve(Bk + sec, -fgxk)
    return dk


def get_new_Bk(Bk: NDArray, sk: NDArray, yk: NDArray) -> NDArray:
    return Bk + (np.outer((yk - (Bk @ sk)), sk) / (sk @ sk))


def _eta(k: int):
    return 1 / ((k + 1) ** 2)


def secant(
    x: NDArray,
    y: NDArray,
    g_list: list[Callable[[NDArray], NDArray]],
) -> NDArray:
    n = len(x)
    res = np.ndarray((n, n))
    x0y1 = np.ndarray([x[0], y[1]])
    for i in range(n):
        gx0y1 = g_list[i](x0y1)
        res[i][0] = (gx0y1 - g_list[i](np.ndarray([y[0], y[1]]))) / (x[0] - y[0])
        res[i][1] = (g_list[i](np.ndarray([x[0], x[1]])) - gx0y1) / (x[1] - y[1])
    return res


def def_g(g_list):
    return lambda x: np.array([g(x) for g in g_list])


def solver(
    x0: NDArray,
    x1: NDArray,
    B0: NDArray,
    f: Callable[[NDArray], NDArray],
    g_list: list[Callable[[NDArray], NDArray]],
    th: float = 0.5,
    s: float = 0.5,
    tol: float = 10e-4,
    t_min: float = 1 / 3,
    t_max: float = 2 / 3,
    eta: Callable[[Number], float] = _eta,
):
    g = def_g(g_list)
    fg = lambda x: f(x) + g(x)
    xk_1 = x0
    xk = x1
    Bk = B0
    fgxk = fg(xk)
    sec = secant(xk_1, xk, g_list)
    dk = find_dk(Bk, fgxk, sec)
    a = 1

    log: dict[str, list[NDArray]] = {"xk": [xk_1], "f+g": [fg(xk_1)]}
    log["xk"].append(xk)
    log["f+g"].append(fgxk)
    k = 1
    while True:
        if norm(fg(xk + a * dk)) <= (1 + a * s * (th - 1)) * norm(fgxk) + eta(k):
            k += 1
            new_xk = xk + a * dk
            new_fgxk = fg(new_xk)
            log["xk"].append(new_xk)
            log["f+g"].append(new_fgxk)
            if norm(new_fgxk) < tol:
                return new_xk, log
            sk = new_xk - xk
            yk = new_fgxk - fgxk
            Bk = get_new_Bk(Bk, sk, yk)
            sec = secant(xk, new_xk, g_list)
            dk = find_dk(Bk, new_fgxk, sec)
            xk_1 = xk
            xk = new_xk
            fgxk = new_fgxk
            a = 1
        else:
            a = uniform(a * t_min, a * t_max)


def flog(log) -> str:
    return np.array2string(log, suppress_small=True)


def main():
    def f1(x: NDArray) -> NDArray:
        return np.array([x[0] ** 2 - x[1], x[0] + x[1]])

    g1_list = [
        lambda x: x[0] ** 2 - x[1] + 1 + abs(x[0] - 1) / 9,
        lambda x: x[0] + x[1] - 7 + abs(x[1]) / 9,
    ]

    x0 = np.array([1, 0])
    x1 = np.array([1, 1])
    B0 = np.array([[1, 0], [0, 1]])

    x, log = solver(x0, x1, B0, f1, g1_list)
    print("x0:")
    print(np.array2string(x0))
    print("x1:")
    print(np.array2string(x1))
    print("B0:")
    print(np.array2string(B0))
    print("Solution found:")
    print(x)
    print("Log:")
    for i in range(len(log["xk"])):
        print(
            f"x{i} = {flog(log['xk'][i])}, (f+g){i} = {flog(log['f+g'][i])}, ||(f+g){i}|| = {flog(norm(log['f+g'][i]))}"
        )


if __name__ == "__main__":
    main()
