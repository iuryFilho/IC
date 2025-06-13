import numpy as np
from random import uniform
from numpy.linalg import norm
from typing import Callable
from numpy.typing import NDArray
from numbers import Number
import time


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
    res = np.empty((n, n))
    for j in range(n):
        a = y.copy()
        b = y.copy()
        a[: j + 1] = x[: j + 1]
        b[:j] = x[:j]
        for i in range(n):
            res_x = g_list[i](a)
            res_y = g_list[i](b)
            res_dif = np.float64(x[j] - y[j])
            res[i][j] = (res_x - res_y) / res_dif
    return res


def def_g(g_list):
    return lambda x: np.array([g(x) for g in g_list])


def get_time() -> float:
    return time.time() * 1e3


def solver(
    x0: NDArray,
    x1: NDArray,
    B0: NDArray,
    f: Callable[[NDArray], NDArray],
    g_list: list[Callable[[NDArray], NDArray]],
    th: float = 0.5,
    s: float = 0.5,
    tol: float = 1e-4,
    t_min: float = 1 / 3,
    t_max: float = 2 / 3,
    eta: Callable[[Number], float] = _eta,
):
    time_mult = 1e3

    g = def_g(g_list)
    fg = lambda x: f(x) + g(x)
    xk_1 = x0
    xk = x1
    Bk = B0
    fgxk = fg(xk)
    sec = secant(xk_1, xk, g_list)
    dk = find_dk(Bk, fgxk, sec)
    a = 1

    log: dict[str, list[NDArray]] = {"xk": [xk_1], "f+g": [fg(xk_1)], "time": [0.0]}
    log["xk"].append(xk)
    log["f+g"].append(fgxk)
    log["time"].append(0.0)
    k = 1

    time_start = get_time()
    while True:
        if norm(fg(xk + a * dk)) <= (1 + a * s * (th - 1)) * norm(fgxk) + eta(k):
            k += 1
            new_xk = xk + a * dk
            new_fgxk = fg(new_xk)
            if norm(new_fgxk) < tol:
                log["time"].append(get_time() - time_start)
                log["xk"].append(new_xk)
                log["f+g"].append(new_fgxk)
                return new_xk, log
            log["time"].append(get_time() - time_start)
            log["xk"].append(new_xk)
            log["f+g"].append(new_fgxk)
            time_start = get_time()
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


number_format = lambda x: f"{x:.6f}" if x < 0 else f" {x:.6f}"


def flog(log) -> str:
    return np.array2string(log, formatter={"float_kind": number_format})


def main():
    def f1(x: NDArray) -> NDArray:
        return np.array([x[0] ** 2 - x[1], x[0] + x[1]])

    g1_list = [
        lambda x: x[0] ** 2 - x[1] + 1 + abs(x[0] - 1) / 9,
        lambda x: x[0] + x[1] - 7 + abs(x[1]) / 9,
    ]

    x0 = np.array([0.0, 0.0])
    x1 = np.array([1.0, 1.0])
    B0 = np.array([[1.0, 0.0], [0.0, 1.0]])

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
    with open("log.txt", "w", encoding="utf-8") as f:
        f.write("k, xk, ||(f+g)||, time (ms) \n")
        for i in range(len(log["xk"])):
            f.write(
                f"{i}, {flog(log['xk'][i])}, {flog(norm(log['f+g'][i]))}, {log['time'][i]:.6f}\n"
            )
            print(
                f"x{i:02} = {flog(log['xk'][i])}, ||(f+g){i:02}|| = {flog(norm(log['f+g'][i]))}, time = {log['time'][i]:.6f}"
            )


if __name__ == "__main__":
    main()
