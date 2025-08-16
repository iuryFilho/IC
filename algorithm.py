import numpy as np
from random import uniform
from numpy.linalg import norm
from typing import Callable, TextIO
from typing import Literal
from numpy.typing import NDArray
import time
import scipy.optimize
from to_overleaf import to_overleaf


def main():
    f1_list = [
        lambda x: x[0] ** 2 - x[1],
        lambda x: x[0] + x[1],
    ]

    g1_list = [
        lambda x: x[0] ** 2 - x[1] + 1 + abs(x[0] - 1) / 9,
        lambda x: x[0] + x[1] - 7 + abs(x[1]) / 9,
    ]

    if len(f1_list) != len(g1_list):
        raise ValueError("f1_list e g1_list devem ter o mesmo tamanho")

    x0 = np.array([0.0, 0.0])
    x1 = np.array([1.0, 1.0])
    if any(x0 == x1):
        raise ValueError("x0 e x1 não podem ter valores iguais em um mesmo índice")

    B0 = np.array([[1.0, 0.0], [0.0, 1.0]])

    x, log = solver(x0, x1, B0, f1_list, g1_list, dk_method="inexact")
    print_var("x", x)
    print_var("x0", x0)
    print_var("x1", x1)
    print_var("B0", B0)
    print_var("Solution found", x)
    print("Log:")
    with open("log.txt", "w", encoding="utf-8") as f:
        f.write("k, xk, ||(f+g)||, time (ms) \n")
        for i in range(len(log["xk"])):
            write_log(f, log, i)
            print_log(log, i)
        total_time = sum(log["time"])
        f.write(f"{total_time:.6f}\n")
        print(f"Total time: {total_time:.6f} ms")

    ol = to_overleaf()
    with open("log_ol.txt", "w", encoding="utf-8") as file:
        file.write(ol)
    print("Log file converted to LaTeX format and saved as log_ol.txt")


def find_dk_exact(Bk: NDArray, fgxk: NDArray, sec: NDArray) -> NDArray:
    A = Bk + sec
    dk = np.linalg.solve(A, -fgxk)
    return dk


def find_dk_inexact(
    Bk: NDArray, fgxk: NDArray, sec: NDArray, thk: float = 0.5
) -> NDArray:
    A = Bk + sec
    n = len(fgxk)
    fgxk_norm = np.linalg.norm(fgxk)

    # Função objetivo: minimizar ||dk||²
    def objective(dk):
        return np.dot(dk, dk)

    # Restrição de inequação: ||(Bk + sec)*dk + fgxk|| <= theta*||fgxk||
    def constraint(dk):
        lhs = np.linalg.norm(A @ dk + fgxk)
        rhs = thk * fgxk_norm
        # Para inequação a <= b, retornamos b - a >= 0
        return rhs - lhs

    constraints = {"type": "ineq", "fun": constraint}

    x0 = np.zeros(n)

    result = scipy.optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=constraints,
    )

    if result.success:
        return result.x
    else:
        # Fallback para solução não restrita
        return np.linalg.solve(A, -fgxk)


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


def get_time() -> float:
    return time.time() * 1e3


def solver(
    x0: NDArray,
    x1: NDArray,
    B0: NDArray,
    f_list: list[Callable[[NDArray], NDArray]],
    g_list: list[Callable[[NDArray], NDArray]],
    th: float = 0.5,
    thk: float = 0.05,
    s: float = 0.5,
    tol: float = 1e-4,
    t_min: float = 1 / 3,
    t_max: float = 2 / 3,
    eta: Callable[[int], float] = _eta,
    dk_method: 'Literal["exact", "inexact"]' = "inexact",
):
    if dk_method == "exact":
        find_dk = find_dk_exact
    elif dk_method == "inexact":
        if 0 <= thk <= th:
            find_dk = lambda a, b, c: find_dk_inexact(a, b, c, thk=thk)
        else:
            raise ValueError("thk must be in the range [0, th]")
    else:
        raise ValueError("dk_method must be 'exact' or 'inexact'")

    def f(x: NDArray) -> NDArray:
        return np.array([f_func(x) for f_func in f_list])

    def g(x: NDArray) -> NDArray:
        return np.array([g_func(x) for g_func in g_list])

    def fg(x: NDArray) -> NDArray:
        return f(x) + g(x)

    xk_1 = x0
    xk = x1
    Bk = B0
    fgxk = fg(xk)
    sec = secant(xk_1, xk, g_list)
    dk = find_dk(Bk, fgxk, sec)
    a = 1

    log: dict[str, list] = {"xk": [xk_1], "f+g": [fg(xk_1)], "time": [0.0]}
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


def print_var(name: str, value: NDArray):
    print(f"{name}:\n{np.array2string(value)}")


def flog(log) -> str:
    return np.array2string(log, formatter={"float_kind": number_format})


def write_log(f: TextIO, log, i):
    f.write(
        f"{i}, {flog(log['xk'][i])}, {flog(norm(log['f+g'][i]))}, {log['time'][i]:.6f}\n"
    )


def print_log(log, i):
    print(
        f"x{i:02} = {flog(log['xk'][i])}, ||(f+g){i:02}|| = {flog(norm(log['f+g'][i]))}, time = {log['time'][i]:.6f}"
    )


if __name__ == "__main__":
    main()
