import re


def to_overleaf():
    with open("log.txt", "r", encoding="utf-8") as file:
        next(file)
        text = file.read()

    patterns = [
        r"(\d)\.0{6}",
        r"(\d\.\d*?)0+\b",
        r"^(\d+\.\d+)$",
        r"(\d*\.?\d+), ",
        r"(\d*\.?\d+)\n",
        r"\[\s*(-?\d*\.?\d+)\s*(-?\d*\.?\d+)\],\s*",
        r"(.*)\n",
    ]
    replace_patterns = [
        r"\1",
        r"\1",
        r"& & & Total: $\1$",
        r"$\1$ & ",
        r"$\1$\n",
        r"{ \\scriptsize $\\begin{pmatrix} \1 \\\\ \2 \\end{pmatrix}$ } & ",
        r"\t\t\t\1 \\\\ [0.3cm]\n",
    ]
    for pattern, replace_pattern in zip(patterns, replace_patterns):
        # print(f"Replacing pattern: {pattern} with {replace_pattern}")
        text = re.sub(pattern, replace_pattern, text, flags=re.MULTILINE)
    return text


def main():
    ol = to_overleaf()
    with open("log_ol.txt", "w", encoding="utf-8") as file:
        file.write(ol)
    print("Log file converted to LaTeX format and saved as log_ol.txt")


if __name__ == "__main__":
    result = main()
