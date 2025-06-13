import re


def main():
    with open("log.txt", "r", encoding="utf-8") as file:
        next(file)
        text = file.read()

    patterns = [
        r"(\d+)\.0{6}",
        r"(\d*\.?\d+), ",
        r"\[\s*(-?\d*\.?\d+)\s*(-?\d*\.?\d+)\],\s*",
        r"(\d*\.?\d+)\n",
        r"(.*\n)",
    ]
    replace_patterns = [
        r"\1",
        r"$\1$ & ",
        r"{ \\scriptsize $\\begin{pmatrix} \1 \\\\ \2 \\end{pmatrix}$ } & ",
        r"$\1$ \\\\ [0.3cm]\n",
        r"\t\t\t\1",
    ]
    for pattern, replace_pattern in zip(patterns, replace_patterns):
        print(f"Replacing pattern: {pattern} with {replace_pattern}")
        text = re.sub(pattern, replace_pattern, text)
    return text


if __name__ == "__main__":
    result = main()
    with open("log_ol.txt", "w", encoding="utf-8") as file:
        file.write(result)
    print("Log file converted to LaTeX format and saved as log_ol.txt")
