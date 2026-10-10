import re
from math import comb


def get_nested(config, key: str, default=None):
    keys = key.split(".")
    value = config
    for k in keys:
        if k in value:
            value = value[k]
        else:
            return default
    return value


def pass_at_k(results, k, n):
    """Compute pass@k from a list of per-problem attempt lists.

    Args:
        results: list of lists, each inner list is booleans for attempts
        k: number of attempts
        n: total samples per problem
    """
    assert k > 0
    total_problems = len(results)
    if total_problems == 0:
        return 0.0

    total_pass_probability = 0.0
    for attempts in results:
        c = sum(attempts)
        if c == 0:
            pass_probability = 0.0
        elif n <= k:
            pass_probability = 1.0
        else:
            failure_combinations = comb(n - c, k) if n - c >= k else 0
            total_combinations = comb(n, k)
            pass_probability = 1 - (failure_combinations / total_combinations)
        total_pass_probability += pass_probability

    return total_pass_probability / total_problems


def pass_at_k_simple(k: int, c: int, n: int) -> float:
    """Compute pass@k for a single problem.

    Args:
        k: number of attempts
        c: number of correct samples
        n: total samples
    """
    import math
    if n - c < k:
        return 1.0
    return 1.0 - math.prod([1.0 - k / i for i in range(n - c + 1, n + 1)])


def extract_code(resp):
    """Extract the first code block from a markdown-formatted response."""
    msg_lines = resp.splitlines()
    i_line = 0
    while i_line < len(msg_lines):
        line = msg_lines[i_line]
        if line.startswith("```"):
            code_lines = []
            i_line += 1
            while i_line < len(msg_lines):
                line = msg_lines[i_line]
                if line.startswith("```"):
                    break
                code_lines.append(line)
                i_line += 1
            return "\n".join(code_lines)
        i_line += 1
    return ""


def esc(name):
    return name.replace("/", "@")


def extract_int(text):
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def extract_ints(text):
    return [int(match) for match in re.findall(r"\d+", text)]
