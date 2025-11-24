import re
from collections import defaultdict
from typing import List, Tuple, Dict


def parse_complexity(complexity: str) -> Tuple[Dict[str, int], List[str]]:
    """
    Parse a Big O complexity string.

    Returns:
        Tuple of (variable_exponents dict, log_terms list)
        where log_terms contains the variables inside log (e.g., ['n', 'L'])
    """
    match = re.search(r'O\((.*?)\)', complexity)
    if not match:
        return {}, []

    content = match.group(1).strip()

    log_terms = []
    log_matches = re.findall(r'\blog\s+(\w+)', content, flags=re.IGNORECASE)
    log_terms.extend(log_matches)

    standalone_logs = re.findall(r'\blog(?!\s+\w)', content, flags=re.IGNORECASE)
    log_terms.extend(['n'] * len(standalone_logs))

    content_no_log = re.sub(r'\blog\s*\w*', '', content, flags=re.IGNORECASE)

    exponents = defaultdict(int)
    tokens = re.findall(r'([a-zA-Z_]\w*)(?:\^(\d+))?', content_no_log)

    for var, exp in tokens:
        if var and var.lower() != 'log':
            exponent = int(exp) if exp else 1
            exponents[var] = max(exponents[var], exponent)

    return dict(exponents), log_terms


def compare_complexities(c1: Tuple[Dict[str, int], List[str]],
                         c2: Tuple[Dict[str, int], List[str]]) -> int:
    """
    Compare two complexities.
    Returns: 1 if c1 > c2, -1 if c1 < c2, 0 if equal
    """
    exp1, log1 = c1
    exp2, log2 = c2

    all_vars = set(exp1.keys()) | set(exp2.keys())

    for var in sorted(all_vars):
        e1 = exp1.get(var, 0)
        e2 = exp2.get(var, 0)
        if e1 > e2:
            return 1
        elif e1 < e2:
            return -1

    if len(log1) > len(log2):
        return 1
    elif len(log1) < len(log2):
        return -1

    return 0


def combine_big_o(complexities: List[str]) -> str:
    """
    Combines multiple Big O complexity strings by taking the dominant complexity.

    Args:
        complexities: List of Big O notation strings

    Returns:
        Combined Big O notation representing the dominant complexity

    Examples:
        >>> combine_big_o(["O(n^2)", "O(n log n)", "O(n^3)"])
        'O(n^3)'
        >>> combine_big_o(["O(n)", "O(n log n)"])
        'O(n log n)'
        >>> combine_big_o(["O(L^2 d^2)", "O(L d^2 k)", "O(L^2 d^2)"])
        'O(L^2 d^2 k)'
    """
    if not complexities:
        return "O(1)"

    parsed = [parse_complexity(c) for c in complexities]

    # Find the dominant complexity
    dominant = parsed[0]
    dominant_str = complexities[0]

    for i in range(1, len(parsed)):
        if compare_complexities(parsed[i], dominant) > 0:
            dominant = parsed[i]
            dominant_str = complexities[i]

    # Find the maximum exponents for all variables
    max_exponents = defaultdict(int)
    for exponents, log_terms in parsed:
        for var, exp in exponents.items():
            max_exponents[var] = max(max_exponents[var], exp)

    # Collect all log terms from complexities with maximum polynomial degree
    all_log_terms = []
    for exponents, log_terms in parsed:
        has_max_degree = True

        # Check if this has any variable at less than max exponent
        for var, exp in exponents.items():
            if exp < max_exponents[var]:
                has_max_degree = False
                break

        # If it has max degree for its variables, keep its log terms
        if has_max_degree:
            all_log_terms.extend(log_terms)

    if not max_exponents and not all_log_terms:
        return "O(1)"

    # Build the result string
    terms = []

    # Add variable terms
    for var in sorted(max_exponents.keys()):
        exp = max_exponents[var]
        if exp == 1:
            terms.append(var)
        else:
            terms.append(f"{var}^{exp}")

    # Add log terms if applicable
    if all_log_terms:
        log_str = " ".join([f"log {var}" for var in sorted(set(all_log_terms))])
        if terms:
            terms.append(log_str)
        else:
            return f"O({log_str})"

    return f"O({' '.join(terms)})"


# Example usage
if __name__ == "__main__":
    # Test cases
    test1 = ["O(L^2 d^2)", "O(L d^2 k)", "O(L^2 d^2)"]
    print(f"Input: {test1}")
    print(f"Output: {combine_big_o(test1)}")
    print()

    test2 = ["O(n^2)", "O(n log n)", "O(n^3)"]
    print(f"Input: {test2}")
    print(f"Output: {combine_big_o(test2)}")
    print()

    test3 = ["O(n)", "O(n log n)"]
    print(f"Input: {test3}")
    print(f"Output: {combine_big_o(test3)}")
    print()

    test4 = ["O(n)", "O(m)", "O(k^2)"]
    print(f"Input: {test4}")
    print(f"Output: {combine_big_o(test4)}")
    print()

    test5 = ["O(L^1)", "O(L log L)"]
    print(f"Input: {test5}")
    print(f"Output: {combine_big_o(test5)}")