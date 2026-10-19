"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouyAEIOUYаеёиоуыэюяАЕЁИОУЫЭЮЯ")
    vowels_count = 0
    for char in text:
        if char in vowels:
            vowels_count += 1           
    return vowels_count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1  
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if not predicted:
        return 0.0
    squared_errors_sum = 0.0
    n = len(predicted)
    for i in range(n):
        p = predicted[i]
        e = expected[i]
        error = p - e
        squared_error = error ** 2
        squared_errors_sum += squared_error
    return squared_errors_sum / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number == 1:
        return "1"
    factors = []
    d = 2
    n = number
    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                count += 1
                n //= d
            factors.append(f"{d}^{count}" if count > 1 else f"{d}")
        d += 1
    if n > 1:
        factors.append(f"{n}")
    return " * ".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    if cube_count < 1:
        return "Error"
    level = 0
    used_cubes = 0
    while True:
        next_level = level + 1
        if used_cubes + next_level <= cube_count:
            level = next_level
            used_cubes += level
        else:
            break
            
    return level


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        left_sum = sum(int(c) for c in s[:mid])
        right_sum = sum(int(c) for c in s[mid:])
    else:
        left_sum = sum(int(c) for c in s[:mid])
        right_sum = sum(int(c) for c in s[mid + 1:])
    return left_sum == right_sum