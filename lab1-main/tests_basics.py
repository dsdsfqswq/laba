import sys
from python_basics import (
    count_vowels, has_unique_characters, count_one_bits, 
    multiplicative_persistence, mse, prime_factorization, 
    pyramid, is_balanced_number
)

class MockInput:
    def __init__(self, value): 
        self.value = value

class MockVectorInput:
    def __init__(self, predicted, expected):
        self.predicted = predicted
        self.expected = expected

def run_tests():
    assert count_vowels(MockInput("hello")) == 2
    assert count_vowels(MockInput("Привет")) == 2
    assert count_vowels(MockInput("bcdfghjkl")) == 0
    assert count_vowels(MockInput("")) == 0
    assert count_vowels(MockInput("АЕЁИОУЫЭЮЯaeiouy")) == 16

    assert has_unique_characters(MockInput("abcdef")) is True
    assert has_unique_characters(MockInput("abcdea")) is False
    assert has_unique_characters(MockInput("")) is True
    assert has_unique_characters(MockInput("aA")) is True

    assert count_one_bits(MockInput(1)) == 1
    assert count_one_bits(MockInput(7)) == 3
    assert count_one_bits(MockInput(8)) == 1
    assert count_one_bits(MockInput(1023)) == 10

    assert multiplicative_persistence(MockInput(39)) == 3
    assert multiplicative_persistence(MockInput(999)) == 4
    assert multiplicative_persistence(MockInput(4)) == 0

    assert mse(MockVectorInput([1, 2], [1, 2])) == 0.0
    assert mse(MockVectorInput([2, 4], [4, 2])) == 4.0
    assert mse(MockVectorInput([], [])) == 0.0
    assert mse(MockVectorInput([10, 20], [12, 15])) == 14.5


    assert prime_factorization(MockInput(1)) == "1"
    assert prime_factorization(MockInput(2)) == "2"
    assert prime_factorization(MockInput(12)) == "2^2 * 3"
    assert prime_factorization(MockInput(100)) == "2^2 * 5^2"
    assert prime_factorization(MockInput(13)) == "13"

    assert pyramid(MockInput(1)) == 1
    assert pyramid(MockInput(3)) == 2
    assert pyramid(MockInput(5)) == 2
    assert pyramid(MockInput(6)) == 3

    assert is_balanced_number(MockInput(12321)) is True
    assert is_balanced_number(MockInput(4321)) is False
    assert is_balanced_number(MockInput(7)) is True
    assert is_balanced_number(MockInput(11)) is True
    
    print("tests passed")

if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError:
        print("tests failed")
        sys.exit(1)
