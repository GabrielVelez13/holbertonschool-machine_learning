### README for Summation, Pi Notation, Derivatives, and Integrals

## Table of Contents
1. [Summation](#summation)
2. [Pi Notation](#pi-notation)
3. [Derivatives](#derivatives)
4. [Integrals](#integrals)

## Summation
Summation is the addition of a sequence of numbers. The result is their sum or total.

### Example
```python
def summation(n):
    return sum(range(1, n + 1))

# Example usage
print(summation(5))  # Output: 15
```

## Pi Notation
Pi notation (Π) is the product of a sequence of terms.

### Example
```python
def pi_notation(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Example usage
print(pi_notation(5))  # Output: 120
```

## Derivatives
A derivative represents the rate at which a function is changing at any given point.

### Example
```python
def poly_derivative(poly):
    if not isinstance(poly, list):
        return None
    if len(poly) == 0 or len(poly) == 1:
        return None
    return [i * num for i, num in enumerate(poly)][1:]

# Example usage
print(poly_derivative([3, 2, 1]))  # Output: [2, 2]
```

## Integrals
An integral represents the area under a curve defined by a function.

### Example
```python
def poly_integral(poly, C=0):
    if not isinstance(poly, list) or C is None:
        return None
    if len(poly) == 0:
        return None
    if poly == [0]:
        return [C]
    integral = []
    for i, num in enumerate(poly):
        coe = num / (i + 1)
        if coe % 1 == 0:
            integral.append(int(coe))
        else:
            integral.append(coe)
    integral.insert(0, C)
    return integral

# Example usage
print(poly_integral([3, 2, 1]))  # Output: [0, 3.0, 1.0, 0.3333333333333333]
```

This README provides basic examples and explanations for summation, pi notation, derivatives, and integrals in Python.