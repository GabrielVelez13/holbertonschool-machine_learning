### README for Plotting

## Overview
This project contains various scripts and functions for plotting data using Python. The primary goal is to provide tools for visualizing data in different formats.

## Requirements
- Python 3.12
- `matplotlib`
- `numpy`

### Basic Plot
To create a basic plot, use the `basic_plot.py` script:
```sh
python basic_plot.py
```

### Scatter Plot
To create a scatter plot, use the `scatter_plot.py` script:
```sh
python scatter_plot.py
```

### Bar Chart
To create a bar chart, use the `bar_chart.py` script:
```sh
python bar_chart.py
```

## Examples
### Basic Plot Example
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Basic Plot Example')
plt.show()
```

### Scatter Plot Example
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')
plt.show()
```

### Bar Chart Example
```python
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D']
values = [4, 7, 1, 8]

plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()
```
