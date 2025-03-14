
# Linear Regression from Scratch




## Description
This project demonstrates a Linear Regression model built from scratch using Python and NumPy. It implements the key mathematical components of linear regression, including the hypothesis function, cost function (Mean Squared Error), and gradient descent for parameter optimization. The model is trained on a dataset from Kaggle and evaluates performance using metrics like Mean Squared Error (MSE) and R² Score.
## Features

- Implemented Linear Regression from scratch using Python
- Computed cost function using both iterative and vectorized approaches
- Applied gradient descent to optimize the model
- Visualized the cost function and the gradient descent path

## Project Structure
- ```LinearRegression.ipynb ```– Jupyter notebook containing the linear regression implementation
- ```advertising.csv``` – Dataset

## Dependencies
Ensure that the following Python libraries are installed:

- ```pandas```
- ```matplotlib```
- ```numpy```
- ```scienceplots```
- ```seaborn```
- ```plotly```
Install dependencies using:

```
 pip install pandas matplotlib numpy scienceplots seaborn plotly
 ```
## Dataset
- ```Dataset Name```: Advertising Dataset
- ```Description```: This dataset contains information on advertising expenditure (TV, Radio, and Newspaper) and the corresponding sales figures.
- ```Source```: Public dataset from Kaggle (available in /advertising.csv)
## Dataset
- ```Dataset Name```: Advertising Dataset
- ```Description```: This dataset contains information on advertising expenditure (TV, Radio, and Newspaper) and the corresponding sales figures.
- ```Source```: Public dataset from Kaggle (available in /advertising.csv)
## Computation
The core steps in linear regression computation include:

**1.Hypothesis Function**
The model predicts values using:


```h(x)=w⋅x+b```
where 
w=weight, 
b=bias, and 
x=input feature.

**2. Cost Function**
The cost function measures the average error between the predicted and actual values.
Two approaches are implemented:

**Iterative Approach**:
Calculates the cost using a loop:
python
```
def cost_function(x, y, w, b):
    total_cost = 0
    n = len(y)
    for i in range(n):
        total_cost += (y[i] - (w*x[i] + b))**2
    return total_cost / float(n)
```

**Vectorized Approach**: 
More efficient using NumPy:
python

```
def vectorised_cost_function(x, y, w, b):
    total_cost = np.sum((y - (w*x + b))**2) / (2*len(y))
    return total_cost
 ```   

## Cost Function Formula
The cost function is defined as:


$$
J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b))^2
$$


where:  
- w is the weight (slope)  
- b  is the bias (intercept)  
-  $$x_i , y_i$$  are the feature and target values  
-  n is the number of samples  

## Gradient Descent
Gradient Descent minimizes the cost function by updating the weight and bias values iteratively:

$$
w = w - \alpha \cdot \frac{\partial J}{\partial w}
$$

$$
b = b - \alpha \cdot \frac{\partial J}{\partial b}
$$

where:  
- <p> α is the learning rate </p>



### Gradient Descent function:
```python
def gradient_descent(x, y, w, b, a, iterations):
    w_history = []
    b_history = []
    cost_history = []

    for i in range(iterations):
        dw = -(2/n) * np.sum((y - (w*x + b)) * x)
        db = -(2/n) * np.sum(y - (w*x + b))

        w_history.append(w)
        b_history.append(b)
        cost_history.append(vectorised_cost_function(x, y, w, b))

        w = w - a * dw
        b = b - a * db

    return w, b, w_history, b_history, cost_history
```
## Visualization

To better understand how the model performs, the following visualizations are included:

- ```Cost vs Weight``` – A plot of the cost function against a range of weight values while keeping the bias constant.

- ``` 3D Cost Surface``` – A 3D surface plot showing the cost function relative to both weight and bias.
-  ```Contour Plot``` – A contour plot visualizing the cost function with the gradient descent path overlaid
## Results

After training the model, the performance is evaluated using:

- ```Mean Squared Error (MSE) ```– Measures the average of the squares of the errors.
- ```R² Score``` – Measures how well the model explains the variance in the target variable.
