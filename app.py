from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import sympy as sp
import pandas as pd
from matplotlib.figure import Figure
import re

app = Flask(__name__)

def parse_function(func_str):
    """Parse a mathematical function string into a callable function"""
    # Replace trigonometric functions without arguments
    func_str = re.sub(r'sin(\d+)x', r'sin(\1*x)', func_str)
    func_str = re.sub(r'cos(\d+)x', r'cos(\1*x)', func_str)
    func_str = re.sub(r'tan(\d+)x', r'tan(\1*x)', func_str)
    
    # Replace implicit multiplication
    func_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', func_str)
    
    # Replace ^ with **
    func_str = func_str.replace('^', '**')
    
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(func_str)
        return sp.lambdify(x, expr, modules=['numpy'])
    except Exception as e:
        print(f"Error parsing function: {e}")
        return None

def generate_points(func, a, b, n):
    """Generate uniformly spaced points for a function"""
    x = np.linspace(float(a), float(b), n)
    try:
        y = func(x)
        return x, y
    except Exception as e:
        print(f"Error generating points: {e}")
        return None, None

def system_interpolation(x_list, y_list):
    """Interpolation using system of linear equations"""
    n = len(x_list)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = x_list[i]**j
    try:
        coeffs = np.linalg.solve(matrix, y_list)
        return lambda x: np.polyval(coeffs[::-1], x)
    except np.linalg.LinAlgError:
        print("Matrix is singular or poorly conditioned")
        return None

def lagrange_interpolation(x_list, y_list):
    """Lagrange interpolation method"""
    def L(k, x):
        result = np.ones_like(x)
        for i in range(len(x_list)):
            if i != k:
                result *= (x - x_list[i]) / (x_list[k] - x_list[i])
        return result

    def p(x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for k in range(len(x_list)):
            result += y_list[k] * L(k, x)
        return result

    return p

def parametric_interpolation(t_list, x_list, y_list):
    """Parametric interpolation method"""
    try:
        x_interpolator = lagrange_interpolation(t_list, x_list)
        y_interpolator = lagrange_interpolation(t_list, y_list)
        return lambda t: (x_interpolator(t), y_interpolator(t))
    except Exception as e:
        print(f"Error in parametric interpolation: {e}")
        return None

def create_plot(x_data, y_data, methods_results, title="Interpolation Results"):
    """Create a plot with original data and interpolation results"""
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot original data points
    ax.plot(x_data, y_data, 'ko', label='Data points', markersize=4)
    
    # Plot interpolation results
    linestyles = ['-', '--', '-.']
    colors = ['black','red', 'blue']
    for (method_name, x_interp, y_interp), ls, color in zip(methods_results, linestyles, colors):
        ax.plot(x_interp, y_interp, color=color, linestyle=ls, 
                label=f'{method_name}', alpha=0.7, linewidth=1)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    
    # Adjust layout and style
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Save plot to memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interpolate', methods=['POST'])
def interpolate():
    data = request.json
    methods = data.get('methods', [])
    
    if not methods:
        return jsonify({'error': 'No interpolation methods selected'})
    
    methods_results = []
    
    if 'file' in data:  # File upload case
        try:
            df = pd.read_csv(io.StringIO(data['file']))
            x_points = df['x'].values
            y_points = df['y'].values
            n = len(x_points)
            x_eval = np.linspace(min(x_points), max(x_points), 200)
            
            for method in methods:
                if method == 'parametric':
                    t_points = np.linspace(0, 1, n)
                    param_func = parametric_interpolation(t_points, x_points, y_points)
                    if param_func:
                        t_eval = np.linspace(0, 1, 200)
                        x_interp, y_interp = param_func(t_eval)
                        methods_results.append(('Parametric', x_interp, y_interp))
                elif method == 'sle':
                    sle_func = system_interpolation(x_points, y_points)
                    if sle_func:
                        y_interp = sle_func(x_eval)
                        methods_results.append(('SLE', x_eval, y_interp))
                elif method == 'lagrange':
                    lagrange_func = lagrange_interpolation(x_points, y_points)
                    if lagrange_func:
                        y_interp = lagrange_func(x_eval)
                        methods_results.append(('Lagrange', x_eval, y_interp))
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})
            
    else:  # Function input case
        try:
            func_str = data['function']
            a, b = float(data['a']), float(data['b'])
            degree = int(data['degree'])
            
            func = parse_function(func_str)
            if not func:
                return jsonify({'error': 'Invalid function'})
                
            x_points, y_points = generate_points(func, a, b, degree + 1)
            if x_points is None:
                return jsonify({'error': 'Error generating points'})
                
            x_eval = np.linspace(a, b, 200)
            y_eval = func(x_eval)
            
            # Add original function
            methods_results.append(('Original', x_eval, y_eval))
            
            for method in methods:
                if method == 'sle':
                    sle_func = system_interpolation(x_points, y_points)
                    if sle_func:
                        y_interp = sle_func(x_eval)
                        methods_results.append(('SLE', x_eval, y_interp))
                elif method == 'lagrange':
                    lagrange_func = lagrange_interpolation(x_points, y_points)
                    if lagrange_func:
                        y_interp = lagrange_func(x_eval)
                        methods_results.append(('Lagrange', x_eval, y_interp))
                        
        except Exception as e:
            return jsonify({'error': f'Error in interpolation: {str(e)}'})
    
    if not methods_results:
        return jsonify({'error': 'No valid interpolation results'})
    
    plot_data = create_plot(x_points, y_points, methods_results)
    return jsonify({'plot': plot_data})

if __name__ == '__main__':
    app.run(debug=True)