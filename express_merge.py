import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def merge_expression_to_final(expression, data):
    """
    归并表达式：将能计算的统计函数替换为实际数值，并将所有中间变量替换到最终公式中
    """
    logging.info("=== 开始归并表达式到最终公式 ===")
    logging.info(f"原始表达式: {expression}")

    # 按分号分割步骤
    steps = [step.strip() for step in expression.split(';') if step.strip()]

    # 存储变量定义
    var_definitions = {}

    # 处理所有步骤（除了最后一步）
    for step in steps[:-1]:
        if '=' in step:
            var_name, expr = [part.strip() for part in step.split('=', 1)]
            logging.info(f"处理变量定义: {var_name} = {expr}")

            # 替换统计函数
            processed_expr = expr
            # 定义所有统计函数映射
            stat_functions = {
                'MEAN': lambda col: data[col].mean(),
                'MEDIAN': lambda col: data[col].median(),
                'STD': lambda col: data[col].std(),
                'VAR': lambda col: data[col].var(),
                'MIN': lambda col: data[col].min(),
                'MAX': lambda col: data[col].max(),
                'SUM': lambda col: data[col].sum(),
                'COUNT': lambda col: len(data[col].dropna()),
                'Q1': lambda col: data[col].quantile(0.25),
                'Q3': lambda col: data[col].quantile(0.75),
                'SKEW': lambda col: data[col].skew(),
                'KURTOSIS': lambda col: data[col].kurtosis()
            }

            # 替换所有统计函数
            for func_name, stat_func in stat_functions.items():
                if f'{func_name}(' in processed_expr:
                    processed_expr = replace_stat_function(processed_expr, func_name, data, stat_func)

            # 特殊处理 QUANTILE 函数
            if 'QUANTILE(' in processed_expr:
                processed_expr = replace_quantile_function(processed_expr, data)

            # 处理协方差和相关系数
            if 'COV(' in processed_expr:
                processed_expr = replace_cov_function(processed_expr, data)
            if 'CORR(' in processed_expr:
                processed_expr = replace_corr_function(processed_expr, data)

            var_definitions[var_name] = processed_expr
            logging.info(f"归并后: {var_name} = {processed_expr}")

    # 处理最后一步（最终公式）
    final_step = steps[-1]
    if '=' in final_step:
        # 提取最终表达式（去掉变量名）
        final_expression = final_step.split('=', 1)[1].strip()
    else:
        final_expression = final_step

    logging.info(f"初始最终表达式: {final_expression}")

    # 递归替换所有中间变量
    def replace_variables(expr, definitions, depth=0):
        if depth > 10:  # 防止无限递归
            return expr

        for var_name, var_expr in definitions.items():
            if var_name in expr:
                # 确保替换的是完整的变量名，不是子字符串
                pattern = r'\b' + re.escape(var_name) + r'\b'
                expr = re.sub(pattern, f'({var_expr})', expr)
                logging.info(f"替换 {var_name} -> ({var_expr})")
                # 递归替换新表达式中的变量
                expr = replace_variables(expr, definitions, depth + 1)
        return expr

    # 替换最终表达式中的所有变量
    final_expression = replace_variables(final_expression, var_definitions)

    # 关键修复：确保最终表达式中的统计函数也被替换
    # 重新处理最终表达式中的统计函数
    stat_functions = {
        'MEAN': lambda col: data[col].mean(),
        'MEDIAN': lambda col: data[col].median(),
        'STD': lambda col: data[col].std(),
        'VAR': lambda col: data[col].var(),
        'MIN': lambda col: data[col].min(),
        'MAX': lambda col: data[col].max(),
        'SUM': lambda col: data[col].sum(),
        'COUNT': lambda col: len(data[col].dropna()),
        'Q1': lambda col: data[col].quantile(0.25),
        'Q3': lambda col: data[col].quantile(0.75),
        'SKEW': lambda col: data[col].skew(),
        'KURTOSIS': lambda col: data[col].kurtosis()
    }

    for func_name, stat_func in stat_functions.items():
        if f'{func_name}(' in final_expression:
            final_expression = replace_stat_function(final_expression, func_name, data, stat_func)

    # 特殊处理 QUANTILE 函数
    if 'QUANTILE(' in final_expression:
        final_expression = replace_quantile_function(final_expression, data)

    # 处理协方差和相关系数
    if 'COV(' in final_expression:
        final_expression = replace_cov_function(final_expression, data)
    if 'CORR(' in final_expression:
        final_expression = replace_corr_function(final_expression, data)

    # 替换数学函数
    final_expression = final_expression.replace('SQRT', 'math.sqrt')

    # 对于多目标优化，我们需要确保返回的是单一表达式，而不是赋值语句
    # 如果最终表达式仍然包含赋值，我们提取等号右边的部分
    if '=' in final_expression:
        final_expression = final_expression.split('=', 1)[1].strip()

    logging.info(f"最终纯公式: {final_expression}")

    # 构建上下文
    context = {'math': math}

    return final_expression, context

def objective_function(weights, final_expression, context):
    """
    目标函数：直接计算最终纯公式
    """
    variables = {f'w{i}': weight for i, weight in enumerate(weights)}
    full_context = {**context, **variables, 'math': math}

    try:
        result = eval(final_expression, full_context)
        return result
    except Exception as e:
        logging.error(f"目标函数计算错误: {e}")
        return float('inf')


def replace_quantile_function(expression, data):
    """替换分位数函数"""
    import re
    pattern = r'QUANTILE\(([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, expression)

    for col, q in matches:
        if col in data.columns:
            try:
                q_value = float(q)
                quantile_value = data[col].quantile(q_value)
                expression = expression.replace(f'QUANTILE({col}, {q})', f'{quantile_value:.8f}')
                logging.info(f"  QUANTILE({col}, {q}) = {quantile_value:.8f}")
            except ValueError:
                logging.warning(f"分位数参数错误: {q}")
        else:
            logging.warning(f"列 '{col}' 不存在")
    return expression


def replace_corr_function(expression, data):
    """替换相关系数函数"""
    import re
    pattern = r'CORR\(([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, expression)

    for col1, col2 in matches:
        if col1 in data.columns and col2 in data.columns:
            corr_value = data[col1].corr(data[col2])
            expression = expression.replace(f'CORR({col1}, {col2})', f'{corr_value:.8f}')
            logging.info(f"  CORR({col1}, {col2}) = {corr_value:.8f}")
        else:
            logging.warning(f"列 '{col1}' 或 '{col2}' 不存在")
    return expression

def replace_stat_function(expression, func_name, data, stat_func):
    """替换统计函数为实际值"""
    pattern = f'{func_name}\\(([^)]+)\\)'
    matches = re.findall(pattern, expression)

    for col in matches:
        if col in data.columns:
            value = stat_func(col)
            expression = expression.replace(f'{func_name}({col})', f'{value:.8f}')
            logging.info(f"  {func_name}({col}) = {value:.8f}")
        else:
            logging.warning(f"列 '{col}' 不存在")
    return expression


def replace_cov_function(expression, data):
    """替换协方差函数为实际值"""
    pattern = r'COV\(([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, expression)

    for col1, col2 in matches:
        if col1 in data.columns and col2 in data.columns:
            cov_value = data[col1].cov(data[col2])
            expression = expression.replace(f'COV({col1}, {col2})', f'{cov_value:.8f}')
            logging.info(f"  COV({col1}, {col2}) = {cov_value:.8f}")
        else:
            logging.warning(f"列 '{col1}' 或 '{col2}' 不存在")
    return expression


def objective_function(weights, final_expression, context):
    """
    目标函数：执行完整的归并表达式，返回最终结果
    """
    # 创建变量上下文
    variables = {f'w{i}': weight for i, weight in enumerate(weights)}
    full_context = {**context, **variables, 'math': math}

    try:
        # 执行完整的归并表达式
        exec(final_expression, full_context)

        # 提取最终结果 - 找到最后一个赋值语句的变量名
        steps = [step.strip() for step in final_expression.split(';') if step.strip()]
        last_step = steps[-1]

        if '=' in last_step:
            # 最后一个步骤是赋值语句，提取变量名
            final_var = last_step.split('=')[0].strip()
            result = full_context.get(final_var)
        else:
            # 最后一个步骤是表达式，直接求值
            result = eval(last_step, full_context)

        return result
    except Exception as e:
        logging.error(f"目标函数计算错误: {e}")
        return float('inf')
def run_optimization(config, data):
    logging.info("=== 开始优化流程 ===")
    original_expression = config['objectives'][0]['expression']
    final_expression, context = merge_expression_to_final(original_expression, data)
    logging.info(f"#完整归并表达式: {final_expression}")

    # 2. 准备优化参数
    n_variables = len(config['variables'])
    x0 = [1.0 / n_variables] * n_variables  # 均匀初始权重
    bounds = [(var['bounds'][0], var['bounds'][1]) for var in config['variables']]

    constraints = []
    for constraint in config.get('constraints', []):
        if constraint['type'] == 'eq':
            constraints.append({         'type': 'eq', 'fun': lambda x, expr=constraint['expression']: eval_constraint(x, expr) })
        elif constraint['type'] == 'ineq':
            constraints.append({         'type': 'ineq', 'fun': lambda x, expr=constraint['expression']: eval_constraint(x, expr) })


    objective_type = config['objectives'][0]['type']
    sign = -1 if objective_type == 'maximize' else 1

    # 5. 包装目标函数
    def wrapped_objective(x):
        result = objective_function(x, final_expression, context)
        return sign * result
    # 6. 运行优化
    try:
        result = minimize(
            wrapped_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        logging.info(f"优化结果: {result.success}")
        logging.info(f"最优权重: {result.x}")
        logging.info(f"目标函数值: {result.fun}")
        logging.info(f"迭代次数: {result.nit}")

        return {
            'success': result.success,
            'optimal_weights': result.x,
            'optimal_value': -result.fun if objective_type == 'maximize' else result.fun,
            'iterations': result.nit,
            'final_expression': final_expression
        }

    except Exception as e:
        logging.error(f"优化失败: {e}")
        return None
def eval_constraint(weights, expression):
    """评估约束条件"""
    variables = {f'w{i}': weight for i, weight in enumerate(weights)}
    try:
        return eval(expression, {'math': math, **variables})
    except Exception as e:
        logging.error(f"约束评估错误: {e}")
        return 0
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    n_days = 100
    returns_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_days),
        'return_A': np.random.normal(0.001, 0.02, n_days),
        'return_B': np.random.normal(0.0012, 0.025, n_days),
        'return_C': np.random.normal(0.0008, 0.018, n_days)
    })
    config = {
        "optimizationMode": "direct",
        "surrogateModel": "noModel",
        "optimizer": "scipy",
        "objectiveType": "single",
        "variables": [
            {"name": "w0", "type": "continuous", "bounds": [0, 1]},
            {"name": "w1", "type": "continuous", "bounds": [0, 1]},
            {"name": "w2", "type": "continuous", "bounds": [0, 1]}
        ],
        "objectives": [
            {   "name": "sharpe_ratio",
                "type": "maximize",
                "expression": "mean_return = MEAN(return_A)*w0 + MEAN(return_B)*w1 + MEAN(return_C)*w2; variance = COV(return_A, return_A)*w0*w0 + COV(return_A, return_B)*w0*w1 + COV(return_A, return_C)*w0*w2 + COV(return_B, return_A)*w1*w0 + COV(return_B, return_B)*w1*w1 + COV(return_B, return_C)*w1*w2 + COV(return_C, return_A)*w2*w0 + COV(return_C, return_B)*w2*w1 + COV(return_C, return_C)*w2*w2; sharpe_ratio = (mean_return - 0.0001) / math.sqrt(variance)"
            }
        ],
        "constraints": [
            {"expression": "w0 + w1 + w2 - 1", "type": "eq"}
        ]
    }
    logging.info(f"数据统计:")
    logging.info(returns_data[['return_A', 'return_B', 'return_C']].describe())
    result = run_optimization(config, returns_data)
    if result:
        logging.info(f"\n=== 优化结果 ===")
        logging.info(f"成功: {result['success']}")
        logging.info(f"最优权重: {result['optimal_weights']}")
        logging.info(f"最优夏普比率: {result['optimal_value']:.6f}")
        logging.info(f"最终公式: {result['final_expression']}")