import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建保存目录
BASE_DIR = Path("optimization_resources")
CONFIG_FILE = BASE_DIR / "all_optimization_configs.json"
DATA_DIR = BASE_DIR / "datasets"

BASE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# 配置名称映射（保持你喜欢的命名方案）
config_names = {
    "portfolio_optimization": "投资组合优化（夏普比率最大化）",
    "engineering_design": "工程设计优化（成本最小化）",
    "rosenbrock": "Rosenbrock函数优化（单目标）",
    "unit_circle_max": "单位圆内最大化x+y（带约束）",
    "surrogate_model": "代理模型拟合精度测试",
    "zdt1": "ZDT1多目标优化",
    "factory_scheduling": "工厂排班优化",
    "production_scheduling": "生产调度优化",
    "supply_chain": "供应链生产分配优化",
    "risk_parity": "风险平价投资组合优化",
    "mean_variance": "均值-方差投资组合优化",
    "target_volatility": "目标波动率投资组合优化",
    "optimizer_comparison": "优化算法性能对比",
    "historical_production": "历史生产数据排程优化",
    "inverse_continuous": "逆向优化-工艺参数（连续变量）",
    "inverse_discrete": "逆向优化-供应链设计（离散变量）",
    "ant_colony": "蚁群算法-物流路径规划",
    "nsga2_production": "NSGA2-生产质量效率多目标优化",
    "nsga2_maintenance": "NSGA2-设备维护多目标优化",
    "nsga2_inventory": "NSGA2-供应链库存多目标优化",
    "nsga2_quality": "NSGA2-质量工艺参数多目标优化",
    "nsga2_simple": "NSGA2-ZDT1简单多目标优化"
}

# 主配置字典（整合所有配置）
all_configs = {
    "metadata": {
        "version": "1.0",
        "created_at": pd.Timestamp.now().isoformat(),
        "total_configs": len(config_names)
    },
    "configs": {}
}


def save_dataset(dataset_name, data_df):
    """保存数据集到CSV文件"""
    if data_df.empty:
        return None

    data_file = DATA_DIR / f"{dataset_name}.csv"
    data_df.to_csv(data_file, index=False, encoding='utf-8')
    logging.info(f"📊 数据集已保存: {data_file.name}")
    return data_file.name


def generate_all_resources():
    """生成所有优化配置和数据集"""

    # 1. 投资组合优化
    portfolio_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'return_A': np.random.normal(0.001, 0.02, 100),
        'return_B': np.random.normal(0.0012, 0.025, 100),
        'return_C': np.random.normal(0.0008, 0.018, 100)
    })
    dataset_file = save_dataset("portfolio_optimization", portfolio_data)

    all_configs["configs"]["portfolio_optimization"] = {
        "title": config_names["portfolio_optimization"],
        "description": "基于夏普比率最大化的投资组合优化，考虑资产收益率的均值和协方差",
        "dataset": dataset_file,
        "type": "portfolio",
        "content": {
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
                {
                    "name": "sharpe_ratio",
                    "type": "maximize",
                    "expression": "mean_return = MEAN(return_A)*w0 + MEAN(return_B)*w1 + MEAN(return_C)*w2; variance = COV(return_A, return_A)*w0*w0 + COV(return_A, return_B)*w0*w1 + COV(return_A, return_C)*w0*w2 + COV(return_B, return_A)*w1*w0 + COV(return_B, return_B)*w1*w1 + COV(return_B, return_C)*w1*w2 + COV(return_C, return_A)*w2*w0 + COV(return_C, return_B)*w2*w1 + COV(return_C, return_C)*w2*w2; sharpe_ratio = (mean_return - 0.0001) / math.sqrt(variance)"
                }
            ],
            "constraints": [
                {"expression": "w0 + w1 + w2 - 1", "type": "eq"}
            ],
            "optimizerParams": {
                "method": "SLSQP",
                "options": {"maxiter": 1000}
            }
        }
    }

    # 2. 工程设计优化
    all_configs["configs"]["engineering_design"] = {
        "title": config_names["engineering_design"],
        "description": "工程设计成本最小化优化，考虑材料成本和加工成本",
        "dataset": None,
        "type": "engineering",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "scipy",
            "objectiveType": "single",
            "variables": [
                {"name": "x1", "type": "continuous", "bounds": [0.1, 2.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.1, 2.0]},
                {"name": "x3", "type": "continuous", "bounds": [1, 10]}
            ],
            "objectives": [
                {
                    "name": "cost",
                    "type": "minimize",
                    "expression": "material_cost = 2*3.14159*x1*x2*x3; end_cost = 3.14159*x2**2*x1; total_cost = material_cost + end_cost"
                }
            ],
            "constraints": [
                {"expression": "3.14159*x2**2*x3 - 10", "type": "ineq"},
                {"expression": "50 - 3.14159*x2**2*x3", "type": "ineq"},
                {"expression": "x1 - 0.05", "type": "ineq"}
            ],
            "optimizerParams": {
                "method": "SLSQP",
                "options": {"maxiter": 1000}
            }
        }
    }

    # 3. Rosenbrock函数优化
    all_configs["configs"]["rosenbrock"] = {
        "title": config_names["rosenbrock"],
        "description": "经典Rosenbrock函数最小值优化测试",
        "dataset": None,
        "type": "benchmark",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "pso",
            "objectiveType": "single",
            "variables": [
                {"name": "x1", "type": "continuous", "bounds": [-5, 10]},
                {"name": "x2", "type": "continuous", "bounds": [-5, 10]}
            ],
            "objectives": [
                {
                    "name": "rosenbrock",
                    "type": "minimize",
                    "expression": "term1 = (1 - x1)**2; term2 = 100*(x2 - x1**2)**2; rosenbrock = term1 + term2"
                }
            ],
            "optimizerParams": {
                "swarmsize": 50,
                "maxiter": 100
            }
        }
    }

    # 4. 单位圆约束优化
    all_configs["configs"]["unit_circle_max"] = {
        "title": config_names["unit_circle_max"],
        "description": "单位圆内最大化x+y的带约束优化问题",
        "dataset": None,
        "type": "constrained",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "scipy",
            "objectiveType": "single",
            "variables": [
                {"name": "x", "type": "continuous", "bounds": [-1, 1]},
                {"name": "y", "type": "continuous", "bounds": [-1, 1]}
            ],
            "objectives": [
                {
                    "name": "max_sum",
                    "type": "maximize",
                    "expression": "objective_value = x + y"
                }
            ],
            "constraints": [
                {"expression": "1 - x**2 - y**2", "type": "ineq"}
            ],
            "optimizerParams": {
                "method": "SLSQP",
                "options": {"maxiter": 200}
            }
        }
    }

    # 5. 代理模型测试
    surrogate_data = pd.DataFrame({
        'feature1': np.random.uniform(-5, 5, 200),
        'feature2': np.random.uniform(-5, 5, 200),
        'feature3': np.random.uniform(-5, 5, 200),
        'target': np.sin(np.random.uniform(-5, 5, 200)) +
                  np.cos(np.random.uniform(-5, 5, 200)) +
                  np.random.normal(0, 0.1, 200)
    })
    dataset_file = save_dataset("surrogate_model", surrogate_data)

    all_configs["configs"]["surrogate_model"] = {
        "title": config_names["surrogate_model"],
        "description": "LightGBM代理模型拟合精度测试",
        "dataset": dataset_file,
        "type": "surrogate",
        "content": {
            "optimizationMode": "training",
            "surrogateModel": "lightgbm",
            "optimizer": "pso",
            "objectiveType": "single",
            "variables": [
                {"name": "feature1", "type": "continuous", "bounds": [-5, 5]},
                {"name": "feature2", "type": "continuous", "bounds": [-5, 5]},
                {"name": "feature3", "type": "continuous", "bounds": [-5, 5]}
            ],
            "objectives": [
                {"name": "minimize_target", "type": "minimize", "expression": "[target]"}
            ],
            "targetVariable": "target",
            "modelParams": {
                "n_estimators": 50,
                "learning_rate": 0.1,
                "verbose": -1
            },
            "optimizerParams": {
                "swarmsize": 20,
                "maxiter": 10
            }
        }
    }

    # 6. ZDT1多目标优化
    all_configs["configs"]["zdt1"] = {
        "title": config_names["zdt1"],
        "description": "经典ZDT1多目标优化问题",
        "dataset": None,
        "type": "multiobjective",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "nsga2",
            "objectiveType": "multi",
            "variables": [
                {"name": "x1", "type": "continuous", "bounds": [0, 1]},
                {"name": "x2", "type": "continuous", "bounds": [0, 1]}
            ],
            "objectives": [
                {
                    "name": "f1",
                    "type": "minimize",
                    "expression": "f1_value = x1"
                },
                {
                    "name": "f2",
                    "type": "minimize",
                    "expression": "g = 1 + 9 * x2; h = 1 - sqrt(x1 / g); f2_value = g * h"
                }
            ],
            "optimizerParams": {
                "population_size": 40,
                "generations": 20
            }
        }
    }

    # 7. 工厂排班优化
    all_configs["configs"]["factory_scheduling"] = {
        "title": config_names["factory_scheduling"],
        "description": "工厂人员排班成本优化",
        "dataset": None,
        "type": "scheduling",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "differential_evolution",
            "objectiveType": "single",
            "variables": [
                {"name": "staff_weekday", "type": "continuous", "bounds": [10, 20]},
                {"name": "staff_weekend", "type": "continuous", "bounds": [5, 15]}
            ],
            "objectives": [
                {
                    "name": "min_cost",
                    "type": "minimize",
                    "expression": "weekday_cost = 5 * staff_weekday * 100; weekend_cost = 2 * staff_weekend * 150; total_cost = weekday_cost + weekend_cost"
                }
            ],
            "constraints": [
                {"expression": "staff_weekday - 8", "type": "ineq"},
                {"expression": "staff_weekend - 5", "type": "ineq"}
            ],
            "optimizerParams": {
                "population_size": 20,
                "generations": 30,
                "tol": 0.1
            }
        }
    }

    # 8. 生产调度优化
    all_configs["configs"]["production_scheduling"] = {
        "title": config_names["production_scheduling"],
        "description": "生产批量和速率优化",
        "dataset": None,
        "type": "production",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "pso",
            "objectiveType": "single",
            "variables": [
                {"name": "batch_size", "type": "continuous", "bounds": [100, 1000]},
                {"name": "production_rate", "type": "continuous", "bounds": [10, 100]},
                {"name": "overtime_hours", "type": "continuous", "bounds": [0, 40]}
            ],
            "objectives": [
                {
                    "name": "min_total_cost",
                    "type": "minimize",
                    "expression": "production_cost = batch_size / production_rate * 50; overtime_cost = overtime_hours * 100; total_cost = production_cost + overtime_cost"
                }
            ],
            "constraints": [
                {"expression": "batch_size - 500", "type": "ineq"},
                {"expression": "production_rate - 20", "type": "ineq"},
                {"expression": "40 - overtime_hours", "type": "ineq"}
            ],
            "optimizerParams": {
                "swarmsize": 30,
                "maxiter": 100
            }
        }
    }

    # 9. 供应链优化
    all_configs["configs"]["supply_chain"] = {
        "title": config_names["supply_chain"],
        "description": "多工厂生产分配优化",
        "dataset": None,
        "type": "supply_chain",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "scipy",
            "objectiveType": "single",
            "variables": [
                {"name": "factory1_output", "type": "continuous", "bounds": [0, 2000]},
                {"name": "factory2_output", "type": "continuous", "bounds": [0, 1500]},
                {"name": "factory3_output", "type": "continuous", "bounds": [0, 2500]}
            ],
            "objectives": [
                {
                    "name": "min_total_cost",
                    "type": "minimize",
                    "expression": "cost1 = factory1_output * 5; cost2 = factory2_output * 6; cost3 = factory3_output * 4.5; total_cost = cost1 + cost2 + cost3"
                }
            ],
            "constraints": [
                {"expression": "factory1_output + factory2_output + factory3_output - 4500", "type": "eq"},
                {"expression": "2000 - factory1_output", "type": "ineq"},
                {"expression": "1500 - factory2_output", "type": "ineq"},
                {"expression": "2500 - factory3_output", "type": "ineq"}
            ],
            "optimizerParams": {
                "method": "SLSQP",
                "options": {"maxiter": 1000}
            }
        }
    }

    # 10. 风险平价投资组合
    risk_parity_data = pd.DataFrame({
        'asset1_returns': np.random.normal(0.001, 0.02, 100),
        'asset2_returns': np.random.normal(0.0012, 0.025, 100),
        'asset3_returns': np.random.normal(0.0008, 0.018, 100),
        'asset4_returns': np.random.normal(0.0011, 0.022, 100)
    })
    dataset_file = save_dataset("risk_parity", risk_parity_data)

    all_configs["configs"]["risk_parity"] = {
        "title": config_names["risk_parity"],
        "description": "风险平价投资组合优化",
        "dataset": dataset_file,
        "type": "portfolio",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "scipy",
            "objectiveType": "single",
            "variables": [
                {"name": "w0", "type": "continuous", "bounds": [0.01, 0.8]},
                {"name": "w1", "type": "continuous", "bounds": [0.01, 0.8]},
                {"name": "w2", "type": "continuous", "bounds": [0.01, 0.8]},
                {"name": "w3", "type": "continuous", "bounds": [0.01, 0.8]}
            ],
            "objectives": [
                {
                    "name": "risk_parity",
                    "type": "minimize",
                    "expression": "var1 = VAR(asset1_returns); var2 = VAR(asset2_returns); var3 = VAR(asset3_returns); var4 = VAR(asset4_returns); cov12 = COV(asset1_returns, asset2_returns); cov13 = COV(asset1_returns, asset3_returns); cov14 = COV(asset1_returns, asset4_returns); cov23 = COV(asset2_returns, asset3_returns); cov24 = COV(asset2_returns, asset4_returns); cov34 = COV(asset3_returns, asset4_returns); portfolio_var = w0*w0*var1 + w1*w1*var2 + w2*w2*var3 + w3*w3*var4 + 2*w0*w1*cov12 + 2*w0*w2*cov13 + 2*w0*w3*cov14 + 2*w1*w2*cov23 + 2*w1*w3*cov24 + 2*w2*w3*cov34; rc1 = (w0*w0*var1 + w0*w1*cov12 + w0*w2*cov13 + w0*w3*cov14) / portfolio_var; rc2 = (w1*w0*cov12 + w1*w1*var2 + w1*w2*cov23 + w1*w3*cov24) / portfolio_var; rc3 = (w2*w0*cov13 + w2*w1*cov23 + w2*w2*var3 + w2*w3*cov34) / portfolio_var; rc4 = (w3*w0*cov14 + w3*w1*cov24 + w3*w2*cov34 + w3*w3*var4) / portfolio_var; risk_parity = (rc1 - 0.25)**2 + (rc2 - 0.25)**2 + (rc3 - 0.25)**2 + (rc4 - 0.25)**2"
                }
            ],
            "constraints": [
                {"expression": "w0 + w1 + w2 + w3 - 1", "type": "eq"}
            ],
            "optimizerParams": {
                "method": "SLSQP",
                "options": {"maxiter": 500}
            }
        }
    }

    # 11. 均值-方差优化
    mean_var_data = pd.DataFrame({
        'stock_returns': np.random.normal(0.0015, 0.025, 100),
        'bond_returns': np.random.normal(0.0008, 0.015, 100),
        'reit_returns': np.random.normal(0.0012, 0.020, 100)
    })
    dataset_file = save_dataset("mean_variance", mean_var_data)

    all_configs["configs"]["mean_variance"] = {
        "title": config_names["mean_variance"],
        "description": "经典均值-方差投资组合优化",
        "dataset": dataset_file,
        "type": "portfolio",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "scipy",
            "objectiveType": "single",
            "variables": [
                {"name": "w_stock", "type": "continuous", "bounds": [0.0, 0.8]},
                {"name": "w_bond", "type": "continuous", "bounds": [0.0, 0.6]},
                {"name": "w_reit", "type": "continuous", "bounds": [0.0, 0.4]}
            ],
            "objectives": [
                {
                    "name": "efficient_frontier",
                    "type": "minimize",
                    "expression": "mean_return = MEAN(stock_returns)*w_stock + MEAN(bond_returns)*w_bond + MEAN(reit_returns)*w_reit; var_stock = VAR(stock_returns); var_bond = VAR(bond_returns); var_reit = VAR(reit_returns); cov_sb = COV(stock_returns, bond_returns); cov_sr = COV(stock_returns, reit_returns); cov_br = COV(bond_returns, reit_returns); portfolio_variance = w_stock*w_stock*var_stock + w_bond*w_bond*var_bond + w_reit*w_reit*var_reit + 2*w_stock*w_bond*cov_sb + 2*w_stock*w_reit*cov_sr + 2*w_bond*w_reit*cov_br; risk_aversion = 2; utility = portfolio_variance * risk_aversion - mean_return"
                }
            ],
            "constraints": [
                {"expression": "w_stock + w_bond + w_reit - 1", "type": "eq"},
                {
                    "expression": "MEAN(stock_returns)*w_stock + MEAN(bond_returns)*w_bond + MEAN(reit_returns)*w_reit - 0.0010",
                    "type": "ineq"}
            ],
            "optimizerParams": {
                "method": "SLSQP",
                "options": {"maxiter": 1000}
            }
        }
    }

    # 12. 目标波动率优化
    target_vol_data = pd.DataFrame({
        'growth_asset': np.random.normal(0.0018, 0.028, 100),
        'value_asset': np.random.normal(0.0012, 0.022, 100),
        'defensive_asset': np.random.normal(0.0006, 0.012, 100)
    })
    dataset_file = save_dataset("target_volatility", target_vol_data)

    all_configs["configs"]["target_volatility"] = {
        "title": config_names["target_volatility"],
        "description": "目标波动率投资组合优化",
        "dataset": dataset_file,
        "type": "portfolio",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "scipy",
            "objectiveType": "single",
            "variables": [
                {"name": "w_growth", "type": "continuous", "bounds": [0.0, 0.7]},
                {"name": "w_value", "type": "continuous", "bounds": [0.0, 0.7]},
                {"name": "w_defensive", "type": "continuous", "bounds": [0.0, 0.6]}
            ],
            "objectives": [
                {
                    "name": "max_return_vol_target",
                    "type": "maximize",
                    "expression": "portfolio_return = MEAN(growth_asset)*w_growth + MEAN(value_asset)*w_value + MEAN(defensive_asset)*w_defensive"
                }
            ],
            "constraints": [
                {"expression": "w_growth + w_value + w_defensive - 1", "type": "eq"},
                {
                    "expression": "sqrt(VAR(growth_asset)*w_growth*w_growth + VAR(value_asset)*w_value*w_value + VAR(defensive_asset)*w_defensive*w_defensive + 2*COV(growth_asset, value_asset)*w_growth*w_value + 2*COV(growth_asset, defensive_asset)*w_growth*w_defensive + 2*COV(value_asset, defensive_asset)*w_value*w_defensive) - 0.015",
                    "type": "eq"}
            ],
            "optimizerParams": {
                "method": "SLSQP",
                "options": {"maxiter": 1000}
            }
        }
    }

    # 13. 优化算法对比
    all_configs["configs"]["optimizer_comparison"] = {
        "title": config_names["optimizer_comparison"],
        "description": "不同优化算法性能对比测试",
        "dataset": None,
        "type": "comparison",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "pso",
            "objectiveType": "single",
            "variables": [
                {"name": "x1", "type": "continuous", "bounds": [-5, 5]},
                {"name": "x2", "type": "continuous", "bounds": [-5, 5]},
                {"name": "x3", "type": "continuous", "bounds": [-5, 5]}
            ],
            "objectives": [
                {
                    "name": "sphere",
                    "type": "minimize",
                    "expression": "x1_sq = x1**2; x2_sq = x2**2; x3_sq = x3**2; sphere = x1_sq + x2_sq + x3_sq"
                }
            ],
            "optimizerParams": {
                "swarmsize": 30,
                "maxiter": 50
            }
        }
    }

    # 14. 历史生产排程
    production_hist_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=180, freq='D'),
        'order_volume': np.random.poisson(50, 180) + np.random.randint(10, 30, 180),
        'machine_breakdowns': np.random.poisson(0.2, 180),
        'worker_absenteeism': np.random.binomial(20, 0.05, 180),
        'material_delivery_delay': np.random.exponential(2, 180),
        'energy_consumption': np.random.normal(1000, 200, 180),
        'production_quality': np.random.beta(8, 2, 180),
        'overtime_hours': np.random.poisson(5, 180)
    })
    dataset_file = save_dataset("historical_production", production_hist_data)

    all_configs["configs"]["historical_production"] = {
        "title": config_names["historical_production"],
        "description": "基于历史数据的生产排程优化",
        "dataset": dataset_file,
        "type": "production",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "pso",
            "objectiveType": "single",
            "variables": [
                {"name": "daily_capacity", "type": "continuous", "bounds": [60, 120]},
                {"name": "safety_stock", "type": "continuous", "bounds": [5, 30]},
                {"name": "maintenance_frequency", "type": "continuous", "bounds": [2, 10]},
                {"name": "worker_shifts", "type": "continuous", "bounds": [1.5, 3]}
            ],
            "objectives": [
                {
                    "name": "max_efficiency",
                    "type": "maximize",
                    "expression": "predicted_breakdowns = MEAN(machine_breakdowns) / maintenance_frequency; predicted_absenteeism = MEAN(worker_absenteeism) / worker_shifts; predicted_quality = MEAN(production_quality) * (1 - 0.1 * predicted_breakdowns); predicted_overtime = MEAN(overtime_hours) * (daily_capacity / MEAN(order_volume)); efficiency_score = predicted_quality / (predicted_overtime + 1)"
                }
            ],
            "constraints": [
                {"expression": "daily_capacity - MEAN(order_volume) - safety_stock", "type": "ineq"},
                {"expression": "1800 - MEAN(energy_consumption) * daily_capacity / MEAN(order_volume)", "type": "ineq"}
            ],
            "optimizerParams": {
                "swarmsize": 30,
                "maxiter": 200
            }
        }
    }

    # 15. 逆向优化-连续变量
    process_data = pd.DataFrame({
        'temperature': np.random.uniform(150, 300, 200),
        'pressure': np.random.uniform(50, 150, 200),
        'speed': np.random.uniform(100, 500, 200),
        'cooling_time': np.random.uniform(10, 60, 200),
        'quality_rate': np.zeros(200)
    })
    for i in range(200):
        temp = process_data.loc[i, 'temperature']
        press = process_data.loc[i, 'pressure']
        speed = process_data.loc[i, 'speed']
        cooling = process_data.loc[i, 'cooling_time']
        quality = (0.6 * np.exp(-((temp - 220) / 50) ** 2) +
                   0.3 * np.exp(-((press - 100) / 30) ** 2) +
                   0.1 * (1 - abs(speed - 300) / 200) +
                   0.2 * np.tanh(cooling / 30) -
                   0.1 * ((temp - 220) * (press - 100)) / 5000)
        process_data.loc[i, 'quality_rate'] = np.clip(quality, 0, 1)

    dataset_file = save_dataset("inverse_continuous", process_data)

    all_configs["configs"]["inverse_continuous"] = {
        "title": config_names["inverse_continuous"],
        "description": "逆向优化-工艺参数反推",
        "dataset": dataset_file,
        "type": "inverse",
        "content": {
            "optimizationMode": "inverse",
            "surrogateModel": "noModel",
            "optimizer": "differential_evolution",
            "objectiveType": "single",
            "variables": [
                {"name": "temperature", "type": "continuous", "bounds": [150, 300]},
                {"name": "pressure", "type": "continuous", "bounds": [50, 150]},
                {"name": "speed", "type": "continuous", "bounds": [100, 500]},
                {"name": "cooling_time", "type": "continuous", "bounds": [10, 60]}
            ],
            "objectives": [
                {
                    "name": "quality_deviation",
                    "type": "minimize",
                    "expression": "abs((0.6 *exp(-((temperature-220)/50)**2) + 0.3 * exp(-((pressure-100)/30)**2) + 0.1 * (1 - abs(speed-300)/200) + 0.2 * tanh(cooling_time/30) - 0.1 * ((temperature-220)*(pressure-100))/5000) - 0.95)"
                }
            ],
            "constraints": [
                {"expression": "temperature + pressure - 350", "type": "ineq"}
            ],
            "inverseTarget": 0.95,
            "inverseTolerance": 0.02,
            "optimizerParams": {
                "population_size": 50,
                "generations": 100
            }
        }
    }

    # 16. 逆向优化-离散变量
    supply_chain_hist = pd.DataFrame({
        'week': range(1, 53),
        'demand_region_A': np.random.poisson(1000, 52) + np.random.randint(-200, 200, 52),
        'demand_region_B': np.random.poisson(800, 52) + np.random.randint(-150, 150, 52),
        'demand_region_C': np.random.poisson(600, 52) + np.random.randint(-100, 100, 52),
        'transport_cost_factor': np.random.uniform(0.8, 1.2, 52),
        'warehouse_utilization': np.random.beta(2, 2, 52)
    })
    dataset_file = save_dataset("inverse_discrete", supply_chain_hist)

    all_configs["configs"]["inverse_discrete"] = {
        "title": config_names["inverse_discrete"],
        "description": "逆向优化-供应链网络设计",
        "dataset": dataset_file,
        "type": "inverse",
        "content": {
            "optimizationMode": "inverse",
            "surrogateModel": "noModel",
            "optimizer": "ga",
            "objectiveType": "single",
            "variables": [
                {"name": "warehouse_A", "type": "integer", "bounds": [0, 2]},
                {"name": "warehouse_B", "type": "integer", "bounds": [0, 2]},
                {"name": "warehouse_C", "type": "integer", "bounds": [0, 3]},
                {"name": "transport_mode", "type": "integer", "bounds": [0, 2]},
                {"name": "inventory_strategy", "type": "integer", "bounds": [0, 2]}
            ],
            "objectives": [
                {
                    "name": "cost_service_tradeoff",
                    "type": "minimize",
                    "expression": "(50000 * (warehouse_A == 0) + 80000 * (warehouse_A == 1) + 120000 * (warehouse_A == 2) + 40000 * (warehouse_B == 0) + 70000 * (warehouse_B == 1) + 100000 * (warehouse_B == 2) + 30000 * (warehouse_C == 0) + 50000 * (warehouse_C == 1) + 80000 * (warehouse_C == 2) + 0 * (warehouse_C == 3) + 1000000 * abs((0.2 * (warehouse_A == 0) + 0.3 * (warehouse_A == 1) + 0.4 * (warehouse_A == 2) + 0.15 * (warehouse_B == 0) + 0.25 * (warehouse_B == 1) + 0.35 * (warehouse_B == 2) + 0.1 * (warehouse_C == 0) + 0.2 * (warehouse_C == 1) + 0.3 * (warehouse_C == 2) + 0 * (warehouse_C == 3) + 0.1 * (transport_mode == 0) + 0.05 * (transport_mode == 1) + 0.08 * (transport_mode == 2) + 0.05 * (inventory_strategy == 0) + 0.1 * (inventory_strategy == 1) + 0.15 * (inventory_strategy == 2)) - 0.95))"
                }
            ],
            "inverseTarget": 0.95,
            "inverseTolerance": 0.03,
            "optimizerParams": {
                "population_size": 30,
                "generations": 50,
                "crossover_prob": 0.8,
                "mutation_prob": 0.1
            }
        }
    }

    # 17. 蚁群算法-物流路径
    locations = pd.DataFrame({
        'location_id': range(8),
        'x': np.random.uniform(0, 100, 8),
        'y': np.random.uniform(0, 100, 8),
        'demand': np.random.randint(10, 50, 8)
    })
    traffic_data = pd.DataFrame({
        'time_slot': range(24),
        'traffic_congestion': [0.2 + 0.6 * (1 + np.sin(2 * np.pi * t / 24 - np.pi / 2)) / 2 for t in range(24)],
        'fuel_cost': [5.0 + 1.0 * np.sin(2 * np.pi * t / 24) for t in range(24)]
    })
    dataset_file = save_dataset("ant_colony", pd.concat([locations, traffic_data], axis=1))

    all_configs["configs"]["ant_colony"] = {
        "title": config_names["ant_colony"],
        "description": "蚁群算法-物流路径规划",
        "dataset": dataset_file,
        "type": "metaheuristic",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "ant_colony",
            "objectiveType": "single",
            "variables": [
                {"name": f"visit_order_{i}", "type": "integer", "bounds": [0, 7]}
                for i in range(8)
            ],
            "objectives": [
                {
                    "name": "total_route_cost",
                    "type": "minimize",
                    "expression": "total_distance = 0; for i in range(7): from_loc = visit_order_i; to_loc = visit_order_{i+1}; total_distance += distance_matrix[from_loc, to_loc]; total_distance += distance_matrix[visit_order_7, visit_order_0]; avg_traffic = MEAN(traffic_congestion); time_penalty = total_distance * avg_traffic * 0.1; avg_fuel_cost = MEAN(fuel_cost); fuel_cost_total = total_distance * avg_fuel_cost * 0.01; total_route_cost = total_distance + time_penalty + fuel_cost_total"
                }
            ],
            "constraints": [
                {"expression": "sum([visit_order_0, visit_order_1, visit_order_2, visit_order_3, visit_order_4, visit_order_5, visit_order_6, visit_order_7]) - 28", "type": "eq"}
            ],
            "optimizerParams": {
                "ants_count": 20,
                "iterations": 50,
                "alpha": 1.0,
                "beta": 2.0,
                "evaporation": 0.5
            }
        }
    }

    # 18. NSGA2-生产质量效率
    production_quality_data = pd.DataFrame({
        'machine_speed': np.random.uniform(80, 120, 500),
        'temperature': np.random.uniform(180, 220, 500),
        'pressure': np.random.uniform(60, 100, 500),
        'material_batch': np.random.choice(['A', 'B', 'C'], 500),
        'operator_skill': np.random.choice([1, 2, 3], 500),
        'actual_quality': np.zeros(500),
        'actual_efficiency': np.zeros(500),
        'actual_energy': np.zeros(500)
    })
    for i in range(500):
        speed = production_quality_data.loc[i, 'machine_speed']
        temp = production_quality_data.loc[i, 'temperature']
        press = production_quality_data.loc[i, 'pressure']
        skill = production_quality_data.loc[i, 'operator_skill']
        quality = (0.6 * (1 - abs(temp - 200) / 40) +
                   0.3 * (1 - abs(press - 80) / 40) +
                   0.1 * (speed - 80) / 40 +
                   0.2 * (skill - 1) / 2)
        efficiency = (0.7 * speed / 120 +
                      0.2 * (1 - abs(temp - 190) / 30) +
                      0.1 * (press - 60) / 40)
        energy = (speed * 0.8 + temp * 0.5 + press * 0.3 -
                  10 * quality + 5 * (production_quality_data.loc[i, 'material_batch'] == 'A'))
        production_quality_data.loc[i, 'actual_quality'] = np.clip(quality, 0, 1)
        production_quality_data.loc[i, 'actual_efficiency'] = np.clip(efficiency, 0, 1)
        production_quality_data.loc[i, 'actual_energy'] = energy

    dataset_file = save_dataset("nsga2_production", production_quality_data)

    all_configs["configs"]["nsga2_production"] = {
        "title": config_names["nsga2_production"],
        "description": "NSGA2-生产质量效率多目标优化",
        "dataset": dataset_file,
        "type": "multiobjective",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "nsga2",
            "objectiveType": "multi",
            "variables": [
                {"name": "machine_speed", "type": "continuous", "bounds": [80, 120]},
                {"name": "temperature", "type": "continuous", "bounds": [180, 220]},
                {"name": "pressure", "type": "continuous", "bounds": [60, 100]},
                {"name": "operator_skill", "type": "integer", "bounds": [1, 3]}
            ],
            "objectives": [
                {
                    "name": "maximize_quality",
                    "type": "maximize",
                    "expression": "(0.6 * (1 - abs(temperature - 200) / 40) + 0.3 * (1 - abs(pressure - 80) / 40) + 0.1 * (machine_speed - 80) / 40 + 0.2 * (operator_skill - 1) / 2)"
                },
                {
                    "name": "maximize_efficiency",
                    "type": "maximize",
                    "expression": "(0.7 * machine_speed / 120 + 0.2 * (1 - abs(temperature - 190) / 30) + 0.1 * (pressure - 60) / 40)"
                },
                {
                    "name": "minimize_energy",
                    "type": "minimize",
                    "expression": "(machine_speed * 0.8 + temperature * 0.5 + pressure * 0.3 - 10 * (0.6 * (1 - abs(temperature - 200) / 40) + 0.3 * (1 - abs(pressure - 80) / 40) + 0.1 * (machine_speed - 80) / 40 + 0.2 * (operator_skill - 1) / 2))"
                }
            ],
            "optimizerParams": {
                "population_size": 100,
                "generations": 200
            }
        }
    }

    # 19. NSGA2-设备维护
    equipment_data = pd.DataFrame({
        'machine_id': np.repeat(range(1, 51), 24),
        'month': np.tile(range(1, 25), 50),
        'operating_hours': np.random.poisson(160, 1200) + np.random.randint(0, 80, 1200),
        'maintenance_count': np.random.poisson(2, 1200),
        'breakdown_count': np.random.poisson(0.3, 1200),
        'energy_consumption': np.random.normal(1000, 200, 1200),
        'output_quality': np.random.beta(8, 2, 1200)
    })
    dataset_file = save_dataset("nsga2_maintenance", equipment_data)

    all_configs["configs"]["nsga2_maintenance"] = {
        "title": config_names["nsga2_maintenance"],
        "description": "NSGA2-设备维护多目标优化",
        "dataset": dataset_file,
        "type": "multiobjective",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "nsga2",
            "objectiveType": "multi",
            "variables": [
                {"name": "preventive_interval", "type": "integer", "bounds": [100, 500]},
                {"name": "spare_part_level", "type": "continuous", "bounds": [0.1, 0.8]},
                {"name": "maintenance_team", "type": "integer", "bounds": [2, 10]},
                {"name": "inspection_frequency", "type": "integer", "bounds": [50, 200]}
            ],
            "objectives": [
                {
                    "name": "minimize_total_cost",
                    "type": "minimize",
                    "expression": "(160 / preventive_interval * 800 + 0.3 * (1 - preventive_interval / 500) * 5000 + spare_part_level * 100000 + maintenance_team * 75000)"
                },
                {
                    "name": "maximize_availability",
                    "type": "maximize",
                    "expression": "((0.9 - 0.3 * (preventive_interval - 100) / 400) * (0.8 - 0.2 * (inspection_frequency - 50) / 150) * (0.7 + 0.2 * (maintenance_team - 2) / 8))"
                },
                {
                    "name": "minimize_energy_waste",
                    "type": "minimize",
                    "expression": "((preventive_interval - 200) ** 2 / 100000 + maintenance_team * 2 * 50)"
                }
            ],
            "optimizerParams": {
                "population_size": 80,
                "generations": 150
            }
        }
    }

    # 20. NSGA2-供应链库存
    inventory_data = pd.DataFrame({
        'product_id': np.repeat(range(1, 21), 36),
        'period': np.tile(range(1, 37), 20),
        'demand': np.random.poisson(100, 720) + np.random.randint(-20, 50, 720),
        'lead_time': np.random.poisson(7, 720) + np.random.randint(-2, 5, 720),
        'stockout_events': np.random.poisson(0.5, 720),
        'holding_cost': np.random.uniform(5, 15, 720),
        'ordering_cost': np.random.uniform(50, 150, 720)
    })
    dataset_file = save_dataset("nsga2_inventory", inventory_data)

    all_configs["configs"]["nsga2_inventory"] = {
        "title": config_names["nsga2_inventory"],
        "description": "NSGA2-供应链库存多目标优化",
        "dataset": dataset_file,
        "type": "multiobjective",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "nsga2",
            "objectiveType": "multi",
            "variables": [
                {"name": "safety_stock_level", "type": "continuous", "bounds": [0.1, 0.4]},
                {"name": "reorder_point", "type": "continuous", "bounds": [0.5, 2.0]},
                {"name": "order_quantity", "type": "continuous", "bounds": [0.8, 3.0]},
                {"name": "review_period", "type": "integer", "bounds": [1, 14]}
            ],
            "objectives": [
                {
                    "name": "minimize_total_cost",
                    "type": "minimize",
                    "expression": "(safety_stock_level * 100 * 10 * 30 + 100 * 100 / (order_quantity * 100) + 0.5 * (1 - safety_stock_level) * 1000)"
                },
                {
                    "name": "maximize_service_level",
                    "type": "maximize",
                    "expression": "((1 - 0.3 * (reorder_point - 1.0)) * (0.8 + 0.4 * safety_stock_level) * (1 - 0.1 * (review_period - 1) / 13))"
                },
                {
                    "name": "minimize_bullwhip_effect",
                    "type": "minimize",
                    "expression": "(abs(order_quantity - 1.5) * 0.3 + abs(review_period - 7) * 0.02)"
                }
            ],
            "optimizerParams": {
                "population_size": 60,
                "generations": 120
            }
        }
    }

    # 21. NSGA2-质量工艺参数
    quality_process_data = pd.DataFrame({
        'process_temp': np.random.uniform(150, 250, 200),
        'process_pressure': np.random.uniform(50, 150, 200),
        'process_time': np.random.uniform(30, 120, 200),
        'material_viscosity': np.random.uniform(1000, 5000, 200),
        'defect_rate': np.zeros(200),
        'throughput_rate': np.zeros(200),
        'energy_usage': np.zeros(200)
    })
    for i in range(200):
        temp = quality_process_data.loc[i, 'process_temp']
        pressure = quality_process_data.loc[i, 'process_pressure']
        time = quality_process_data.loc[i, 'process_time']
        viscosity = quality_process_data.loc[i, 'material_viscosity']
        defect_rate = (0.4 * (abs(temp - 200) / 50) +
                       0.3 * (abs(pressure - 100) / 50) +
                       0.2 * (abs(time - 75) / 45) +
                       0.1 * (abs(viscosity - 3000) / 2000))
        throughput = (0.5 * (1 / time) * 120 +
                      0.3 * (pressure / 150) +
                      0.2 * (temp / 250))
        energy = (temp * 2.0 + pressure * 1.5 + time * 0.8 + viscosity * 0.001)
        quality_process_data.loc[i, 'defect_rate'] = np.clip(defect_rate, 0, 1)
        quality_process_data.loc[i, 'throughput_rate'] = throughput / 3.0
        quality_process_data.loc[i, 'energy_usage'] = energy / 1000

    dataset_file = save_dataset("nsga2_quality", quality_process_data)

    all_configs["configs"]["nsga2_quality"] = {
        "title": config_names["nsga2_quality"],
        "description": "NSGA2-质量工艺参数多目标优化",
        "dataset": dataset_file,
        "type": "multiobjective",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "nsga2",
            "objectiveType": "multi",
            "variables": [
                {"name": "process_temp", "type": "continuous", "bounds": [150, 250]},
                {"name": "process_pressure", "type": "continuous", "bounds": [50, 150]},
                {"name": "process_time", "type": "continuous", "bounds": [30, 120]},
                {"name": "material_viscosity", "type": "continuous", "bounds": [1000, 5000]}
            ],
            "objectives": [
                {
                    "name": "minimize_defect_rate",
                    "type": "minimize",
                    "expression": "(0.4 * (abs(process_temp - 200) / 50) + 0.3 * (abs(process_pressure - 100) / 50) + 0.2 * (abs(process_time - 75) / 45) + 0.1 * (abs(material_viscosity - 3000) / 2000))"
                },
                {
                    "name": "maximize_throughput",
                    "type": "maximize",
                    "expression": "((0.5 * (1 / process_time) * 120 + 0.3 * (process_pressure / 150) + 0.2 * (process_temp / 250)) / 3.0)"
                },
                {
                    "name": "minimize_energy",
                    "type": "minimize",
                    "expression": "((process_temp * 2.0 + process_pressure * 1.5 + process_time * 0.8 + material_viscosity * 0.001) / 1000)"
                }
            ],
            "optimizerParams": {
                "population_size": 50,
                "generations": 100
            }
        }
    }

    # 22. NSGA2-简单测试
    all_configs["configs"]["nsga2_simple"] = {
        "title": config_names["nsga2_simple"],
        "description": "NSGA2-ZDT1简单多目标测试",
        "dataset": None,
        "type": "multiobjective",
        "content": {
            "optimizationMode": "direct",
            "surrogateModel": "noModel",
            "optimizer": "nsga2",
            "objectiveType": "multi",
            "variables": [
                {"name": "x1", "type": "continuous", "bounds": [0, 1]},
                {"name": "x2", "type": "continuous", "bounds": [0, 1]}
            ],
            "objectives": [
                {
                    "name": "f1",
                    "type": "minimize",
                    "expression": "x1"
                },
                {
                    "name": "f2",
                    "type": "minimize",
                    "expression": "(1 + 9 * x2) * (1 - sqrt(x1 / (1 + 9 * x2)))"
                }
            ],
            "optimizerParams": {
                "population_size": 40,
                "generations": 20
            }
        }
    }

    # 保存整合的配置文件
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_configs, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ 所有配置已保存到: {CONFIG_FILE}")
    logging.info(f"📁 数据集保存在: {DATA_DIR}")
    logging.info(f"📊 共生成{len(all_configs['configs'])}个优化配置")


# 执行生成
if __name__ == "__main__":
    generate_all_resources()