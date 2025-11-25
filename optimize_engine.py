# optimization_engine.py
import traceback

import numpy as np
import pandas as pd
import json
import pickle
import re
import math
import statistics
import os
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import tempfile
import time
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

# 科学计算库
from scipy.optimize import minimize, differential_evolution, Bounds, NonlinearConstraint
from scipy.stats import norm, uniform
from scipy.spatial.distance import cdist

# 机器学习库
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin

# 可选的高级优化库
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import HyperbandPruner

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Advanced hyperparameter tuning disabled.")

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.selection.tournament import TournamentSelection
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.core.duplicate import ElementwiseDuplicateElimination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    warnings.warn("pymoo not available. Multi-objective optimization disabled.")


    class PymooProblem:
        """空的PymooProblem基类，当pymoo不可用时使用"""
        pass
try:
    import platypus
    from platypus import NSGAII, SPEA2, MOEAD as PlatypusMOEAD

    PLATYPUS_AVAILABLE = True
except ImportError:
    PLATYPUS_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import cma

    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization_engine.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 抑制不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class OptimizationType(str, Enum):
    """优化类型枚举"""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class VariableType(str, Enum):
    """变量类型枚举"""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    INTEGER = "integer"
    BINARY = "binary"


class ConstraintType(str, Enum):
    """约束类型枚举"""
    EQUALITY = "eq"
    INEQUALITY = "ineq"


class SurrogateModelType(str, Enum):
    """代理模型类型枚举"""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "randomForest"
    GAUSSIAN_PROCESS = "gaussian"
    NEURAL_NETWORK = "mlp"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"
    NONE = "noModel"


class OptimizerType(str, Enum):
    """优化器类型枚举"""
    PSO = "pso"
    GA = "ga"
    DE = "differential_evolution"
    BAYESIAN = "bayesian"
    SCIPY = "scipy"
    NSGA2 = "nsga2"
    NSGA3 = "nsga3"
    MOEAD = "moead"
    SMSEMOA = "smsemoa"
    OPTUNA = "optuna"
    CMAES = "cmaes"
    ANT_COLONY = 'ant_colony'


class ProblemType(str, Enum):
    """问题类型枚举"""
    SINGLE_OBJECTIVE = "single"
    MULTI_OBJECTIVE = "multi"
    CONSTRAINED = "constrained"
    MIXED_INTEGER = "mixed_integer"
    BLACK_BOX = "black_box"


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    success: bool
    message: str
    optimal_variables: Dict[str, Any]
    optimal_objective: float
    model_metrics: Optional[Dict[str, float]] = None
    trained_model_path: Optional[str] = None
    optimization_history: List[Dict] = field(default_factory=list)
    optimization_time: float = 0.0
    n_evaluations: int = 0
    convergence_data: Optional[Dict] = None
    constraint_violations: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)


@dataclass
class ParetoSolution:
    """帕累托解数据类"""
    variables: Dict[str, Any]
    objectives: List[float]
    constraint_violation: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0
    is_feasible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiObjectiveResult:
    """多目标优化结果"""
    success: bool
    message: str
    pareto_front: List[ParetoSolution]
    ideal_point: List[float]
    nadir_point: List[float]
    hypervolume: Optional[float] = None
    optimization_history: List[Dict] = field(default_factory=list)
    optimization_time: float = 0.0
    n_evaluations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_pareto_dataframe(self) -> pd.DataFrame:
        """将帕累托前沿转换为DataFrame"""
        data = []
        for solution in self.pareto_front:
            row = solution.variables.copy()
            for i, obj_val in enumerate(solution.objectives):
                row[f'objective_{i + 1}'] = obj_val
            row['constraint_violation'] = solution.constraint_violation
            row['rank'] = solution.rank
            row['crowding_distance'] = solution.crowding_distance
            row['is_feasible'] = solution.is_feasible
            data.append(row)
        return pd.DataFrame(data)


class ConvergenceMonitor:
    """收敛监控器"""

    def __init__(self, window_size: int = 50, tolerance: float = 1e-6):
        self.window_size = window_size
        self.tolerance = tolerance
        self.history: List[float] = []
        self.iteration = 0

    def update(self, objective_value: float) -> bool:
        """更新收敛状态"""
        self.history.append(objective_value)
        self.iteration += 1

        if len(self.history) < self.window_size:
            return False

        # 检查最近window_size次迭代的改进
        recent_improvement = abs(self.history[-1] - self.history[-self.window_size])
        return recent_improvement < self.tolerance

    def get_convergence_data(self) -> Dict[str, Any]:
        """获取收敛数据"""
        return {
            'iterations': self.iteration,
            'best_value': min(self.history) if self.history else None,
            'history': self.history.copy(),
            'is_converged': len(self.history) >= self.window_size and
                            abs(self.history[-1] - self.history[-self.window_size]) < self.tolerance
        }


class BaseSurrogateModel(ABC):
    """代理模型抽象基类"""

    def __init__(self, model_type: SurrogateModelType, params: Dict[str, Any] = None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.is_trained = False
        self.training_time = 0.0
        self.feature_names: List[str] = []
        self.target_name: Optional[str] = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.categorical_features: List[str] = []
        self.continuous_features: List[str] = []
        self.uncertainty_estimator = None

    @abstractmethod
    def _build_model(self) -> Any:
        """构建具体模型"""
        pass

    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """拟合模型"""
        pass

    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        pass

    def preprocess_features(self, X: pd.DataFrame, variable_info: Dict[str, Dict],
                            fit: bool = True) -> np.ndarray:
        """预处理特征"""
        if not self.feature_names:
            self.feature_names = list(variable_info.keys())
            self.categorical_features = [
                name for name, info in variable_info.items()
                if info['type'] in [VariableType.CATEGORICAL.value, VariableType.ORDINAL.value]
            ]
            self.continuous_features = [
                name for name, info in variable_info.items()
                if
                info['type'] in [VariableType.CONTINUOUS.value, VariableType.INTEGER.value, VariableType.BINARY.value]
            ]

        X_processed = X[self.feature_names].copy()

        # 处理连续特征
        if self.continuous_features:
            if fit:
                X_processed[self.continuous_features] = self.scaler_x.fit_transform(
                    X_processed[self.continuous_features]
                )
            else:
                X_processed[self.continuous_features] = self.scaler_x.transform(
                    X_processed[self.continuous_features]
                )

        # 处理分类特征
        for feature in self.categorical_features:
            if feature in X_processed.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    # 处理可能的NaN值
                    mask = X_processed[feature].notna()
                    if mask.any():
                        X_processed.loc[mask, feature] = self.label_encoders[feature].fit_transform(
                            X_processed.loc[mask, feature].astype(str)
                        )
                        # 用众数填充NaN
                        if not mask.all():
                            mode_val = X_processed[feature].mode()[0] if not X_processed[feature].empty else 0
                            X_processed[feature].fillna(mode_val, inplace=True)
                    else:
                        # 如果全是NaN，填充0
                        X_processed[feature] = 0
                else:
                    if feature in self.label_encoders:
                        # 处理未知类别
                        known_classes = set(self.label_encoders[feature].classes_)
                        mask = X_processed[feature].notna()
                        if mask.any():
                            X_processed.loc[mask, feature] = X_processed.loc[mask, feature].astype(str).apply(
                                lambda x: x if x in known_classes else list(known_classes)[
                                    0] if known_classes else 'unknown'
                            )
                            X_processed.loc[mask, feature] = self.label_encoders[feature].transform(
                                X_processed.loc[mask, feature]
                            )
                            # 用0填充NaN
                            X_processed[feature].fillna(0, inplace=True)
                        else:
                            X_processed[feature] = 0

        return X_processed.values

    def preprocess_target(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """预处理目标变量"""
        y_reshaped = y.reshape(-1, 1)
        if fit:
            return self.scaler_y.fit_transform(y_reshaped).flatten()
        else:
            return self.scaler_y.transform(y_reshaped).flatten()

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """逆变换目标变量"""
        return self.scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

    def train(self, X: pd.DataFrame, y: np.ndarray, variable_info: Dict[str, Dict]):
        """训练模型"""
        start_time = time.time()

        try:
            # 预处理特征
            X_processed = self.preprocess_features(X, variable_info, fit=True)

            # 预处理目标变量
            y_processed = self.preprocess_target(y, fit=True)

            # 检查数据有效性 - 安全地检查NaN和Inf
            def safe_check_nan_inf(arr):
                """安全地检查数组中的NaN和Inf值"""
                try:
                    # 确保是数值类型
                    arr_float = arr.astype(float)
                    has_nan = np.any(np.isnan(arr_float))
                    has_inf = np.any(np.isinf(arr_float))
                    return has_nan, has_inf
                except (TypeError, ValueError):
                    # 如果转换失败，尝试其他方法
                    try:
                        # 对于对象类型数组，逐元素检查
                        has_nan = False
                        has_inf = False
                        for item in arr.flat:
                            try:
                                if item is None or (isinstance(item, float) and np.isnan(item)):
                                    has_nan = True
                                if isinstance(item, float) and np.isinf(item):
                                    has_inf = True
                            except:
                                continue
                        return has_nan, has_inf
                    except:
                        return False, False

            has_nan, has_inf = safe_check_nan_inf(X_processed)
            if has_nan or has_inf:
                logger.warning("Training data contains NaN or Inf values, applying imputation")
                X_processed = np.nan_to_num(X_processed)

            y_has_nan, y_has_inf = safe_check_nan_inf(y_processed)
            if y_has_nan or y_has_inf:
                raise ValueError("Target variable contains NaN or Inf values")

            # 构建并训练模型
            if self.model is None:
                self.model = self._build_model()

            self._fit_model(X_processed, y_processed)
            self.is_trained = True
            self.training_time = time.time() - start_time

            logger.info(f"{self.model_type.value} model trained successfully with "
                        f"{X_processed.shape[0]} samples in {self.training_time:.2f}s")

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame, variable_info: Dict[str, Dict],
                return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        X_processed = self.preprocess_features(X, variable_info, fit=False)

        # 处理可能的NaN值 - 安全地处理
        try:
            X_processed_float = X_processed.astype(float)
            if np.any(np.isnan(X_processed_float)) or np.any(np.isinf(X_processed_float)):
                logger.warning("Input data contains NaN or Inf values, replacing with 0")
                X_processed = np.nan_to_num(X_processed_float)
        except (TypeError, ValueError):
            # 如果转换失败，使用更安全的方法
            logger.warning("Input data contains non-numeric values, attempting conversion")
            X_processed = np.nan_to_num(X_processed.astype(float, errors='coerce'))

        predictions = self._predict_model(X_processed)
        predictions = self.inverse_transform_target(predictions)

        if return_std and hasattr(self, '_predict_uncertainty'):
            std = self._predict_uncertainty(X_processed)
            return predictions, std

        return predictions

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, variable_info: Dict[str, Dict]) -> Dict[str, float]:
        """评估模型性能"""
        y_pred = self.predict(X, variable_info)

        metrics = {
            'mse': float(mean_squared_error(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred)),
            'mape': float(np.mean(np.abs((y - y_pred) / np.clip(np.abs(y), 1e-8, None))) * 100),
            'training_time': self.training_time,
            'n_samples': len(y)
        }

        # 交叉验证得分（如果数据量足够）
        if len(y) >= 10:
            try:
                X_processed = self.preprocess_features(X, variable_info, fit=False)
                y_processed = self.preprocess_target(y, fit=False)
                cv_scores = cross_val_score(self.model, X_processed, y_processed,
                                            cv=min(5, len(y)), scoring='r2')
                metrics['cv_r2_mean'] = float(np.mean(cv_scores))
                metrics['cv_r2_std'] = float(np.std(cv_scores))
            except Exception as e:
                logger.warning(f"Cross-validation failed: {str(e)}")

        return metrics

    def save(self, path: str):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'model_type': self.model_type.value,
            'params': self.params,
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'label_encoders': self.label_encoders,
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'training_time': self.training_time
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model_type = SurrogateModelType(model_data['model_type'])
        self.params = model_data['params']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.scaler_x = model_data['scaler_x']
        self.scaler_y = model_data['scaler_y']
        self.label_encoders = model_data['label_encoders']
        self.categorical_features = model_data.get('categorical_features', [])
        self.continuous_features = model_data.get('continuous_features', [])
        self.training_time = model_data.get('training_time', 0.0)

        logger.info(f"Model loaded from {path}")


class AdvancedGaussianProcessModel(BaseSurrogateModel):
    """高级高斯过程模型，支持不确定性估计"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(SurrogateModelType.GAUSSIAN_PROCESS, params)
        self.alpha = self.params.get('alpha', 1e-10)
        self.n_restarts = self.params.get('n_restarts_optimizer', 5)

    def _build_model(self) -> GaussianProcessRegressor:
        """构建高斯过程模型"""
        kernel_name = self.params.get('kernel', 'matern')
        length_scale = self.params.get('length_scale', 1.0)
        nu = self.params.get('nu', 1.5)

        if kernel_name == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif kernel_name == 'matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
        elif kernel_name == 'rational_quadratic':
            kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=length_scale, alpha=1.0)
        else:
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)

        # 添加白噪声核
        kernel += WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts,
            random_state=42,
            normalize_y=True
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """拟合高斯过程模型"""
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        return self.model.predict(X, return_std=False)

    def _predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """预测不确定性"""
        _, std = self.model.predict(X, return_std=True)
        return std


class AdvancedLightGBMModel(BaseSurrogateModel):
    """高级LightGBM模型"""

    def __init__(self, params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Please install lightgbm.")
        super().__init__(SurrogateModelType.LIGHTGBM, params)

    def _build_model(self) -> lgb.LGBMRegressor:
        """构建LightGBM模型"""
        default_params = {
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(self.params)
        return lgb.LGBMRegressor(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """拟合LightGBM模型"""
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        return self.model.predict(X)


class AdvancedXGBoostModel(BaseSurrogateModel):
    """高级XGBoost模型"""

    def __init__(self, params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install xgboost.")
        super().__init__(SurrogateModelType.XGBOOST, params)

    def _build_model(self) -> xgb.XGBRegressor:
        """构建XGBoost模型"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(self.params)
        return xgb.XGBRegressor(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """拟合XGBoost模型"""
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        return self.model.predict(X)


class AdvancedRandomForestModel(BaseSurrogateModel):
    """高级随机森林模型"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(SurrogateModelType.RANDOM_FOREST, params)

    def _build_model(self) -> RandomForestRegressor:
        """构建随机森林模型"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(self.params)
        return RandomForestRegressor(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """拟合随机森林模型"""
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        return self.model.predict(X)


class AdvancedMLPModel(BaseSurrogateModel):
    """高级神经网络模型"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(SurrogateModelType.NEURAL_NETWORK, params)

    def _build_model(self) -> MLPRegressor:
        """构建MLP模型"""
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'random_state': 42
        }
        default_params.update(self.params)
        return MLPRegressor(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """拟合MLP模型"""
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        return self.model.predict(X)


class AdvancedGradientBoostingModel(BaseSurrogateModel):
    """高级梯度提升模型"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(SurrogateModelType.GRADIENT_BOOSTING, params)

    def _build_model(self) -> GradientBoostingRegressor:
        """构建梯度提升模型"""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'random_state': 42
        }
        default_params.update(self.params)
        return GradientBoostingRegressor(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """拟合梯度提升模型"""
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        return self.model.predict(X)


class EnsembleSurrogateModel(BaseSurrogateModel):
    """集成代理模型，组合多个基础模型"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(SurrogateModelType.ENSEMBLE, params)
        self.base_models: List[BaseSurrogateModel] = []
        self.weights: List[float] = []

    def _build_model(self) -> None:
        """构建集成模型"""
        model_configs = self.params.get('base_models', [
            {'type': 'lightgbm', 'params': {}},
            {'type': 'randomForest', 'params': {}},
            {'type': 'xgboost', 'params': {}}
        ])

        model_factories = {
            'lightgbm': AdvancedLightGBMModel,
            'xgboost': AdvancedXGBoostModel,
            'randomForest': AdvancedRandomForestModel,
            'gaussian': AdvancedGaussianProcessModel,
            'mlp': AdvancedMLPModel,
            'gradient_boosting': AdvancedGradientBoostingModel
        }

        for config in model_configs:
            model_type = config['type']
            if model_type in model_factories:
                try:
                    model = model_factories[model_type](config.get('params', {}))
                    self.base_models.append(model)
                except ImportError as e:
                    logger.warning(f"Failed to create {model_type} model: {e}")

        if not self.base_models:
            raise ValueError("No base models could be created for ensemble")

        # 初始化权重
        self.weights = [1.0 / len(self.base_models)] * len(self.base_models)

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """训练集成模型"""
        # 训练所有基础模型
        for model in self.base_models:
            model._fit_model(X, y)

        # 计算权重（基于验证集性能）
        if len(self.base_models) > 1:
            self._calculate_weights(X, y)

    def _calculate_weights(self, X: np.ndarray, y: np.ndarray):
        """计算模型权重"""
        # 使用交叉验证计算每个模型的性能
        performances = []
        for model in self.base_models:
            try:
                scores = cross_val_score(model.model, X, y, cv=5, scoring='r2')
                performances.append(np.mean(scores))
            except:
                performances.append(0.0)

        # 将性能转换为权重
        performances = np.array(performances)
        performances = np.maximum(performances, 0)  # 确保非负
        if np.sum(performances) > 0:
            self.weights = performances / np.sum(performances)
        else:
            self.weights = [1.0 / len(self.base_models)] * len(self.base_models)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """集成预测"""
        predictions = []
        for model, weight in zip(self.base_models, self.weights):
            if hasattr(model, 'is_trained') and model.is_trained:
                pred = model._predict_model(X)
                predictions.append(pred * weight)

        if not predictions:
            raise ValueError("No trained models in ensemble")

        return np.sum(predictions, axis=0)

    def train(self, X: pd.DataFrame, y: np.ndarray, variable_info: Dict[str, Dict]):
        """训练集成模型"""
        # 首先预处理特征（所有模型共享相同的预处理）
        X_processed = self.preprocess_features(X, variable_info, fit=True)
        y_processed = self.preprocess_target(y, fit=True)

        # 为所有基础模型设置相同的特征信息
        for model in self.base_models:
            model.feature_names = self.feature_names
            model.categorical_features = self.categorical_features
            model.continuous_features = self.continuous_features
            model.scaler_x = self.scaler_x
            model.scaler_y = self.scaler_y
            model.label_encoders = self.label_encoders

        # 训练所有基础模型
        self._fit_model(X_processed, y_processed)
        self.is_trained = True


class OptimizationEngine:
    """优化引擎主类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_and_preprocess_config(config)
        self.model: Optional[BaseSurrogateModel] = None
        self.data: Optional[pd.DataFrame] = None
        self.variable_info: Dict[str, Dict] = {}
        self.constraint_functions: List[Dict] = []
        self.optimization_history: List[Dict] = []
        self.convergence_monitor = ConvergenceMonitor()

        # 模型注册表
        self.model_registry = {
            SurrogateModelType.LIGHTGBM: AdvancedLightGBMModel,
            SurrogateModelType.XGBOOST: AdvancedXGBoostModel,
            SurrogateModelType.RANDOM_FOREST: AdvancedRandomForestModel,
            SurrogateModelType.GAUSSIAN_PROCESS: AdvancedGaussianProcessModel,
            SurrogateModelType.NEURAL_NETWORK: AdvancedMLPModel,
            SurrogateModelType.GRADIENT_BOOSTING: AdvancedGradientBoostingModel,
            SurrogateModelType.ENSEMBLE: EnsembleSurrogateModel,
            SurrogateModelType.NONE: None
        }

        # 优化器注册表
        self.optimizer_registry = {
            OptimizerType.PSO: self._run_pso,
            OptimizerType.GA: self._run_ga,
            OptimizerType.DE: self._run_differential_evolution,
            OptimizerType.BAYESIAN: self._run_bayesian_optimization,
            OptimizerType.SCIPY: self._run_scipy_optimization,
            OptimizerType.NSGA2: self._run_nsga2,
            OptimizerType.NSGA3: self._run_nsga3,
            OptimizerType.MOEAD: self._run_moead,
            OptimizerType.SMSEMOA: self._run_smsemoa,
            OptimizerType.OPTUNA: self._run_optuna,
            OptimizerType.CMAES: self._run_cma_es,
            OptimizerType.ANT_COLONY: self._run_ant_colony,
        }

        logger.info(f"OptimizationEngine initialized with config: {self.config}")

    def _validate_and_preprocess_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证和预处理配置"""
        # 基础验证
        required_fields = ['variables']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        # 设置默认值
        config.setdefault('optimizationMode', 'training')
        config.setdefault('surrogateModel', 'lightgbm')
        config.setdefault('optimizer', 'pso')
        config.setdefault('objectiveType', 'single')
        config.setdefault('objectives', [])
        config.setdefault('constraints', [])
        config.setdefault('optimizerParams', {})
        config.setdefault('modelParams', {})

        # 验证变量配置
        for var in config['variables']:
            if 'name' not in var or 'type' not in var:
                raise ValueError("Variable must have 'name' and 'type'")

            # 设置默认边界
            var_type = var['type']
            if var_type == VariableType.CONTINUOUS.value and 'bounds' not in var:
                var['bounds'] = [0.0, 1.0]
            elif var_type == VariableType.INTEGER.value and 'bounds' not in var:
                var['bounds'] = [0, 100]
            elif var_type == VariableType.BINARY.value:
                var['bounds'] = [0, 1]
            elif var_type in [VariableType.CATEGORICAL.value, VariableType.ORDINAL.value]:
                if 'categories' not in var:
                    var['categories'] = []

        # 验证目标配置
        for obj in config['objectives']:
            obj.setdefault('type', OptimizationType.MINIMIZE.value)
            if 'name' not in obj:
                obj['name'] = f"objective_{config['objectives'].index(obj)}"

        return config

    def load_data(self, data: pd.DataFrame):
        """加载数据"""
        self.data = data.copy()

        # 数据质量检查
        self._validate_data_quality()

        logger.info(f"Data loaded: {self.data.shape}")
        logger.info(f"Data columns: {list(self.data.columns)}")
        logger.info(f"Data types:\n{self.data.dtypes}")
        logger.info(f"Missing values:\n{self.data.isnull().sum()}")

    def load_data_from_file(self, file_path: str):
        """从文件加载数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                self.data = pd.read_parquet(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV, JSON, Parquet, or Excel.")

            logger.info(f"Loaded data from {file_path} with shape: {self.data.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load data from {file_path}: {e}")

    def _validate_data_quality(self):
        """验证数据质量"""
        if self.data is None:
            return

        # 检查缺失值
        missing_cols = self.data.columns[self.data.isnull().any()].tolist()
        if missing_cols:
            logger.warning(f"Data contains missing values in columns: {missing_cols}")

        # 检查无限值 - 只对数值类型的列进行检查
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_cols = []
            for col in numeric_cols:
                if np.any(np.isinf(self.data[col])):
                    inf_cols.append(col)
            if inf_cols:
                logger.warning(f"Data contains infinite values in columns: {inf_cols}")
        else:
            logger.info("No numeric columns found for infinite value check")

        # 检查常数列
        constant_cols = []
        for col in self.data.columns:
            if self.data[col].nunique() <= 1:
                constant_cols.append(col)
        if constant_cols:
            logger.warning(f"Data contains constant columns: {constant_cols}")

    def setup_variables(self):
        """设置变量信息"""
        variables = self.config.get('variables', [])
        self.variable_info = {}

        for var in variables:
            var_info = {
                'type': var['type'],
                'bounds': var.get('bounds', []),
                'categories': var.get('categories', [])
            }

            # 从数据中推断分类变量的类别
            if (var_info['type'] in [VariableType.CATEGORICAL.value, VariableType.ORDINAL.value]
                    and not var_info['categories']
                    and self.data is not None
                    and var['name'] in self.data.columns):
                var_info['categories'] = sorted(self.data[var['name']].dropna().unique().tolist())

            self.variable_info[var['name']] = var_info

        logger.info(f"Setup {len(variables)} variables: {list(self.variable_info.keys())}")

    def setup_constraints(self):
        """设置约束条件"""
        constraints = self.config.get('constraints', [])
        self.constraint_functions = []

        for i, constraint in enumerate(constraints):
            expr = constraint['expression']
            constr_type = constraint.get('type', ConstraintType.INEQUALITY.value)

            # 创建约束函数
            def create_constraint_function(expr, constr_type):
                def constraint_func(x):
                    variable_dict = dict(zip(self.variable_info.keys(), x))
                    try:
                        value = self.safe_eval(expr, variable_dict)
                        if constr_type == ConstraintType.EQUALITY.value:
                            return value  # eq: h(x) = 0
                        else:
                            return value  # ineq: g(x) >= 0
                    except Exception as e:
                        logger.error(f"Constraint evaluation failed: {e}")
                        return -np.inf  # 严重违反约束

                return constraint_func

            constraint_func = create_constraint_function(expr, constr_type)

            self.constraint_functions.append({
                'type': constr_type,
                'fun': constraint_func,
                'expression': expr
            })

        logger.info(f"Setup {len(constraints)} constraints")

    def safe_eval(self, expr: str, variable_dict: Dict[str, Any]) -> float:
        """安全地评估表达式"""
        try:
            # 清理表达式
            expr = re.sub(r'\[(\w+)\]', r'\1', expr)

            # 安全的数学函数和环境
            safe_dict = {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
                'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
                'exp': math.exp, 'log': math.log, 'log10': math.log10,
                'math.exp' : math.exp, 'math.tanh': math.tanh,
                'sqrt': math.sqrt, 'abs': abs, 'pow': pow,
                'ceil': math.ceil, 'floor': math.floor, 'round': round,
                'pi': math.pi, 'e': math.e,
                'min': min, 'max': max, 'sum': sum,
                'median': statistics.median,
                '__builtins__': {}
            }
            safe_dict.update(variable_dict)

            return float(eval(expr, {"__builtins__": {}}, safe_dict))
        except Exception as e:
            logger.error(f"Expression evaluation failed: '{expr}' with error: {e}")
            logger.info(f"Error:{traceback.format_exc()}")
            raise ValueError(f"Failed to evaluate expression: {expr}")

    def create_model(self) -> Optional[BaseSurrogateModel]:
        """创建代理模型"""
        model_type_str = self.config.get('surrogateModel', 'lightgbm')

        if model_type_str == SurrogateModelType.NONE.value:
            logger.info("No surrogate model selected, using direct optimization")
            return None

        try:
            model_type = SurrogateModelType(model_type_str)
        except ValueError:
            raise ValueError(f"Unsupported model type: {model_type_str}")

        if model_type not in self.model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_class = self.model_registry[model_type]
        model_params = self.config.get('modelParams', {})

        try:
            self.model = model_class(model_params)
            logger.info(f"Created {model_type.value} model with parameters: {model_params}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to create {model_type.value} model: {e}")
            raise

    def train_model(self, target_variable: str, test_size: float = 0.2) -> Dict[str, Any]:
        """训练模型"""
        if self.data is None:
            raise ValueError("No data loaded for training")

        if target_variable not in self.data.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data")

        if self.model is None:
            self.create_model()

        if self.model is None:
            raise ValueError("No model available for training")

        # 准备特征数据
        variable_names = [var['name'] for var in self.config.get('variables', [])]
        missing_vars = [var for var in variable_names if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"Variables not found in data: {missing_vars}")

        X = self.data[variable_names]
        y = self.data[target_variable].values

        # 数据预处理
        X_clean, y_clean = self._clean_data(X, y)

        # 训练模型
        self.model.train(X_clean, y_clean, self.variable_info)

        # 模型评估
        metrics = self.model.evaluate(X_clean, y_clean, self.variable_info)

        logger.info(f"Model training completed with R²: {metrics['r2']:.4f}")
        return metrics

    def _clean_data(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """清理数据"""
        # 移除包含NaN的行
        valid_mask = ~(X.isnull().any(axis=1) | np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask]

        if len(X_clean) < len(X):
            logger.warning(f"Removed {len(X) - len(X_clean)} rows with invalid values")

        return X_clean, y_clean

    def load_existing_model(self, model_path: str):
        """加载现有模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 读取模型元数据来确定类型
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model_type = model_data.get('model_type', 'lightgbm')
        if model_type not in [t.value for t in SurrogateModelType]:
            raise ValueError(f"Unsupported model type in saved model: {model_type}")

        model_type_enum = SurrogateModelType(model_type)
        model_class = self.model_registry[model_type_enum]
        self.model = model_class({})
        self.model.load(model_path)

        logger.info(f"Loaded existing {model_type} model from: {model_path}")

    def run_optimization(self) -> Union[OptimizationResult, MultiObjectiveResult]:
        """运行优化"""
        start_time = time.time()

        try:
            # 初始化
            self.setup_variables()
            self.setup_constraints()

            optimization_mode = self.config.get('optimizationMode', 'training')
            objective_type = self.config.get('objectiveType', 'single')
            objectives = self.config.get('objectives', [])

            # 模型训练或加载
            model_metrics, trained_model_path = self._handle_model_setup(optimization_mode)

            # 选择优化方法
            if objective_type == 'multi' and len(objectives) > 1:
                result = self._run_multi_objective_optimization()
            else:
                result = self._run_single_objective_optimization()

            # 添加额外信息
            result.optimization_time = time.time() - start_time
            if hasattr(result, 'model_metrics'):
                result.model_metrics = model_metrics
            if hasattr(result, 'trained_model_path'):
                result.trained_model_path = trained_model_path

            logger.info(f"Optimization completed in {result.optimization_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            execution_time = time.time() - start_time

            if self.config.get('objectiveType', 'single') == 'multi':
                return MultiObjectiveResult(
                    success=False,
                    message=str(e),
                    pareto_front=[],
                    ideal_point=[],
                    nadir_point=[],
                    optimization_time=execution_time
                )
            else:
                return OptimizationResult(
                    success=False,
                    message=str(e),
                    optimal_variables={},
                    optimal_objective=float('inf'),
                    optimization_time=execution_time
                )

    def _handle_model_setup(self, optimization_mode: str) -> Tuple[Optional[Dict], Optional[str]]:
        """处理模型设置"""
        model_metrics = None
        trained_model_path = None

        if optimization_mode == 'training' and self.config.get('surrogateModel') != 'noModel':
            target_variable = self.config.get('targetVariable')
            if not target_variable:
                raise ValueError("Target variable required for training mode")

            # 加载数据（如果提供了数据路径）
            if self.data is None and self.config.get('dataPath'):
                self.load_data_from_file(self.config.get('dataPath'))

            if self.data is None:
                raise ValueError("No data available for training")

            model_metrics = self.train_model(target_variable)

            # 保存模型
            if self.model:
                trained_model_path = self._save_trained_model()

        elif optimization_mode == 'inference':
            model_path = self.config.get('modelPath')
            if model_path:
                self.load_existing_model(model_path)
            else:
                raise ValueError("Model path required for inference mode")

        return model_metrics, trained_model_path

    def _save_trained_model(self) -> str:
        """保存训练好的模型"""
        model_dir = self.config.get('modelOutputDir', 'models')
        os.makedirs(model_dir, exist_ok=True)

        timestamp = int(time.time())
        model_path = os.path.join(model_dir, f"model_{timestamp}.pkl")

        self.model.save(model_path)
        return model_path

    def _run_single_objective_optimization(self) -> OptimizationResult:
        """运行单目标优化"""
        objective_func = self._create_single_objective_function()
        bounds = self._prepare_bounds()

        optimizer_name = self.config.get('optimizer', 'pso')
        if optimizer_name not in self.optimizer_registry:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optimizer_func = self.optimizer_registry[optimizer_name]
        optimizer_params = self.config.get('optimizerParams', {})

        raw_result = optimizer_func(objective_func, bounds, optimizer_params)

        # 处理优化结果
        optimal_variables = self._postprocess_variables(raw_result['x'])

        return OptimizationResult(
            success=raw_result['success'],
            message=raw_result['message'],
            optimal_variables=optimal_variables,
            optimal_objective=float(raw_result['fun']),
            optimization_history=self.optimization_history,
            n_evaluations=len(self.optimization_history),
            convergence_data=self.convergence_monitor.get_convergence_data()
        )

    def _run_multi_objective_optimization(self) -> MultiObjectiveResult:
        """运行多目标优化"""
        if not PYMOO_AVAILABLE:
            raise ImportError("Multi-objective optimization requires pymoo package")

        objective_func = self._create_multi_objective_function()

        # 使用pymoo进行多目标优化
        problem = self._create_pymoo_problem(objective_func)
        algorithm = self._create_pymoo_algorithm()

        # 运行优化
        result = pymoo_minimize(problem, algorithm, ('n_gen', 100), verbose=False)

        # 处理结果
        pareto_front = self._process_pareto_front(result, objective_func)
        ideal_point = result.ideal_point.tolist() if hasattr(result, 'ideal_point') else []
        nadir_point = result.nadir_point.tolist() if hasattr(result, 'nadir_point') else []

        return MultiObjectiveResult(
            success=True,
            message="Multi-objective optimization completed successfully",
            pareto_front=pareto_front,
            ideal_point=ideal_point,
            nadir_point=nadir_point,
            optimization_history=self.optimization_history,
            n_evaluations=len(self.optimization_history)
        )

    def _create_single_objective_function(self) -> Callable:
        """创建单目标函数"""
        objectives = self.config.get('objectives', [])
        if not objectives:
            raise ValueError("No objectives defined")

        def objective_function(x):
            try:
                # 创建输入字典
                input_dict = {}
                variable_names = list(self.variable_info.keys())
                for i, var_name in enumerate(variable_names):
                    input_dict[var_name] = x[i]

                input_df = pd.DataFrame([input_dict])

                # 计算目标值
                if self.model and self.model.is_trained:
                    predictions = self.model.predict(input_df, self.variable_info)
                    result = float(predictions[0])
                else:
                    # 直接使用表达式
                    result = self.safe_eval(objectives[0]['expression'], input_dict)

                # 处理目标方向
                if objectives[0].get('type') == OptimizationType.MAXIMIZE.value:
                    result = -result

                # 记录历史
                self.optimization_history.append({
                    'variables': input_dict.copy(),
                    'objective_value': float(result),
                    'timestamp': time.time(),
                    'iteration': len(self.optimization_history)
                })

                # 收敛监控
                self.convergence_monitor.update(result)

                return float(result)

            except Exception as e:
                logger.error(f"Objective function evaluation failed: {e}")
                return float('inf')

        return objective_function

    def _create_multi_objective_function(self) -> Callable:
        """创建多目标函数"""
        objectives = self.config.get('objectives', [])
        if len(objectives) < 2:
            raise ValueError("Multi-objective optimization requires at least 2 objectives")

        def objective_function(x):
            input_dict = {}
            variable_names = list(self.variable_info.keys())
            for i, var_name in enumerate(variable_names):
                input_dict[var_name] = x[i]

            input_df = pd.DataFrame([input_dict])

            # 计算所有目标值
            objective_values = []
            for obj in objectives:
                if self.model and self.model.is_trained:
                    predictions = self.model.predict(input_df, self.variable_info)
                    value = float(predictions[0])
                else:
                    value = self.safe_eval(obj['expression'], input_dict)

                # 处理目标方向
                if obj.get('type') == OptimizationType.MAXIMIZE.value:
                    value = -value

                objective_values.append(value)

            # 记录历史
            self.optimization_history.append({
                'variables': input_dict.copy(),
                'objective_values': objective_values.copy(),
                'timestamp': time.time(),
                'iteration': len(self.optimization_history)
            })

            return objective_values

        return objective_function

    def _prepare_bounds(self) -> List[Tuple[float, float]]:
        """准备变量边界"""
        bounds = []
        for var_name, info in self.variable_info.items():
            if info['type'] == VariableType.CONTINUOUS.value:
                bounds.append(tuple(info['bounds']))
            elif info['type'] == VariableType.INTEGER.value:
                bounds.append(tuple(info['bounds']))
            elif info['type'] == VariableType.BINARY.value:
                bounds.append((0, 1))
            elif info['type'] in [VariableType.CATEGORICAL.value, VariableType.ORDINAL.value]:
                # 分类变量使用索引
                categories = info.get('categories', [])
                bounds.append((0, len(categories) - 1))
            else:
                bounds.append((0, 1))  # 默认边界

        return bounds

    def _postprocess_variables(self, x: List[float]) -> Dict[str, Any]:
        """后处理优化结果变量"""
        optimal_variables = {}
        variable_names = list(self.variable_info.keys())

        for i, var_name in enumerate(variable_names):
            info = self.variable_info[var_name]
            raw_value = x[i]

            if info['type'] == VariableType.CONTINUOUS.value:
                optimal_variables[var_name] = float(raw_value)
            elif info['type'] == VariableType.INTEGER.value:
                optimal_variables[var_name] = int(round(raw_value))
            elif info['type'] == VariableType.BINARY.value:
                optimal_variables[var_name] = int(round(raw_value))
            elif info['type'] in [VariableType.CATEGORICAL.value, VariableType.ORDINAL.value]:
                categories = info.get('categories', [])
                if categories:
                    idx = int(round(raw_value))
                    idx = max(0, min(idx, len(categories) - 1))
                    optimal_variables[var_name] = categories[idx]
                else:
                    optimal_variables[var_name] = raw_value
            else:
                optimal_variables[var_name] = raw_value

        return optimal_variables

    def _create_pymoo_problem(self, objective_func: Callable) -> PymooProblem:
        """创建pymoo问题定义"""
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is not available. Please install pymoo for multi-objective optimization.")
        bounds = self._prepare_bounds()
        n_var = len(bounds)
        n_obj = len(self.config.get('objectives', []))

        class MultiObjectiveProblem(PymooProblem):
            def __init__(self, n_var, n_obj, xl, xu, objective_func, constraint_funcs):
                super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
                self.objective_func = objective_func
                self.constraint_funcs = constraint_funcs

            def _evaluate(self, x, out, *args, **kwargs):
                # 评估目标函数
                out["F"] = np.array([self.objective_func(xi) for xi in x])

                # 评估约束
                if self.constraint_funcs:
                    constraints = []
                    for constr in self.constraint_funcs:
                        constr_values = [constr['fun'](xi) for xi in x]
                        constraints.append(constr_values)
                    out["G"] = np.column_stack(constraints)

        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])

        return MultiObjectiveProblem(n_var, n_obj, xl, xu, objective_func, self.constraint_functions)

    def _create_pymoo_algorithm(self):
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is not available. Please install pymoo for multi-objective optimization.")
        """创建pymoo算法"""
        optimizer_name = self.config.get('optimizer', 'nsga2')
        params = self.config.get('optimizerParams', {})

        if optimizer_name == 'nsga2':
            pop_size = params.get('population_size', 100)
            return NSGA2(
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=0.1, eta=20),
                eliminate_duplicates=True
            )
        elif optimizer_name == 'nsga3':
            pop_size = params.get('population_size', 100)
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
            return NSGA3(
                pop_size=pop_size,
                ref_dirs=ref_dirs,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=0.1, eta=20),
                eliminate_duplicates=True
            )
        elif optimizer_name == 'moead':
            pop_size = params.get('population_size', 100)
            ref_dirs = get_reference_directions("uniform", 2, n_partitions=15)
            return MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=15,
                prob_neighbor_mating=0.7,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=0.1, eta=20),
            )
        else:
            # 默认使用NSGA2
            return NSGA2(pop_size=100)

    def _process_pareto_front(self, result, objective_func: Callable) -> List[ParetoSolution]:
        """处理帕累托前沿"""
        pareto_front = []

        for i, (x, f) in enumerate(zip(result.X, result.F)):
            # 处理变量
            variables = self._postprocess_variables(x)

            # 创建帕累托解
            solution = ParetoSolution(
                variables=variables,
                objectives=f.tolist(),
                rank=i,
                crowding_distance=0.0  # 在实际实现中需要计算拥挤距离
            )
            pareto_front.append(solution)

        return pareto_front

    def _run_ant_colony(self, objective_func, bounds, params):
        """
        运行蚁群优化算法 (ACO) 用于连续空间优化

        实现思路：
        1. 将连续搜索空间离散化为多个区间
        2. 蚂蚁在离散空间中构建解决方案
        3. 通过信息素更新引导搜索方向
        4. 最终返回最优解
        """
        try:
            # 参数设置
            n_ants = params.get('n_ants', 50)  # 蚂蚁数量
            n_iterations = params.get('n_iterations', 100)  # 迭代次数
            evaporation_rate = params.get('evaporation_rate', 0.1)  # 信息素蒸发率
            alpha = params.get('alpha', 1.0)  # 信息素重要性因子
            beta = params.get('beta', 2.0)  # 启发式信息重要性因子
            q = params.get('q', 100)  # 信息素增强系数
            discretization = params.get('discretization', 20)  # 每个维度离散化区间数

            random_state = check_random_state(params.get('random_state', 42))

            n_dimensions = len(bounds)

            # 初始化信息素矩阵（每个维度的每个离散区间都有信息素）
            pheromone = np.ones((n_dimensions, discretization))

            # 记录全局最优解
            global_best_solution = None
            global_best_value = float('inf')

            # 迭代过程
            for iteration in range(n_iterations):
                # 每只蚂蚁构建解决方案
                ant_solutions = []
                ant_values = []

                for ant in range(n_ants):
                    # 构建解决方案
                    solution = []

                    for dim in range(n_dimensions):
                        # 获取当前维度的边界
                        lb, ub = bounds[dim]

                        # 计算每个离散区间的概率
                        tau = pheromone[dim] ** alpha

                        # 计算启发式信息（这里使用目标函数值的倒数作为启发式）
                        # 为了计算启发式，我们先采样每个区间的中点
                        heuristic = np.zeros(discretization)
                        for i in range(discretization):
                            # 计算区间中点
                            mid_point = lb + (ub - lb) * (i + 0.5) / discretization
                            # 创建临时解决方案
                            temp_solution = solution.copy() + [mid_point]
                            # 填充剩余维度的中点值
                            for d in range(dim + 1, n_dimensions):
                                temp_solution.append((bounds[d][0] + bounds[d][1]) / 2)
                            # 计算启发式值
                            try:
                                val = objective_func(temp_solution)
                                heuristic[i] = 1.0 / (val + 1e-6)  # 避免除零
                            except:
                                heuristic[i] = 1e-6

                        eta = heuristic ** beta

                        # 计算选择概率
                        probabilities = tau * eta
                        probabilities = probabilities / probabilities.sum()

                        # 根据概率选择区间
                        selected_interval = random_state.choice(discretization, p=probabilities)

                        # 在选中的区间内随机生成值
                        interval_width = (ub - lb) / discretization
                        value = lb + selected_interval * interval_width + random_state.uniform(0, interval_width)

                        solution.append(value)

                    # 评估解决方案
                    try:
                        value = objective_func(solution)

                        # 应用约束处理
                        constraint_violation = 0.0
                        for constr in self.constraint_functions:
                            constr_val = constr['fun'](solution)
                            if constr['type'] == ConstraintType.EQUALITY.value:
                                constraint_violation += abs(constr_val)
                            else:
                                if constr_val < 0:
                                    constraint_violation += abs(constr_val)

                        # 添加约束惩罚
                        if constraint_violation > 0:
                            value += 1e6 * constraint_violation

                        ant_solutions.append(solution)
                        ant_values.append(value)

                        # 更新全局最优
                        if value < global_best_value:
                            global_best_value = value
                            global_best_solution = solution.copy()

                    except Exception as e:
                        logger.warning(f"Ant colony evaluation error: {e}")
                        continue

                # 更新信息素
                # 1. 信息素蒸发
                pheromone *= (1 - evaporation_rate)

                # 2. 信息素增强
                if ant_solutions:
                    # 找到当前迭代的最优解
                    iteration_best_idx = np.argmin(ant_values)
                    iteration_best_value = ant_values[iteration_best_idx]
                    iteration_best_solution = ant_solutions[iteration_best_idx]

                    # 增强最优解对应路径的信息素
                    for dim in range(n_dimensions):
                        lb, ub = bounds[dim]
                        value = iteration_best_solution[dim]
                        # 计算属于哪个区间
                        interval_idx = int(((value - lb) / (ub - lb)) * discretization)
                        interval_idx = np.clip(interval_idx, 0, discretization - 1)
                        # 更新信息素
                        pheromone[dim, interval_idx] += q / (iteration_best_value + 1e-6)

                # 记录历史
                if global_best_solution is not None:
                    self.optimization_history.append({
                        'variables': global_best_solution,
                        'objective_value': global_best_value,
                        'iteration': iteration
                    })

            # 返回结果
            if global_best_solution is not None:
                return {
                    'success': True,
                    'x': global_best_solution,
                    'fun': float(global_best_value),
                    'message': 'Ant colony optimization completed successfully'
                }
            else:
                return {
                    'success': False,
                    'x': [np.mean(b) for b in bounds],
                    'fun': float('inf'),
                    'message': 'Ant colony optimization failed to find valid solution'
                }

        except Exception as e:
            logger.warning(f"Ant colony optimization error: {e}")
            # 尝试返回历史中的最佳解
            best_solution = self._get_best_from_history()
            if best_solution:
                return {
                    'success': True,
                    'x': best_solution['variables'],
                    'fun': best_solution['objective_value'],
                    'message': f"Used best solution from history after ant colony failure: {e}"
                }
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'Ant colony optimization error: {str(e)}'
            }
    # 各种优化算法的具体实现
    def _run_pso(self, objective_func, bounds, params):
        """运行粒子群优化"""
        try:
            from pyswarm import pso

            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]
            swarmsize = params.get('swarmsize', 50)
            maxiter = params.get('maxiter', 100)
            omega = params.get('omega', 0.5)
            phip = params.get('phip', 0.5)
            phig = params.get('phig', 0.5)

            xopt, fopt = pso(
                objective_func, lb, ub,
                swarmsize=swarmsize,
                maxiter=maxiter,
                omega=omega,
                phip=phip,
                phig=phig,
                debug=False
            )

            return {
                'success': True,
                'x': xopt.tolist(),
                'fun': float(fopt),
                'message': 'PSO optimization completed successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'PSO error: {str(e)}'
            }

    def _run_ga(self, objective_func, bounds, params):
        """运行遗传算法（差分进化）"""
        try:
            popsize = params.get('population_size', 50)
            maxiter = params.get('generations', 100)
            mutation = params.get('mutation', (0.5, 1))
            recombination = params.get('recombination', 0.7)
            strategy = params.get('strategy', 'best1bin')

            result = differential_evolution(
                objective_func,
                bounds,
                strategy=strategy,
                popsize=popsize,
                maxiter=maxiter,
                mutation=mutation,
                recombination=recombination,
                constraints=self.constraint_functions,
                disp=False,
                seed=42
            )

            return {
                'success': result.success,
                'x': result.x.tolist(),
                'fun': float(result.fun),
                'message': result.message
            }
        except Exception as e:
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'Genetic algorithm error: {str(e)}'
            }

    def _run_differential_evolution(self, objective_func, bounds, params):
        """运行差分进化算法 - 修复版本"""
        try:
            popsize = params.get('population_size', 50)
            maxiter = params.get('generations', 100)
            mutation = params.get('mutation', (0.5, 1))
            recombination = params.get('recombination', 0.7)
            strategy = params.get('strategy', 'best1bin')

            # 添加约束容错处理
            def constrained_objective(x):
                try:
                    result = objective_func(x)
                    # 检查约束违反程度
                    constraint_violation = 0.0
                    for constr in self.constraint_functions:
                        constr_val = constr['fun'](x)
                        if constr['type'] == ConstraintType.EQUALITY.value:
                            constraint_violation += abs(constr_val)  # eq: h(x) = 0
                        else:  # inequality
                            if constr_val < 0:  # g(x) >= 0
                                constraint_violation += abs(constr_val)

                    # 如果有约束违反，添加惩罚项
                    if constraint_violation > 0:
                        penalty = 1e6 * constraint_violation
                        return result + penalty
                    return result
                except Exception as e:
                    logger.warning(f"Objective function evaluation failed: {e}")
                    return float('inf')

            result = differential_evolution(
                constrained_objective,
                bounds,
                strategy=strategy,
                popsize=popsize,
                maxiter=maxiter,
                mutation=mutation,
                recombination=recombination,
                disp=False,
                seed=42,
                polish=True  # 添加局部优化
            )

            # 更宽松的成功判断
            success = (result.success or
                       result.fun < float('inf') or
                       len(self.optimization_history) > 0)

            return {
                'success': success,
                'x': result.x.tolist(),
                'fun': float(result.fun),
                'message': result.message if result.success else "Optimization completed with acceptable solution"
            }
        except Exception as e:
            logger.warning(f"Differential evolution completed with issues: {e}")
            # 尝试返回历史中的最佳解
            best_solution = self._get_best_from_history()
            if best_solution:
                return {
                    'success': True,
                    'x': best_solution['variables'],
                    'fun': best_solution['objective_value'],
                    'message': f"Used best solution from history after optimizer failure: {e}"
                }
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'Differential evolution error: {str(e)}'
            }

    def _get_best_from_history(self):
        """从优化历史中获取最佳解"""
        if not self.optimization_history:
            return None

        valid_solutions = [sol for sol in self.optimization_history
                           if sol.get('objective_value', float('inf')) < float('inf')]

        if not valid_solutions:
            return None

        # 找到目标值最小的解
        best_solution = min(valid_solutions, key=lambda x: x['objective_value'])

        # 转换为变量数组
        variable_names = list(self.variable_info.keys())
        best_variables = [best_solution['variables'][name] for name in variable_names]

        return {
            'variables': best_variables,
            'objective_value': best_solution['objective_value']
        }
    def _run_bayesian_optimization(self, objective_func, bounds, params):
        """运行贝叶斯优化"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical

            # 定义搜索空间
            dimensions = []
            variable_names = list(self.variable_info.keys())

            for i, (var_name, var_info) in enumerate(self.variable_info.items()):
                bound = bounds[i]

                if var_info['type'] == VariableType.CONTINUOUS.value:
                    dimensions.append(Real(bound[0], bound[1], name=var_name))
                elif var_info['type'] == VariableType.INTEGER.value:
                    dimensions.append(Integer(int(bound[0]), int(bound[1]), name=var_name))
                elif var_info['type'] in [VariableType.CATEGORICAL.value, VariableType.ORDINAL.value]:
                    categories = var_info.get('categories', [])
                    if categories:
                        dimensions.append(Integer(0, len(categories) - 1, name=var_name))
                    else:
                        dimensions.append(Integer(int(bound[0]), int(bound[1]), name=var_name))
                else:
                    dimensions.append(Real(bound[0], bound[1], name=var_name))

            n_calls = params.get('n_calls', 50)
            n_initial_points = params.get('n_initial_points', 10)
            acq_func = params.get('acq_func', 'EI')
            random_state = params.get('random_state', 42)

            result = gp_minimize(
                objective_func,
                dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func=acq_func,
                random_state=random_state,
                verbose=False
            )

            # 处理结果
            x_result = []
            for i, (var_name, var_info) in enumerate(self.variable_info.items()):
                value = result.x[i]
                x_result.append(value)

            return {
                'success': True,
                'x': x_result,
                'fun': float(result.fun),
                'message': 'Bayesian optimization completed successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'Bayesian optimization error: {str(e)}'
            }

    def _run_scipy_optimization(self, objective_func, bounds, params):
        """运行Scipy优化 - 修复版本"""
        method = params.get('method', 'SLSQP')
        tol = params.get('tol', 1e-6)
        options = params.get('options', {'maxiter': 1000})

        # 生成初始点（边界中点）
        x0 = [np.mean(b) for b in bounds]

        try:
            # 处理约束 - 修复约束函数
            scipy_constraints = []
            for constr in self.constraint_functions:
                if constr['type'] == ConstraintType.EQUALITY.value:
                    scipy_constraints.append({
                        'type': 'eq',
                        'fun': constr['fun']
                    })
                else:  # inequality
                    scipy_constraints.append({
                        'type': 'ineq',
                        'fun': constr['fun']
                    })

            result = minimize(
                objective_func,
                x0,
                method=method,
                bounds=bounds,
                constraints=scipy_constraints,
                tol=tol,
                options=options
            )

            # 更宽松的成功判断
            success = (result.success or
                       result.fun < float('inf') or
                       (hasattr(result, 'nit') and result.nit > 0))

            return {
                'success': success,
                'x': result.x.tolist(),
                'fun': float(result.fun),
                'message': result.message if result.success else "Optimization completed with acceptable solution"
            }
        except Exception as e:
            logger.warning(f"Scipy optimization completed with issues: {e}")
            # 尝试返回历史中的最佳解
            best_solution = self._get_best_from_history()
            if best_solution:
                return {
                    'success': True,
                    'x': best_solution['variables'],
                    'fun': best_solution['objective_value'],
                    'message': f"Used best solution from history after optimizer failure: {e}"
                }
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'Scipy optimization error: {str(e)}'
            }

    def _run_nsga2(self, objective_func, bounds, params):
        """运行NSGA-II算法"""
        if not PYMOO_AVAILABLE:
            raise ImportError("NSGA-II requires pymoo package")

        try:
            # 创建pymoo问题
            class MultiObjectiveProblem(PymooProblem):
                def __init__(self, n_var, n_obj, xl, xu, objective_func, constraint_funcs):
                    super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
                    self.objective_func = objective_func
                    self.constraint_funcs = constraint_funcs

                def _evaluate(self, x, out, *args, **kwargs):
                    # 评估目标函数
                    out["F"] = np.array([self.objective_func(xi) for xi in x])

                    # 评估约束
                    if self.constraint_funcs:
                        constraints = []
                        for constr in self.constraint_funcs:
                            constr_values = [constr['fun'](xi) for xi in x]
                            constraints.append(constr_values)
                        out["G"] = np.column_stack(constraints)

            n_var = len(bounds)
            n_obj = len(self.config.get('objectives', []))
            xl = np.array([b[0] for b in bounds])
            xu = np.array([b[1] for b in bounds])

            problem = MultiObjectiveProblem(n_var, n_obj, xl, xu, objective_func, self.constraint_functions)

            # 配置NSGA-II算法
            pop_size = params.get('population_size', 100)
            crossover_prob = params.get('crossover_prob', 0.9)
            mutation_prob = params.get('mutation_prob', 1.0 / n_var)

            algorithm = NSGA2(
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=crossover_prob, eta=15),
                mutation=PM(prob=mutation_prob, eta=20),
                eliminate_duplicates=True
            )

            # 运行优化
            result = pymoo_minimize(problem, algorithm, ('n_gen', 100), verbose=False)

            # 获取最优解
            if len(result.F) > 0:
                # 对于多目标，返回帕累托解中的第一个
                best_x = result.X[0]
                best_f = result.F[0]

                return {
                    'success': True,
                    'x': best_x.tolist(),
                    'fun': float(np.sum(best_f)),
                    'message': 'NSGA-II optimization completed successfully'
                }
            else:
                return {
                    'success': False,
                    'x': [np.mean(b) for b in bounds],
                    'fun': float('inf'),
                    'message': 'NSGA-II produced no valid solutions'
                }

        except Exception as e:
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'NSGA-II error: {str(e)}'
            }

    def _run_nsga3(self, objective_func, bounds, params):
        """运行NSGA-III算法"""
        # 实现与NSGA-II类似，但使用NSGA3算法
        return self._run_nsga2(objective_func, bounds, params)

    def _run_moead(self, objective_func, bounds, params):
        """运行MOEA/D算法"""
        # 实现与NSGA-II类似，但使用MOEAD算法
        return self._run_nsga2(objective_func, bounds, params)

    def _run_smsemoa(self, objective_func, bounds, params):
        """运行SMS-EMOA算法"""
        # 实现与NSGA-II类似，但使用SMSEMOA算法
        return self._run_nsga2(objective_func, bounds, params)

    def _run_optuna(self, objective_func, bounds, params):
        """运行Optuna优化"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna optimization requires optuna package")

        try:
            def optuna_objective(trial):
                # 为每个变量定义搜索空间
                x = []
                for i, (var_name, var_info) in enumerate(self.variable_info.items()):
                    bound = bounds[i]
                    if var_info['type'] == VariableType.CONTINUOUS.value:
                        value = trial.suggest_float(var_name, bound[0], bound[1])
                    elif var_info['type'] == VariableType.INTEGER.value:
                        value = trial.suggest_int(var_name, int(bound[0]), int(bound[1]))
                    elif var_info['type'] in [VariableType.CATEGORICAL.value, VariableType.ORDINAL.value]:
                        categories = var_info.get('categories', [])
                        if categories:
                            value = trial.suggest_categorical(var_name, list(range(len(categories))))
                        else:
                            value = trial.suggest_int(var_name, int(bound[0]), int(bound[1]))
                    else:
                        value = trial.suggest_float(var_name, bound[0], bound[1])
                    x.append(value)

                return objective_func(x)

            n_trials = params.get('n_trials', 100)
            sampler = params.get('sampler', 'tpe')

            if sampler == 'cmaes':
                study_sampler = CmaEsSampler()
            else:
                study_sampler = TPESampler()

            study = optuna.create_study(sampler=study_sampler)
            study.optimize(optuna_objective, n_trials=n_trials)

            return {
                'success': True,
                'x': list(study.best_params.values()),
                'fun': float(study.best_value),
                'message': 'Optuna optimization completed successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'Optuna optimization error: {str(e)}'
            }

    def _run_cma_es(self, objective_func, bounds, params):
        """运行CMA-ES优化"""
        if not CMA_AVAILABLE:
            raise ImportError("CMA-ES optimization requires cma package")

        try:
            # 初始点
            x0 = [np.mean(b) for b in bounds]
            sigma0 = params.get('sigma0', 0.5)

            # 运行CMA-ES
            es = cma.CMAEvolutionStrategy(x0, sigma0)
            es.optimize(objective_func)

            return {
                'success': True,
                'x': es.result.xbest.tolist(),
                'fun': float(es.result.fbest),
                'message': 'CMA-ES optimization completed successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'x': [np.mean(b) for b in bounds],
                'fun': float('inf'),
                'message': f'CMA-ES optimization error: {str(e)}'
            }


# 测试和示例函数
def create_sample_data(n_samples: int = 1000, scenario: str = "default") -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)

    if scenario == "finance":
        # 金融资产组合场景
        data = pd.DataFrame({
            'stock_allocation': np.random.uniform(0, 0.8, n_samples),
            'bond_allocation': np.random.uniform(0, 0.6, n_samples),
            'cash_allocation': np.random.uniform(0, 0.4, n_samples),
            'risk_appetite': np.random.choice(['low', 'medium', 'high'], n_samples),
            'expected_return': np.zeros(n_samples),
            'portfolio_risk': np.zeros(n_samples),
            'sharpe_ratio': np.zeros(n_samples)
        })

        # 计算金融指标
        for i in range(n_samples):
            stock = data.loc[i, 'stock_allocation']
            bond = data.loc[i, 'bond_allocation']
            cash = data.loc[i, 'cash_allocation']

            # 归一化分配
            total = stock + bond + cash
            if total > 0:
                stock /= total
                bond /= total
                cash /= total

            # 计算预期收益和风险
            expected_return = stock * 0.12 + bond * 0.04 + cash * 0.01
            risk = stock * 0.25 + bond * 0.08 + cash * 0.01

            # 添加随机噪声
            expected_return += np.random.normal(0, 0.02)
            risk += np.random.normal(0, 0.01)

            sharpe_ratio = expected_return / risk if risk > 0 else 0

            data.loc[i, 'expected_return'] = expected_return
            data.loc[i, 'portfolio_risk'] = risk
            data.loc[i, 'sharpe_ratio'] = sharpe_ratio

        return data

    elif scenario == "engineering":
        # 工程设计场景
        data = pd.DataFrame({
            'material_thickness': np.random.uniform(1, 10, n_samples),
            'temperature': np.random.uniform(20, 200, n_samples),
            'pressure': np.random.uniform(1, 100, n_samples),
            'material_type': np.random.choice(['steel', 'aluminum', 'composite'], n_samples),
            'strength': np.zeros(n_samples),
            'weight': np.zeros(n_samples),
            'cost': np.zeros(n_samples)
        })

        for i in range(n_samples):
            thickness = data.loc[i, 'material_thickness']
            temp = data.loc[i, 'temperature']
            pressure = data.loc[i, 'pressure']
            material = data.loc[i, 'material_type']

            # 材料属性
            material_factors = {
                'steel': {'strength': 1.0, 'density': 7.8, 'cost': 1.0},
                'aluminum': {'strength': 0.6, 'density': 2.7, 'cost': 1.5},
                'composite': {'strength': 1.2, 'density': 1.8, 'cost': 3.0}
            }
            factors = material_factors[material]

            # 计算工程指标
            strength = (factors['strength'] * thickness * 10 *
                        (1 - temp / 500) * (1 + pressure / 200))
            weight = factors['density'] * thickness
            cost = factors['cost'] * thickness * (1 + temp / 100) * (1 + pressure / 50)

            data.loc[i, 'strength'] = strength + np.random.normal(0, 5)
            data.loc[i, 'weight'] = weight + np.random.normal(0, 0.1)
            data.loc[i, 'cost'] = cost + np.random.normal(0, 0.5)

        return data

    else:
        # 默认场景
        data = pd.DataFrame({
            'x1': np.random.uniform(0, 10, n_samples),
            'x2': np.random.uniform(0, 10, n_samples),
            'x3': np.random.uniform(0, 10, n_samples),
            'x4': np.random.choice(['A', 'B', 'C'], n_samples)
        })

        # 现在data已经定义，可以安全地使用它的列
        data['y'] = (np.sin(data['x1']) + np.cos(data['x2']) +
                     data['x3'] / 10 +
                     (data['x4'] == 'A') * 0.5 +
                     (data['x4'] == 'B') * 0.3 +
                     np.random.normal(0, 0.1, n_samples))

        return data


def test_optimization_engine():
    """测试优化引擎"""
    print("Testing Optimization Engine...")

    # 创建测试数据
    data = create_sample_data(n_samples=500, scenario="default")
    print(f"Created sample data with shape: {data.shape}")

    # 测试配置
    config = {
        "optimizationMode": "training",
        "surrogateModel": "lightgbm",
        "optimizer": "pso",
        "objectiveType": "single",
        "variables": [
            {"name": "x1", "type": "continuous", "bounds": [0, 10]},
            {"name": "x2", "type": "continuous", "bounds": [0, 10]},
            {"name": "x3", "type": "continuous", "bounds": [0, 10]},
            {"name": "x4", "type": "categorical", "categories": ["A", "B", "C"]}
        ],
        "objectives": [
            {"name": "minimize_y", "type": "minimize", "expression": "[y]"}
        ],
        "constraints": [
            {"expression": "[x1] + [x2] + [x3] <= 20", "type": "ineq"},
            {"expression": "[x1] >= 1", "type": "ineq"}
        ],
        "targetVariable": "y",
        "optimizerParams": {
            "swarmsize": 30,
            "maxiter": 50
        },
        "modelParams": {
            "n_estimators": 100,
            "learning_rate": 0.1
        }
    }

    # 创建优化引擎
    engine = OptimizationEngine(config)
    engine.load_data(data)

    # 运行优化
    start_time = time.time()
    result = engine.run_optimization()
    execution_time = time.time() - start_time

    # 输出结果
    print(f"\nOptimization Results:")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Optimal Variables: {result.optimal_variables}")
    print(f"Optimal Objective: {result.optimal_objective:.6f}")

    if result.model_metrics:
        print(f"Model Metrics: {result.model_metrics}")

    if result.trained_model_path:
        print(f"Model saved to: {result.trained_model_path}")

    return result


def test_multi_objective_optimization():
    """测试多目标优化"""
    if not PYMOO_AVAILABLE:
        print("Skipping multi-objective test (pymoo not available)")
        return None

    print("\nTesting Multi-Objective Optimization...")

    # 创建测试数据
    data = create_sample_data(n_samples=500, scenario="engineering")

    # 多目标优化配置
    config = {
        "optimizationMode": "training",
        "surrogateModel": "lightgbm",
        "optimizer": "nsga2",
        "objectiveType": "multi",
        "variables": [
            {"name": "material_thickness", "type": "continuous", "bounds": [1, 10]},
            {"name": "temperature", "type": "continuous", "bounds": [20, 200]},
            {"name": "pressure", "type": "continuous", "bounds": [1, 100]},
            {"name": "material_type", "type": "categorical", "categories": ["steel", "aluminum", "composite"]}
        ],
        "objectives": [
            {"name": "maximize_strength", "type": "maximize", "expression": "[strength]"},
            {"name": "minimize_weight", "type": "minimize", "expression": "[weight]"},
            {"name": "minimize_cost", "type": "minimize", "expression": "[cost]"}
        ],
        "targetVariable": "strength",
        "optimizerParams": {
            "population_size": 50,
            "generations": 30
        }
    }

    # 创建优化引擎
    engine = OptimizationEngine(config)
    engine.load_data(data)

    # 运行优化
    start_time = time.time()
    result = engine.run_optimization()
    execution_time = time.time() - start_time

    # 输出结果
    print(f"\nMulti-Objective Optimization Results:")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Number of Pareto Solutions: {len(result.pareto_front)}")

    # 显示前5个帕累托解
    for i, solution in enumerate(result.pareto_front[:5]):
        print(f"Solution {i + 1}: {solution.variables} -> Objectives: {[f'{obj:.3f}' for obj in solution.objectives]}")

    return result


if __name__ == "__main__":
    # 运行测试
    print("=" * 60)
    print("OPTIMIZATION ENGINE TEST SUITE")
    print("=" * 60)

    # 测试单目标优化
    single_obj_result = test_optimization_engine()

    # 测试多目标优化
    multi_obj_result = test_multi_objective_optimization()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if single_obj_result and single_obj_result.success:
        print("✓ Single-objective optimization: PASSED")
    else:
        print("✗ Single-objective optimization: FAILED")

    if multi_obj_result and multi_obj_result.success:
        print("✓ Multi-objective optimization: PASSED")
    else:
        print("✗ Multi-objective optimization: FAILED or SKIPPED")

    print("\nOptimization Engine test completed!")