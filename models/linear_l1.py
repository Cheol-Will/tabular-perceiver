import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import optuna
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Optional, Tuple, Dict, List

from torch_frame import Metric, TaskType, TensorFrame, stype

DEFAULT_METRIC = {
    TaskType.REGRESSION: Metric.RMSE,
    TaskType.BINARY_CLASSIFICATION: Metric.ROCAUC,
    TaskType.MULTICLASS_CLASSIFICATION: Metric.ACCURACY,
}

class LinearL1:
    """L1-regularized regression/logistic model with Optuna tuning."""
    def __init__(
        self,
        task_type: TaskType,
        num_classes: Optional[int] = None,
        metric: Optional[Metric] = None,
    ):
        self.task_type = task_type
        self._num_classes = num_classes
        self._is_fitted = False
        self.metric = DEFAULT_METRIC[task_type]
        if metric is not None:
            if metric.supports_task_type(task_type):
                self.metric = metric
            else:
                raise ValueError(
                    f"{task_type} does not support {metric}. "
                    f"Choose from {task_type.supported_metrics}."
                )
        self.params: Dict[str, float] = {}
        self.columns: List[str] = []

    def _to_linear_input(
        self,
        tf: TensorFrame,
        columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print(f"input columns: {columns}")
        tf = tf.cpu()
        y = tf.y.numpy() if tf.y is not None else None
        dfs: List[pd.DataFrame] = []
        cat_idxs: List[np.ndarray] = []
        offset = 0
        # build raw features
        if stype.categorical in tf.feat_dict:
            arr = tf.feat_dict[stype.categorical].numpy()
            cols = np.arange(offset, offset + arr.shape[1])
            # print(f"debug arr: {arr}")
            # print(f"cols: {cols}")
            # print(f"Added Df: ")
            print(pd.DataFrame(arr, columns=cols.astype(str)))
            dfs.append(pd.DataFrame(arr, columns=cols.astype(str)))
            cat_idxs.append(cols)
            offset += arr.shape[1]
        if stype.numerical in tf.feat_dict:
            arr = tf.feat_dict[stype.numerical].numpy()
            cols = np.arange(offset, offset + arr.shape[1])
            dfs.append(pd.DataFrame(arr, columns=cols.astype(str)))
            offset += arr.shape[1]
        if stype.embedding in tf.feat_dict:
            emb = tf.feat_dict[stype.embedding]
            arr = emb.values.view(emb.size(0), -1).numpy()
            cols = np.arange(offset, offset + arr.shape[1])
            dfs.append(pd.DataFrame(arr, columns=cols.astype(str)))
            offset += arr.shape[1]
        if not dfs:
            raise ValueError("Input TensorFrame is empty.")
        df = pd.concat(dfs, axis=1)
        # one-hot encode categoricals
        if cat_idxs:
            cols_to_dummy = [str(c) for arr in cat_idxs for c in arr]
            df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)
            # print("Debug: ")
            # print(f"df with dummies {df.head()}")   
            # print(df.shape)
        # align columns
        # filter only categories(dummy-columns) seen during training.
        if columns is not None:
            df = df.reindex(columns=columns, fill_value=0)
            # print("Debug: ")
            # print(f"df after reindexing {df.head()}")   
            # print(df.shape)
        # return updated columns
        return df.values, y, df.columns.tolist()

    def compute_metric(self, target: Tensor, pred: Tensor) -> float:
        y_true = target.cpu().numpy()
        y_pred = pred.cpu().numpy()
        if self.metric == Metric.RMSE:
            return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        elif self.metric == Metric.MAE:
            return float(np.mean(np.abs(y_pred - y_true)))
        elif self.metric == Metric.ROCAUC:
            return float(roc_auc_score(y_true, y_pred))
        elif self.metric == Metric.ACCURACY:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                y_pred = (y_pred > 0.5).astype(int)
            return float(accuracy_score(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported metric {self.metric}")

    def _objective(
        self,
        trial: optuna.Trial,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
    ) -> float:
        if self.task_type == TaskType.REGRESSION:
            alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
            model = Lasso(alpha=alpha, max_iter=10000)
        else:
            C = trial.suggest_float('C', 1e-4, 10.0, log=True)
            model = LogisticRegression(
                penalty='l1', C=C, solver='saga', max_iter=10000,
                multi_class='auto'
            )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_x)
        X_val = scaler.transform(val_x)
        model.fit(X_tr, train_y)
        if self.task_type == TaskType.REGRESSION:
            preds = model.predict(X_val)
        else:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                preds = model.predict_proba(X_val)[:, 1]
            else:
                preds = model.predict(X_val)
        return self.compute_metric(torch.from_numpy(val_y), torch.from_numpy(preds))

    def tune(
        self,
        tf_train: TensorFrame,
        tf_val: TensorFrame,
        num_trials: int = 50,
    ):
        # prepare train/val with consistent columns
        train_x, train_y, cols = self._to_linear_input(tf_train, columns=None)
        print(f"Cat features: {cols}")
        val_x, val_y, _ = self._to_linear_input(tf_val, columns=cols)
        self.columns = cols
        direction = 'minimize' if self.metric in (Metric.RMSE, Metric.MAE) else 'maximize'
        study = optuna.create_study(direction=direction)
        study.optimize(
            lambda tr: self._objective(tr, train_x, train_y, val_x, val_y),
            n_trials=num_trials
        )
        self.params = study.best_params
        # final model
        if self.task_type == TaskType.REGRESSION:
            self.model = Lasso(alpha=self.params['alpha'], max_iter=10000)
        else:
            self.model = LogisticRegression(
                penalty='l1', C=self.params['C'], solver='saga', max_iter=10000,
                multi_class='auto'
            )
        self.scaler = StandardScaler()
        X_full, y_full, _ = self._to_linear_input(tf_train, columns=self.columns)
        X_full_scaled = self.scaler.fit_transform(X_full)
        self.model.fit(X_full_scaled, y_full)
        self._is_fitted = True

    def predict(self, tf_test: TensorFrame) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call tune() first.")
        X_test, _, _ = self._to_linear_input(tf_test, columns=self.columns)
        X_scaled = self.scaler.transform(X_test)
        if self.task_type == TaskType.REGRESSION:
            preds = self.model.predict(X_scaled)
        else:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                preds = self.model.predict_proba(X_scaled)[:, 1]
            else:
                preds = self.model.predict(X_scaled)
        return torch.from_numpy(preds).to(tf_test.device)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'scaler': self.scaler, 'model': self.model, 'params': self.params, 'columns': self.columns}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.model = data['model']
        self.params = data.get('params', {})
        self.columns = data.get('columns', [])
        self._is_fitted = True

    def cross_validation(self, tf_train: TensorFrame, tf_test: TensorFrame, seed: int):
        """Grid Search with Cross Validation."""
        from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold

        def compute_metric(clf_in, X, y):
            """Helper function for calculating metric"""
            p = np.argmax(clf_in.predict_proba(X), axis=1)
            metric_score = np.sum(p == np.array(y)) / p.shape[0]
            return metric_score

        parameters = {
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10.0,],
        }

        # define model
        estimator = LogisticRegression(
            penalty='l1', 
            solver='saga', 
            max_iter=10000, 
            # multi_class='auto'
        )

        # convert input tensorframe into pandas data frame
        train_x, train_y, cols = self._to_linear_input(tf_train)
        test_x, test_y, _ = self._to_linear_input(tf_test, cols)

        # create KFold
        folds = 5
        inner_cv = KFold(n_splits=folds, shuffle=True, random_state=seed)

        # create GridSearchCV and fit
        clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv, scoring="accuracy", n_jobs=40, verbose=0)
        clf.fit(train_x, train_y)

        # test with best hyperparameters        
        score_train = compute_metric(clf, train_x, train_y)
        score_test = compute_metric(clf, test_x, test_y)
        return score_train, score_test, clf.best_params_