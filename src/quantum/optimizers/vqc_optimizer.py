
"""VQC training loop with multiple optimizer support."""

import time
import numpy as np
from typing import Callable, Optional, List, Dict
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA, COBYLA, NELDER_MEAD, Optimizer
import os


class VQCTrainer:
    """
    Trains a Variational Quantum Classifier.
    
    Uses expectation-based loss with configurable optimizer.
    Supports training curve recording and checkpointing.
    """
    
    def __init__(
        self,
        estimator,
        circuit: QuantumCircuit,
        input_params: list,
        weight_params: list,
        observables: list,
        optimizer_name: str = 'SPSA',
        optimizer_kwargs: dict = None,
        shots: int = 1024,
        seed: int = 42,
        label: str = 'VQC',
    ):
        """
        Initialize the VQC trainer.
        
        Args:
            estimator: Qiskit Estimator primitive
            circuit: Parameterized ansatz circuit
            input_params: Input (feature) parameters
            weight_params: Trainable weight parameters
            observables: Observables to measure (one per class for multi-class)
            optimizer_name: Optimizer type ('SPSA', 'COBYLA', 'NM' for Nelder-Mead)
            optimizer_kwargs: Additional optimizer arguments
            shots: Number of shots
            seed: Random seed
            label: Label for logging
        """
        self.estimator = estimator
        self.circuit = circuit
        self.input_params = input_params
        self.weight_params = weight_params
        self.observables = observables
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.shots = shots
        self.seed = seed
        self.label = label
        
        self._training_history = []
        self._best_weights = None
        self._best_loss = float('inf')
        self._n_evaluations = 0
    
    def _create_optimizer(self) -> Optimizer:
        """Create the optimizer instance."""
        if self.optimizer_name == 'SPSA':
            return SPSA(
                maxiter=self.optimizer_kwargs.get('maxiter', 50),
                learning_rate=self.optimizer_kwargs.get('learning_rate', 0.1),
                perturbation=self.optimizer_kwargs.get('perturbation', 0.01),
                averaging=self.optimizer_kwargs.get('averaging', 1),
            )
        elif self.optimizer_name == 'COBYLA':
            return COBYLA(
                maxiter=self.optimizer_kwargs.get('maxiter', 100),
                rhobeg=self.optimizer_kwargs.get('rhobeg', 0.5),
            )
        elif self.optimizer_name == 'NM':
            return NELDER_MEAD(
                maxiter=self.optimizer_kwargs.get('maxiter', 100),
                initial_step_size=self.optimizer_kwargs.get('initial_step_size', 0.5),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _loss_function(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the classification loss.
        
        Uses mean squared error between predicted expectation values and targets.
        For multi-class: one-vs-rest approach with multiple observables.
        """
        self._n_evaluations += 1
        
        # Evaluate circuit for all samples
        # For each sample, bind input params and weight params
        losses = []
        for x, y_true in zip(X, y):
            # Bind parameters
            param_dict = {}
            for inp, val in zip(self.input_params, x):
                param_dict[inp] = val
            for w, val in zip(self.weight_params, weights):
                param_dict[w] = val
            
            # Run estimator
            try:
                job = self.estimator.run(
                    [(self.circuit, obs, param_dict) for obs in self.observables]
                )
                result = job.result()
                
                # Get expectation values
                exp_vals = [r.data.evs[0] for r in result]
                
                # Convert to class prediction (argmax of expectations)
                # For binary: sign of expectation
                # For multi-class: one-hot decode
                if len(self.observables) == 1:
                    # Binary classification
                    pred = 1 if exp_vals[0] > 0 else 0
                    loss = (pred - y_true) ** 2
                else:
                    # Multi-class: use argmax
                    pred = int(np.argmax(exp_vals))
                    loss = (pred - y_true) ** 2
            except Exception as e:
                # Fallback on error
                loss = 1.0
            
            losses.append(loss)
        
        return float(np.mean(losses))
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iterations: int = 50,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Train the VQC.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_iterations: Number of optimizer iterations
            callback: Optional callback function called each iteration
        
        Returns:
            Dictionary with training info, best weights, history
        """
        self._training_history = []
        self._n_evaluations = 0
        
        optimizer = self._create_optimizer()
        
        initial_weights = np.random.randn(len(self.weight_params)) * 0.1
        
        def objective(weights):
            loss = self._loss_function(weights, X_train, y_train)
            self._training_history.append({
                'iteration': len(self._training_history),
                'loss': loss,
                'weights': weights.tolist(),
            })
            if callback:
                callback(len(self._training_history) - 1, loss, weights)
            return loss
        
        start_time = time.time()
        
        if self.optimizer_name == 'SPSA':
            result = optimizer.minimize(
                function=objective,
                x0=initial_weights,
                niter=n_iterations,
            )
        elif self.optimizer_name == 'COBYLA':
            result = optimizer.minimize(
                function=objective,
                x0=initial_weights,
            )
        else:
            result = optimizer.minimize(
                function=objective,
                x0=initial_weights,
            )
        
        training_time = time.time() - start_time
        
        self._best_weights = result.x
        self._best_loss = result.fun
        
        return {
            'best_weights': result.x.tolist(),
            'best_loss': float(result.fun),
            'training_time': training_time,
            'n_evaluations': self._n_evaluations,
            'history': self._training_history,
            'optimizer_result': {
                'fun': float(result.fun),
                'nit': getattr(result, 'nit', None),
                'nfev': getattr(result, 'nfev', None),
            },
        }
    
    def predict(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Make predictions using trained weights.
        
        Args:
            weights: Trained weight parameters
            X: Feature vectors
            y_true: True labels (if provided, compute metrics)
        
        Returns:
            Dictionary with predictions and metrics
        """
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        
        predictions = []
        
        for x in X:
            param_dict = {}
            for inp, val in zip(self.input_params, x):
                param_dict[inp] = val
            for w, val in zip(self.weight_params, weights):
                param_dict[w] = val
            
            try:
                job = self.estimator.run(
                    [(self.circuit, obs, param_dict) for obs in self.observables]
                )
                result = job.result()
                exp_vals = [r.data.evs[0] for r in result]
                
                if len(self.observables) == 1:
                    pred = 1 if exp_vals[0] > 0 else 0
                else:
                    pred = int(np.argmax(exp_vals))
                
                predictions.append(pred)
            except Exception:
                predictions.append(0)
        
        result = {
            'predictions': predictions,
            'n_trainable_params': len(self.weight_params),
            'n_circuit_evaluations': len(X) * len(self.observables),
        }
        
        if y_true is not None:
            result['metrics'] = {
                'accuracy': float(accuracy_score(y_true, predictions)),
                'balanced_accuracy': float(balanced_accuracy_score(y_true, predictions)),
                'macro_f1': float(f1_score(y_true, predictions, average='macro')),
                'precision': float(precision_score(y_true, predictions, average='macro', zero_division=0)),
                'recall': float(recall_score(y_true, predictions, average='macro')),
                'confusion_matrix': confusion_matrix(y_true, predictions).tolist(),
            }
        
        return result
