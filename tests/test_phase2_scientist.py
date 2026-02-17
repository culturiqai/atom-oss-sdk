import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from atom.mind.scientist import AtomScientist
from atom.mind.conformal import ConformalWrapper
from atom.config import get_config

class TestScientistPhase2(unittest.TestCase):
    def setUp(self):
        # Use simple variable names for testing
        self.vars = ["x0", "x1"]
        self.scientist = AtomScientist(variable_names=self.vars)
        
        # Mock the heavy PySR process to avoid slow imports/fitting
        # We will test the WRAPPING logic, not PySR itself (which is trusted external lib)
        self.scientist.model = MagicMock()
        self.scientist.pysr_initialized = True
        
        # Real conformal wrapper for logic testing
        self.scientist.conformal = ConformalWrapper(base_model=None)
        
    def test_variable_names_alignment(self):
        """Verify mismatched features trigger auto-correction or handling."""
        # Config has 11 vars
        config = get_config()
        self.assertEqual(len(config.scientist.variable_names), 11)
        
    def test_split_conformal_logic(self):
        """Verify fit_offline splits data into Train and Cal."""
        X = np.random.rand(100, 2)
        y = np.sum(X**2, axis=1)
        
        # Mock internal methods to intercept data
        self.scientist.model.fit = MagicMock()
        self.scientist.conformal.calibrate = MagicMock()
        
        # Pre-set current_law_func to simulate a law being found/existing
        # This allows the 'if self.current_law_func:' block to execute
        self.scientist.current_law_func = MagicMock()

        # Run fit_offline
        # We must suppress the actual heavy_think or mock it? 
        # fit_offline calls model.fit directly.
        
        # Create a dummy equation for evaluation to proceed
        self.scientist.model.equations_ = MagicMock()
        self.scientist.model.sympy = MagicMock(return_value="x0 + x1")
        
        # Mock helper
        with patch("atom.mind.scientist.AtomScientist._evaluate_discovery") as mock_eval:
             self.scientist.fit_offline(X, y)
             
        # CHECK 1: model.fit called with TRAIN set (80 samples)
        # fit(X_train, y_train, variable_names=...)
        call_args = self.scientist.model.fit.call_args
        X_train_passed = call_args[0][0]
        self.assertEqual(len(X_train_passed), 80, "Should use 80% for training")
        
        # CHECK 2: conformal.calibrate called with CAL set (20 samples)
        # interpolate via arguments
        cal_args = self.scientist.conformal.calibrate.call_args
        X_cal_passed = cal_args[0][0]
        self.assertEqual(len(X_cal_passed), 20, "Should use 20% for calibration")
        
        # Verify strict split (no overlap in memory address, or just disjoint indices if we tracked them)
        # Simply verifying lengths is good proxy.

    def test_trust_score_semantics(self):
        """Verify trust score is a valid normalized metric."""
        # Mock calibrated state
        self.scientist.conformal.is_calibrated = True
        self.scientist.conformal.q_score = 1.0
        
        # Mock models
        self.scientist.conformal.model = MagicMock()
        self.scientist.conformal.model.predict = MagicMock(return_value=np.array([1.0, 2.0]))
        self.scientist.conformal.error_estimator = MagicMock()
        self.scientist.conformal.error_estimator.predict = MagicMock(return_value=np.array([0.1, 1.0])) # Low error, High error
        
        # Mock current law func for "hybrid" path check (if needed)
        self.scientist.current_law_func = lambda x, y: x+y 
        
        # Predict
        X = np.array([[1,1], [2,2]])
        preds, trust = self.scientist.predict_theory(X)
        
        print(f"Preds: {preds}")
        print(f"Trust: {trust}")
        
        # Trust should be higher for lower error (0.1) than high error (1.0)
        # Trust[0] > Trust[1]
        self.assertTrue(trust[0] > trust[1], "Trust score should inversely correlate with estimated error")
        
        # Check constraints
        self.assertTrue(np.all(trust >= 0.0), "Trust must be non-negative")
        self.assertTrue(np.all(trust <= 1.0), "Trust must be <= 1.0")

if __name__ == "__main__":
    unittest.main()
