# # --- ADAPTATION NOTICE ---
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This Python module is a wrapper for the LEAP (MESI) Java implementation.
# Original Java Source: https://infolab.usc.edu/Luan/Outlier/CountBasedWindow/DODDS/
# Author: Luan Tran (USC Infolab)
#
# MODIFICATIONS IN StrAD:
# - Stateful Feature-wise Scaling: Implemented global mean/std persistence in fit() 
#   to replace window-local normalization, ensuring training distribution is preserved.
# - Java Integration: Developed a JPype wrapper for the MCOD (Micro-cluster based 
#   Outlier Detection) Java implementation.

import numpy as np
import jpype
import jpype.imports
from jpype.types import JDouble, JInt, JArray
import os

class MCOD:
    """
    MCOD Wrapper via JPype (Point-by-point logic).
    MCOD Hyperparameters: k (neighbors), R (radius), W (window size), slide.
    API:
      - fit(X_train): Pre-feeds MCOD with initial data to establish the window.
      - decision_function(x): Ingests data and returns anomaly scores.
    """

    def __init__(self, k=5, R=1.0, W=100,
                 jar_path="target/mcod-1.0.0-jar-with-dependencies.jar",
                 jvm_opts=None, mode="normalized"):
        self.k = int(k)
        self.R = float(R)
        self.W = int(W)
        self.slide = 1  # MCOD typically processes point-by-point
        self.jar_path = os.path.abspath(jar_path)
        self._java_runner = None
        self._jvm_started_here = False
        self._jvm_opts = (jvm_opts or [])
        self.mode = mode

    def _start_jvm(self):
        """Initializes the JVM if it is not already running."""
        print("Checking JVM status...")
        if jpype.isJVMStarted():
            return 

        jar = self.jar_path
        if not os.path.exists(jar):
            raise FileNotFoundError(f"MCOD JAR not found: {jar}")

        # Add the JAR to the classpath
        jpype.addClassPath(jar)
        print(f"JAR added to classpath: {jar}")

        try:
            jvm_path = jpype.getDefaultJVMPath()
            print(f"Using JVM path: {jvm_path}")
            
            jpype.startJVM(
                jvm_path,
                *self._jvm_opts,
                convertStrings=True
            )
            print("JVM started successfully.")
            self._jvm_started_here = True
        except Exception as e:
            print(f"Fatal error during JVM startup: {e}")
            raise

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensures input data is a 2D NumPy array."""
        arr = np.array(arr, dtype=np.float64, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)  # Reshape single point to 2D
        elif arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D data; received shape={arr.shape}")
        return arr

    @staticmethod
    def _to_java_point(np_row: np.ndarray):
        """Converts a single NumPy row into a Java Double array."""
        row = np.array(np_row, dtype=np.float64).ravel()
        return JArray(JDouble)(row.tolist())

    @staticmethod
    def _to_java_2d_array(np_array: np.ndarray):
        """Converts a 2D NumPy array into a Java 2D Double array."""
        arr = np.array(np_array, dtype=np.float64, copy=False)
        outer = JArray(JArray(JDouble))(arr.shape[0])
        for i in range(arr.shape[0]):
            outer[i] = JArray(JDouble)(arr[i].tolist())
        return outer

    def fit(self, X_train: np.ndarray, warmup_first_W: bool = True):
        """
        Pre-feeds MCOD with the first W points and establishes the training baseline.
        Implements stateful feature-wise scaling to align with StrAD standards.
        """
        self._start_jvm()
        from outlierdetection import MCODRunner
        self._java_runner = MCODRunner(JInt(self.k), JDouble(self.R), JInt(self.W), JInt(self.slide))

        # Stateful Feature-wise Scaling
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        self.std = np.where(self.std == 0, 1e-8, self.std)
        X = self._ensure_2d((X_train - self.mean) / self.std)

        if warmup_first_W and X.shape[0] >= self.W:
            init_block = X[:self.W]
            self._java_runner.fit(self._to_java_2d_array(init_block))
            for i in range(self.W, X.shape[0]):
                # Silent ingestion (establish window state without returning scores)
                self._java_runner.ingest(self._to_java_2d_array(X[i:i+1]))
        else:
            # Block ingestion if training set < window size or no warmup is requested
            self._java_runner.ingest(self._to_java_2d_array(X))

        return self

    def decision_function(self, x: np.ndarray, mode: str = "normalized", return_array: bool = False):
        """
        Ingests data point-by-point and returns anomaly scores.
        - x 1D: single point -> returns score for that specific point.
        - x 2D: window -> ingests each point sequentially, returns score of last point (or full array).
        """
        if self._java_runner is None:
            raise RuntimeError("MCOD must be fit() before calling decision_function().")
        
        mode = mode or self.mode
        
        # Apply the stored training-set scaling to incoming data
        X = self._ensure_2d((x - self.mean) / self.std)
        n, d = X.shape

        def score_one(row):
            jrow = self._to_java_point(row)
            if mode == "binary":
                return int(self._java_runner.scorePointBinary(jrow))
            elif mode == "deficit":
                return int(self._java_runner.scorePointDeficit(jrow))
            elif mode == "normalized":
                return float(self._java_runner.scorePointNormalized(jrow))
            else:
                raise ValueError("Mode must be 'binary', 'deficit', or 'normalized'")

        if n == 1:
            return score_one(X[0])

        scores = [score_one(X[i]) for i in range(n)]
        return np.array(scores, dtype=np.float64 if mode == "normalized" else np.int32) if return_array else scores[-1]

    def close(self):
        """Shutdown the JVM if it was started by this instance."""
        if jpype.isJVMStarted() and self._jvm_started_here:
            jpype.shutdownJVM()