# --- ADAPTATION NOTICE ---
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This Python module is a wrapper for the LEAP (MESI) Java implementation.
# Original Java Source: https://infolab.usc.edu/Luan/Outlier/CountBasedWindow/DODDS/
# Author: Luan Tran (USC Infolab)
# 
# MODIFICATIONS IN StrAD:
# - Developed JPype bridge to interface with the original Java LeapRunner.
# - Implemented Stateful Feature-wise Scaling to align with StrAD standards.
# - Added internal buffering logic to manage streaming 'slide' requirements.

import numpy as np
import jpype
import jpype.imports
from jpype.types import JDouble, JInt, JArray
import os

class LEAP:
    """
    LEAP (MESI) Wrapper via JPype.
    API:
      - fit(X_train): Pre-trains on the initial window and ingests remaining train data.
      - decision_function(window): Processes streaming data and returns anomaly scores.
    """

    def __init__(self, k=5, R=1.0, W=100, slide=10,
                 jar_path="target/leap-1.0.0-jar-with-dependencies.jar",
                 jvm_opts=None, mode="normalized"):
        self.k = int(k)
        self.R = float(R)
        self.W = int(W)
        self.slide = int(slide)
        self.jar_path = os.path.abspath(jar_path)
        self._java_runner = None
        self._jvm_started_here = False
        self._jvm_opts = (jvm_opts or [])
        self._n_seen = 0
        self.mode = mode
        self.pretrain_on_full_train = True
        self._buffer = None  # np.ndarray to store incoming streaming points

    def _start_jvm(self):
        """Initializes the Java Virtual Machine if not already running."""
        print("Checking JVM status...")
        if jpype.isJVMStarted():
            return 

        jar = self.jar_path
        if not os.path.exists(jar):
            raise FileNotFoundError(f"LEAP JAR not found: {jar}")

        # Add JAR to classpath via dedicated method to avoid OS-specific separators
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
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"Data must be 1D or 2D; received shape={arr.shape}")
        return arr

    @staticmethod
    def _to_java_2d_array(np_array: np.ndarray):
        """Converts a NumPy array into a Java 2D double array."""
        arr = LEAP._ensure_2d(np_array)
        outer = JArray(JArray(JDouble))(arr.shape[0])
        for i in range(arr.shape[0]):
            outer[i] = JArray(JDouble)(arr[i].tolist())
        return outer

    def fit(self, train_data: np.ndarray):
        """
        Pre-trains MESI on initial window W, then ingests remaining data in 'slide' packets.
        Implements stateful feature-wise scaling.
        """
        if not os.path.exists(self.jar_path):
            raise FileNotFoundError(f"JAR not found: {self.jar_path}")

        self._start_jvm()
        from outlierdetection import LeapRunner

        # Initialize the Java underlying model
        self._java_runner = LeapRunner(JInt(self.k), JDouble(self.R), JInt(self.W), JInt(self.slide))

        # Stateful Feature-wise Scaling logic
        self.mean = np.mean(train_data, axis=0)
        self.std = np.std(train_data, axis=0)
        self.std = np.where(self.std == 0, 1e-8, self.std)
        data_normalized = (train_data - self.mean) / self.std

        train_data = self._ensure_2d(data_normalized)
        n = train_data.shape[0]

        # 1) Initial window filling
        if n > self.W:
            init_block = train_data[:self.W]
            rest = train_data[self.W:]
        elif n < self.W:
            print(f"Warning: Train data size ({n}) is smaller than window W ({self.W}). Padding applied.")
            padding_size = self.W - n
            padding = train_data[:padding_size]
            train_data_padded = np.vstack([padding, train_data])  
            init_block = train_data_padded
            rest = train_data[0:0]
        else:
            init_block = train_data
            rest = train_data[0:0]

        self._java_runner.fit(self._to_java_2d_array(init_block))
        self._n_seen = init_block.shape[0]

        # 2) Full pre-training: ingestion by packets of size 'slide'
        if self.pretrain_on_full_train and rest.shape[0] > 0:
            n_complete_chunks = (rest.shape[0] // self.slide) * self.slide
            rest_to_ingest = rest[:n_complete_chunks]
            leftover = rest[n_complete_chunks:]
            
            for start in range(0, rest_to_ingest.shape[0], self.slide):
                batch = rest_to_ingest[start:start + self.slide]
                self._java_runner.ingest(self._to_java_2d_array(batch))
                self._n_seen += batch.shape[0]
                
            # Initialize buffer with orphan points that didn't fit in a chunk
            self._buffer = leftover.copy()
        else:
            self._buffer = np.empty((0, train_data.shape[1]), dtype=np.float64)

        return self

    def reset_stream(self, start_seen: int, n_features):
        """
        Resets the streaming state to a specific position.
        - start_seen: number of points considered 'seen' (usually = W)
        """
        if start_seen < 0:
            raise ValueError("start_seen must be >= 0")
        self._n_seen = int(start_seen)
        self._buffer = np.empty((0, int(n_features)), dtype=np.float64)

    def decision_function(self, window: np.ndarray, return_array: bool = True):        
        """
        Processes streaming data. Applies stateful normalization and returns scores 
        only when a full 'slide' chunk is accumulated.
        """
        if self._java_runner is None:
            raise RuntimeError("LEAP must be fit() before calling decision_function().")
        
        # Apply stored training scaling to the incoming window
        data_normalized = (window - self.mean) / self.std

        window = self._ensure_2d(data_normalized)
        n_features = window.shape[1]

        if self._buffer is None:
            self._buffer = np.empty((0, n_features), dtype=np.float64)

        # Get only the newest point from the window
        new_block = window[-1:].copy()

        # Add to buffer
        if new_block.shape[0] > 0:
            self._buffer = np.vstack([self._buffer, new_block])

        # Ingest exactly one chunk if enough data is accumulated
        if self._buffer.shape[0] >= self.slide:
            chunk = self._buffer[:self.slide]
            jchunk = self._to_java_2d_array(chunk)

            # Retrieve scores from Java based on the selected mode
            if self.mode == "binary":
                jscores = self._java_runner.scoreBinary(jchunk)
                chunk_scores = np.array(list(jscores), dtype=np.int32)
            elif self.mode == "deficit":
                jscores = self._java_runner.scoreDeficit(jchunk)
                chunk_scores = np.array(list(jscores), dtype=np.int32)
            elif self.mode == "normalized":
                jscores = self._java_runner.scoreNormalized(jchunk)
                chunk_scores = np.array(list(jscores), dtype=np.float64)
            else:
                raise ValueError("Mode must be 'binary', 'deficit', or 'normalized'")

            # Clear consumed chunk from buffer
            self._buffer = self._buffer[self.slide:]
            self._n_seen += chunk.shape[0]

            if return_array:
                return chunk_scores
            else:
                last = chunk_scores[-1]
                return float(last) if self.mode == "normalized" else int(last)

        # Buffer not full yet -> return empty array or None
        if return_array:
            return np.array([], dtype=np.float64 if self.mode == "normalized" else np.int32)
        else:
            return None

    def close(self):
        """Shutdown the JVM if it was started by this instance."""
        if jpype.isJVMStarted() and self._jvm_started_here:
            jpype.shutdownJVM()