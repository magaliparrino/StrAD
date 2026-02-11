# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# --- ADAPTATION NOTICE ---
# This file is adapted from MemStream (https://github.com/Stream-AD/MemStream)
# Original License: Apache License 2.0
#
#
# MODIFICATIONS IN StrAD:
# - Adapted to TSB-AD pipeline


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import tqdm
import math

from TSB_AD.utils.torch_utility import EarlyStoppingTorch, adjust_learning_rate, get_gpu
from TSB_AD.utils.dataset import ReconstructDataset

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

class MemStream(nn.Module):
    def __init__(self, in_dim,
                 memory_len=256,
                 beta=0.1,
                 lr=1e-2,
                 epochs=100,
                 win_size=1,
                 batch_size=128,
                 validation_size=0.2,
                 patience=10,
                 noise_std=1e-3,
                 lradj="type1",
                 cuda=True,
                 adapt_stats=True):
        """
        in_dim           : Input dimension (D)
        memory_len       : Size of the memory (number of stored representations)
        beta             : Update threshold (L1 distance <= beta triggers memory update)
        lr               : Learning rate
        epochs           : Maximum number of epochs
        win_size         : Window size for the reconstruction dataset
        batch_size       : Batch size for training
        validation_size  : Fraction of training data used for validation
        patience         : Patience for early stopping
        noise_std        : Standard deviation of Gaussian noise added for stability
        adapt_stats      : If True, recalculates mean/std on mem_data at each update
        """
        super(MemStream, self).__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim * 2

        # Hyperparameters
        self.memory_len = memory_len
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.win_size = win_size
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.patience = patience
        self.noise_std = noise_std
        self.adapt_stats = adapt_stats
        self.lradj = lradj
        
        # Device configuration
        self.device = get_gpu(cuda)
        self.max_thres = torch.tensor(self.beta, device=self.device)  
        
        self.K = 3  # Number of neighbors
        self.gamma = 0  # Discount weighting
        self.exp = torch.tensor([self.gamma**i for i in range(self.K)], device=self.device)      

        # Memory storage (representations + raw data)
        self.memory = torch.randn(self.memory_len, self.out_dim, device=self.device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim, device=self.device)
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False

        # Encoder/Decoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.ReLU(),
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.in_dim),
        ).to(self.device)

        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Normalization statistics (fixed based on training set)
        self.mean = torch.zeros(self.in_dim, device=self.device)
        self.std  = torch.ones(self.in_dim, device=self.device)
        self.eps  = 1e-8

        # Counter for circular memory updates
        self.count = 0

    # ---------------- Utilities ----------------
    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(x):
            return x.to(self.device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

    def _normalize(self, x):
        """Standardization using fixed training stats."""
        safe_std = torch.where(self.std < self.eps, torch.ones_like(self.std), self.std)
        x_norm = (x - self.mean) / safe_std
        zero_mask = self.std < self.eps
        if zero_mask.any():
            x_norm[:, zero_mask] = 0.0
        return x_norm

    def _min_l1_scores(self, enc):
        """Calculates min L1 distance between encoding and memory."""
        # Broadcasting: (T,1,out) - (1,M,out) -> (T,M,out) -> L1 sum -> (T,M)
        dists = (enc.unsqueeze(1) - self.memory.unsqueeze(0)).abs().sum(dim=-1)
        scores, _ = dists.min(dim=1)  
        return scores
    
    def _knn_score(self, enc):
        """Calculates distance-based anomaly score using K-Nearest Neighbors in memory."""
        dists = (enc.unsqueeze(1) - self.memory.unsqueeze(0)).abs().sum(dim=-1)  

        M = self.memory.shape[0]
        k = min(self.K, M)
        if k <= 0:
            return torch.zeros(enc.shape[0], device=self.device)

        # Get top-K nearest neighbors
        topk_vals, _ = torch.topk(dists, k=k, largest=False, dim=1)  

        if self.gamma == 0.0:
            return topk_vals.mean(dim=1)

        # Weighted distance with gamma discount
        exp = torch.tensor([self.gamma**i for i in range(k)], device=self.device)
        weighted = (topk_vals * exp).sum(dim=1) / exp.sum()
        return weighted

    def _maybe_update_memory(self, enc, raw_x, scores):
        """Updates memory slots if the anomaly score is below the beta threshold."""
        for t in range(enc.shape[0]):
            if scores[t] <= self.max_thres:
                pos = self.count % self.memory_len
                self.memory[pos] = enc[t]
                self.mem_data[pos] = raw_x[t]
                self.count += 1

        # Optional: Adaptive stats recalculation
        if self.adapt_stats and self.count > 0:
            self.mean = self.mem_data.mean(dim=0)
            self.std  = self.mem_data.std(dim=0)
            self.std = torch.where(self.std < self.eps, torch.ones_like(self.std), self.std)

    # ---------------- Fit Process ----------------

    def fit(self, data, save_path=None, patience=7, verbose=False, delta=1e-4, stride=1):
        """
        Trains the autoencoder and initializes the memory buffer.
        """
        # Train / Validation Split
        N = len(data)
        cut = int((1 - self.validation_size) * N)
        tsTrain = np.asarray(data[:cut], dtype=np.float32)
        tsValid = np.asarray(data[cut:], dtype=np.float32)

        # Global feature-wise stats
        self.mean = torch.as_tensor(tsTrain.mean(axis=0), dtype=torch.float32, device=self.device)
        self.std  = torch.as_tensor(tsTrain.std(axis=0),  dtype=torch.float32, device=self.device)
        self.std = torch.where(self.std < self.eps, torch.ones_like(self.std), self.std)

        # Dataloaders
        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size, stride=stride, normalize=False),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size, stride=stride, normalize=False),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        if len(train_loader) == 0:
            raise ValueError(f"No windows found in training set: N_train={len(tsTrain)}")

        early_stopping = EarlyStoppingTorch(save_path=save_path, patience=patience, verbose=verbose, delta=delta)

        # Training Loop
        for epoch in range(1, self.epochs + 1):
            self.train()
            train_loss_sum, train_count = 0.0, 0

            tloop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True,
                            desc=f"Training Epoch [{epoch}/{self.epochs}]")
            for i, (batch_x, _) in tloop:
                batch_x = torch.as_tensor(batch_x, dtype=torch.float32, device=self.device)
                B, T, D = batch_x.shape
                x_flat = batch_x.reshape(B * T, D)

                # Stateful scaling
                x_norm = (x_flat - self.mean) / self.std

                # Augment with light noise
                if self.noise_std > 0:
                    x_norm = x_norm + (self.noise_std * torch.randn_like(x_norm))

                self.optimizer.zero_grad()
                z = self.encoder(x_norm)
                recon = self.decoder(z)
                loss = self.loss_fn(recon, x_norm)
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item() * x_norm.shape[0]
                train_count += x_norm.shape[0]
                tloop.set_postfix(loss=loss.item(), avg_loss=train_loss_sum / train_count)

            # Validation Phase
            self.eval()
            valid_loss_sum, valid_count = 0.0, 0
            with torch.no_grad():
                vloop = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True,
                                desc=f"Valid Epoch [{epoch}/{self.epochs}]")
                for i, (batch_x, _) in vloop:
                    batch_x = torch.as_tensor(batch_x, dtype=torch.float32, device=self.device)
                    B, T, D = batch_x.shape
                    x_flat = batch_x.reshape(B * T, D)
                    x_norm = (x_flat - self.mean) / self.std

                    z = self.encoder(x_norm)
                    recon = self.decoder(z)
                    vloss = self.loss_fn(recon, x_norm)

                    valid_loss_sum += vloss.item() * x_norm.shape[0]
                    valid_count += x_norm.shape[0]
                    vloop.set_postfix(valid_loss=valid_loss_sum / max(1, valid_count))

            valid_loss = valid_loss_sum / max(1, valid_count)
            early_stopping(valid_loss, self)
            if early_stopping.early_stop:
                print("   Early stopping <<<")
                break
            
            adjust_learning_rate(self.optimizer, epoch + 1, self.lradj, self.lr)

        # Initialize Memory Buffer from encoded training data
        train_tensor = torch.as_tensor(tsTrain, dtype=torch.float32, device=self.device)
        x_norm_all = (train_tensor - self.mean) / self.std

        with torch.no_grad():
            enc_all = self.encoder(x_norm_all)

        N_train = enc_all.shape[0]
        if N_train >= self.memory_len:
            # Uniform sampling to fill initial memory
            idx = np.linspace(0, N_train - 1, num=self.memory_len).astype(int)
        else:
            idx = np.arange(N_train)

        with torch.no_grad():
            self.memory[:len(idx)] = enc_all[idx]
            self.mem_data[:len(idx)] = train_tensor[idx]

        self.count = len(idx) % self.memory_len
        return self

    # ---------------- Inference ----------------
    def forward(self, x, update=True):
        """
        Returns torch scores (T,). 
        If update=True, memory is updated for steps where score <= beta.
        """
        X = self._to_tensor(x)
        if X.ndim == 1:
            X = X.unsqueeze(0)

        X_norm = self._normalize(X)
        enc = self.encoder(X_norm)
        scores = self._knn_score(enc)

        if update:
            self._maybe_update_memory(enc, X, scores)

        return scores

    def decision_function(self, x):
        """
        Standard API for anomaly score retrieval.
        """
        scores_torch = self.forward(x, update=True)
        scores_np = scores_torch.detach().cpu().numpy().astype(np.float32)
        if scores_np.shape[0] == 1:
            return float(scores_np[0])
        return scores_np
