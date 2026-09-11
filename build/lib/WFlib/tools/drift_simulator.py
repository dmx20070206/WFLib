import numpy as np
import torch
import sys
import os


# ==========================================
# 1. Core Simulator Class
# ==========================================
class TrafficDriftSimulator:
    """
    Simulates physical network traffic drifts including network jitter,
    packet loss, protocol evolution, and user behavior changes.
    """

    def __init__(self, max_len=10000):
        self.max_len = max_len

    def _extract_valid_sequence(self, row):
        """Helper: Extract non-zero elements from padded sequence."""
        return row[row != 0]

    def _pad_sequence(self, seq):
        """Helper: Pad sequence with zeros to max_len."""
        seq_len = len(seq)
        if seq_len >= self.max_len:
            return seq[: self.max_len]
        return np.pad(seq, (0, self.max_len - seq_len), "constant", constant_values=0)

    # ------------------------------------------------------------------------
    # Simulation Operators
    # ------------------------------------------------------------------------

    def apply_global_shift(self, batch_data, shift_val):
        """Simulates RTT (Round Trip Time) baseline changes."""
        result = []
        for row in batch_data:
            valid_seq = self._extract_valid_sequence(row)
            if len(valid_seq) == 0:
                result.append(row)
                continue
            timestamps = np.abs(valid_seq)
            directions = np.sign(valid_seq)
            # Allow negative shift (faster network), clamped at 0
            new_timestamps = np.maximum(0, timestamps + shift_val)
            result.append(self._pad_sequence(new_timestamps * directions))
        return np.array(result)

    def apply_jitter(self, batch_data, sigma):
        """Simulates network jitter (queueing delay variance)."""
        result = []
        rng = np.random.default_rng()
        for row in batch_data:
            valid_seq = self._extract_valid_sequence(row)
            if len(valid_seq) == 0:
                result.append(row)
                continue
            timestamps = np.abs(valid_seq)
            directions = np.sign(valid_seq)
            noise = rng.normal(0, sigma, size=timestamps.shape)
            new_timestamps = np.maximum(0, timestamps + noise)
            # Re-sort is critical for time series
            sort_idx = np.argsort(new_timestamps)
            result.append(
                self._pad_sequence(new_timestamps[sort_idx] * directions[sort_idx])
            )
        return np.array(result)

    def apply_packet_loss(self, batch_data, drop_rate):
        """Simulates random packet loss on the link."""
        result = []
        rng = np.random.default_rng()
        for row in batch_data:
            valid_seq = self._extract_valid_sequence(row)
            if len(valid_seq) == 0:
                result.append(row)
                continue
            if drop_rate > 0:
                mask = rng.random(len(valid_seq)) > drop_rate
                new_seq = valid_seq[mask]
                result.append(self._pad_sequence(new_seq))
            else:
                result.append(row)
        return np.array(result)

    def apply_burst_scaling(self, batch_data, scale_factor, direction_filter=-1):
        """Simulates server-side content size changes."""
        if abs(scale_factor) < 1e-3:
            return batch_data

        result = []
        for row in batch_data:
            valid_seq = self._extract_valid_sequence(row)
            if len(valid_seq) == 0:
                result.append(row)
                continue

            directions = np.sign(valid_seq)
            timestamps = np.abs(valid_seq)
            new_seq_list = []

            # Group into bursts
            change_indices = np.where(np.diff(directions) != 0)[0] + 1
            burst_splits = np.split(np.arange(len(valid_seq)), change_indices)

            for indices in burst_splits:
                if len(indices) == 0:
                    continue
                burst_dir = directions[indices[0]]

                if burst_dir == direction_filter:
                    current_len = len(indices)
                    target_len = max(1, int(current_len * (1 + scale_factor)))

                    if target_len != current_len:
                        burst_timestamps = timestamps[indices]
                        new_burst_timestamps = np.interp(
                            np.linspace(0, 1, target_len),
                            np.linspace(0, 1, current_len),
                            burst_timestamps,
                        )
                        new_seq_list.append(new_burst_timestamps * burst_dir)
                    else:
                        new_seq_list.append(valid_seq[indices])
                else:
                    new_seq_list.append(valid_seq[indices])

            full_new_seq = (
                np.concatenate(new_seq_list) if new_seq_list else np.array([])
            )
            result.append(self._pad_sequence(full_new_seq))
        return np.array(result)

    def apply_burst_fragmentation(self, batch_data, frag_prob, gap=0.01):
        """
        [NEW] Simulates protocol fragmentation / "Chattiness".
        Matches "Day 270" observation where 'Burst Count' increased significantly.
        """
        if frag_prob <= 0:
            return batch_data

        result = []
        rng = np.random.default_rng()

        for row in batch_data:
            # 1. Use class helper to extract valid sequence
            valid_seq = self._extract_valid_sequence(row)
            if len(valid_seq) == 0:
                result.append(row)
                continue

            directions = np.sign(valid_seq)
            timestamps = np.abs(valid_seq)
            new_seq_parts = []

            # 2. Identify existing bursts
            change_indices = np.where(np.diff(directions) != 0)[0] + 1
            burst_splits = np.split(np.arange(len(valid_seq)), change_indices)

            for indices in burst_splits:
                if len(indices) < 2:
                    new_seq_parts.append(valid_seq[indices])
                    continue

                # 3. Randomly decide to fragment
                if rng.random() < frag_prob:
                    split_idx = rng.integers(1, len(indices))
                    part1_idx = indices[:split_idx]
                    part2_idx = indices[split_idx:]

                    part1_seq = valid_seq[part1_idx]

                    # Add gap to second part to create temporal separation
                    part2_ts = timestamps[part2_idx] + gap
                    part2_seq = part2_ts * directions[part2_idx]

                    new_seq_parts.append(part1_seq)
                    new_seq_parts.append(part2_seq)
                else:
                    new_seq_parts.append(valid_seq[indices])

            # 4. Concatenate and Re-sort (Crucial for time monotonicity)
            full_new_seq = (
                np.concatenate(new_seq_parts) if new_seq_parts else np.array([])
            )
            full_ts = np.abs(full_new_seq)
            full_dir = np.sign(full_new_seq)

            sort_idx = np.argsort(full_ts)
            sorted_seq = full_ts[sort_idx] * full_dir[sort_idx]

            # 5. Use class helper to pad
            result.append(self._pad_sequence(sorted_seq))

        return np.array(result)

    def apply_front_pad_crop(self, batch_data, k_head):
        """Simulates handshake changes (padding) or fast-open (cropping)."""
        if k_head == 0:
            return batch_data
        result = []
        for row in batch_data:
            valid_seq = self._extract_valid_sequence(row)
            if len(valid_seq) == 0:
                result.append(row)
                continue

            if k_head > 0:
                first_ts = np.abs(valid_seq[0])
                pad_timestamps = np.linspace(
                    max(0, first_ts - 0.1), first_ts, k_head + 1
                )[:-1]
                pad_seq = pad_timestamps * 1.0  # Assume +1 direction
                new_seq = np.concatenate([pad_seq, valid_seq])
            else:
                new_seq = (
                    valid_seq[abs(k_head) :]
                    if len(valid_seq) > abs(k_head)
                    else np.array([])
                )

            result.append(self._pad_sequence(new_seq))
        return np.array(result)

    def apply_tail_crop(self, batch_data, keep_ratio):
        """Simulates user behavior (early termination)."""
        if keep_ratio >= 1.0:
            return batch_data
        result = []
        for row in batch_data:
            valid_seq = self._extract_valid_sequence(row)
            if len(valid_seq) == 0:
                result.append(row)
                continue
            target_len = int(len(valid_seq) * keep_ratio)
            result.append(self._pad_sequence(valid_seq[:target_len]))
        return np.array(result)


# ==========================================
# 2. Parameter Space & Pipeline
# ==========================================
# DEFAULT = {
#     "global_shift": (-0.5, 0.5), 
#     "jitter_sigma": (0.001, 0.05), 
#     "packet_loss": (0.0, 0.01),
#     "burst_scale": (-0.05, 0.05), 
#     "frag_prob": (0.0, 0.0), 
#     "front_pad_crop": (-1, 5), 
#     "tail_crop_ratio": (0.95, 1.0),
# }

DEFAULT = {
    "global_shift": (-0.0, 0.0), 
    "jitter_sigma": (0.05, 0.05), 
    "packet_loss": (0.0, 0.0),
    "burst_scale": (-0.0, 0.0), 
    "frag_prob": (0.0, 0.0), 
    "front_pad_crop": (0, 0), 
    "tail_crop_ratio": (1.0, 1.0),
}

def sample_random_params(param_space=DEFAULT):
    """Sample random parameters from the defined space."""
    rng = np.random.default_rng()
    params = {}

    params["shift"] = rng.uniform(
        *param_space.get("global_shift", DEFAULT["global_shift"])
    )
    params["jitter"] = rng.uniform(
        *param_space.get("jitter_sigma", DEFAULT["jitter_sigma"])
    )
    params["loss"] = rng.uniform(
        *param_space.get("packet_loss", DEFAULT["packet_loss"])
    )
    params["burst_scale"] = rng.uniform(
        *param_space.get("burst_scale", DEFAULT["burst_scale"])
    )
    params["tail_crop"] = rng.uniform(
        *param_space.get("tail_crop_ratio", DEFAULT["tail_crop_ratio"])
    )

    frag_range = param_space.get("frag_prob", (0.0, 0.0))
    params["frag_prob"] = rng.uniform(*frag_range)

    low, high = param_space.get("front_pad_crop", DEFAULT["front_pad_crop"])
    params["front_k"] = rng.integers(low, high + 1)

    return params


def apply_random_parametric_drift(data, max_len=10000, specific_params=None):
    """Main transformation pipeline."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    sim = TrafficDriftSimulator(max_len=max_len)

    if specific_params:
        params = specific_params
    else:
        params = sample_random_params(DEFAULT)

    # 1. Content & Structure (Server/Proto)
    data_aug = sim.apply_burst_scaling(data, scale_factor=params.get("burst_scale", 0))

    # [UPDATED] Call class method for fragmentation
    data_aug = sim.apply_burst_fragmentation(
        data_aug, frag_prob=params.get("frag_prob", 0.0)
    )

    # 2. Behavior (Client)
    data_aug = sim.apply_front_pad_crop(data_aug, k_head=params.get("front_k", 0))
    data_aug = sim.apply_tail_crop(data_aug, keep_ratio=params.get("tail_crop", 1.0))

    # 3. Transmission (Network)
    data_aug = sim.apply_packet_loss(data_aug, drop_rate=params.get("loss", 0))
    data_aug = sim.apply_jitter(data_aug, sigma=params.get("jitter", 0))
    data_aug = sim.apply_global_shift(data_aug, shift_val=params.get("shift", 0))

    return data_aug, params