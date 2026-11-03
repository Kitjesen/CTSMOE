"""An opt-in compute path for Isaac Lab 2.3.2 DelayBuffer.

Keeps the upstream buffer, lag validation and reset implementation. Capacity and
reset notifications are already known on the host, so compute need not read
GPU scalars or ask the GPU whether a reset happened. No RNG calls are added.
"""

import copy

import torch


class FastDelayBuffer:
    def __init__(self, reference):
        self._reference = reference
        self._ring = reference._circular_buffer
        if self._ring._buffer is not None:
            raise ValueError("Install before the first compute; active buffers are not replaced")
        self._capacity = reference.history_length + 1
        self._pending_all = True
        self._pending_resets = []

    def __getattr__(self, name):
        return getattr(self._reference, name)

    def reset(self, batch_ids=None):
        self._reference.reset(batch_ids)
        if batch_ids is None:
            self._pending_all = True
            self._pending_resets.clear()
        elif not self._pending_all:
            # Caller may reuse a tensor/list of reset IDs before the next compute.
            ids = batch_ids.clone() if isinstance(batch_ids, torch.Tensor) else copy.copy(batch_ids)
            self._pending_resets.append(ids)

    def compute(self, data):
        ring = self._ring
        if data.shape[0] != self.batch_size:
            raise ValueError(f"Expected batch size {self.batch_size}, got {data.shape[0]}")
        data = data.to(self.device)
        if ring._buffer is None:
            ring._pointer = -1
            ring._buffer = torch.empty(
                (self._capacity, *data.shape), dtype=data.dtype, device=self.device
            )
        ring._pointer = (ring._pointer + 1) % self._capacity
        ring._buffer[ring._pointer] = data
        if self._pending_all:
            ring._buffer[:] = data
            self._pending_all = False
        else:
            for ids in self._pending_resets:
                ring._buffer[:, ids] = data[ids]
        self._pending_resets.clear()
        ring._num_pushes += 1
        valid_lag = torch.minimum(self.time_lags, ring._num_pushes - 1)
        indices = torch.remainder(ring._pointer - valid_lag, self._capacity)
        # Advanced indexing already owns its result; no extra clone is necessary.
        return ring._buffer[indices, ring._ALL_INDICES]
