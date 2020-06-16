import torch
import numpy as np
# code from AllenNLP

class Triangular(torch.optim.lr_scheduler._LRScheduler): # pylint: disable=protected-access
    """
    Slanted triangular learning rate scheduler.
    The LR will start at ``lr / ratio`` and increase linearly for ``warm_up`` epochs
    until reaching ``lr``, at which point it will decrease linearly for ``cool_down``
    epochs until reaching ``lr / ratio`` again. Then the LR will continue
    linearly decreasing down to 0 for the remaining number of epochs.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 warm_up: int,
                 cool_down: int,
                 ratio: int = 10,
                 last_epoch: int = -1) -> None:
        if num_epochs < warm_up + cool_down:
            raise ConfigurationError(f"'num_epochs' should be greater than the sum of 'warm_up' and 'cool_down'. "
                                     f"Got 'num_epochs' = {num_epochs} >= 'warm_up' ({warm_up}) + "
                                     f"'cool_down' ({cool_down}) = {warm_up + cool_down}.")
        self.num_epochs = num_epochs
        self.warm_up = warm_up
        self.cool_down = cool_down
        self.ratio = ratio
        self._initialized: bool = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized.
        if not self._initialized and self.last_epoch == 0:
            self._initialized = True
            step = 0
        else:
            step = min(self.last_epoch, self.num_epochs - 2) + 1
        if step <= self.warm_up:
            # Warm up phase: increase LR linearly.
            lrs = [lr / self.ratio + (lr - lr / self.ratio) * (step / self.warm_up)
                   for lr in self.base_lrs]
        elif step <= self.warm_up + self.cool_down:
            # Cool down phase: decrease LR linearly.
            lrs = [lr - (lr - lr / self.ratio) * (step - self.warm_up) / self.cool_down
                   for lr in self.base_lrs]
        else:
            # "Trickle-off" phase: continue decreasing linearly down to 0.
            lrs = [lr / self.ratio - (lr / self.ratio) * (step - self.warm_up - self.cool_down)
                   / (self.num_epochs - self.warm_up - self.cool_down)
                   for lr in self.base_lrs]
        return lrs

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs
    
    