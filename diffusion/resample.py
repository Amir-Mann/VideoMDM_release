from abc import ABC, abstractmethod

import numpy as np
import torch as th
from math import ceil
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion, args=None):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if args is not None and getattr(args, "t_star_method", "none") not in ["none", None]:
        if "multistep" in args.t_star_method:
            return TStarUniformSampler(diffusion, args)
        elif args.t_star_method == "curriculum":
            return CurriculumTStarSampler(diffusion, args)
        else:
            raise NotImplementedError(f"unknown t* method: {args.t_star_method}")

    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


def sample_above_t_star(sampler, batch_size, device):
    w = th.cat([th.zeros((sampler.effective_t_star, ), device=device), th.ones((sampler.T - sampler.effective_t_star,), device=device)], dim=0)
    p = w / th.sum(w)
    indices = th.multinomial(p, batch_size, replacement=True).long()
    weights = th.ones((batch_size,), device=device)
    return indices, weights

def sample_all_steps(sampler, batch_size, device, target=0):
    weights = th.tensor(1, device=device).float().expand(batch_size)
    return th.arange(sampler.effective_t_star - 1, target - 1, -1, device=device).expand(batch_size, -1), weights

def sample_buckets(sampler, batch_size, device):
    buckets = sampler.arg
    bucket_size = sampler.bucket_size
    assert buckets is not None and buckets < sampler.effective_t_star, "When using a bucket t_star sampler must provide t_star_arg as num buckets"

    # Generate random indices within each bucket
    bucket_offsets = th.arange((buckets - 1) * bucket_size, -1, -bucket_size, device=device)
    random_offsets = th.randint(0, bucket_size, (batch_size, buckets), device=device)
    indices = bucket_offsets.expand(batch_size, -1) + random_offsets
    weights = th.tensor(1, device=device).float().expand(batch_size)

    return indices, weights


class TStarUniformSampler(ScheduleSampler):
    def __init__(self, diffusion, args):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])
        self.T = self.diffusion.num_timesteps
        assert args.t_star is not None and args.t_star > 0, "When using a t* method which is not 'None' --t_star must be a positive integer."
        assert args.t_star < self.T, "t* must be less then T (the num_timesteps for the diffusion)"
        self.t_star = args.t_star
        self.arg = args.t_star_arg
        if "b1" in args.t_star_method:
            assert isinstance(self.arg, int) and self.arg > 1, "When using a bucket t* sampler must provide t_star_arg as num buckets"
            self.bucket_size = ceil((self.t_star) / (self.arg-1))
            self.effective_t_star = self.bucket_size * self.arg
            assert self.effective_t_star <= self.T, \
                f"t* must be less then num_timesteps, with the calculation ceil((t_star)/(t_star_arg-1))={self.effective_t_star} and T={self.T}"
            self.sample_method = sample_buckets
        else:
            self.effective_t_star = self.t_star
            self.sample_method = sample_all_steps
        t_star_frac = (self.T - self.effective_t_star) / self.effective_t_star
        self.probability_of_t_star_batch = 1 / (1 + t_star_frac)
        self.method = args.t_star_method

    def weights(self):
        return self._weights

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        if np.random.random() > self.probability_of_t_star_batch:
            indices, weights = sample_above_t_star(self, batch_size, device)
        else:
            indices, weights = self.sample_method(self, batch_size, device)
        return indices, weights

class CurriculumTStarSampler(ScheduleSampler):
    def __init__(self, diffusion, args):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])
        self.T = self.diffusion.num_timesteps
        assert args.t_star is not None and args.t_star > 0, "When using a t* method which is not 'None' --t_star must be a positive integer."
        assert args.t_star < self.T, "t* must be less then T (the num_timesteps for the diffusion)"
        self.t_star = args.t_star
        self.effective_t_star = args.t_star
        self.num_steps_for_each_t = args.t_star_arg
        self.num_steps_at_curr_t = 0
        self.curr_t = args.t_star - 1

        t_star_frac = (self.T - self.t_star) / self.t_star
        self.probability_of_t_star_batch = 1 / (1 + 1 * t_star_frac)
        self.method = args.t_star_method

    def weights(self):
        return self._weights

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        if np.random.random() > self.probability_of_t_star_batch:
            indices, weights = sample_above_t_star(self, batch_size, device)
        else:
            if self.num_steps_at_curr_t >= self.num_steps_for_each_t:
                self.num_steps_at_curr_t = 0
                self.curr_t = max(0, self.curr_t - 1)
            indices, weights = sample_all_steps(self, batch_size, device, target=self.curr_t)
            self.num_steps_at_curr_t += 1
        return indices, weights

if __name__=="__main__":
    import sys
    class Dummy:
        def __init__(self):
            self.t_star_method = sys.argv[1]
            self.t_star_arg = 4
            self.t_star = 15
            self.num_timesteps = 50
    diffusion = Dummy()
    args = Dummy()
    inst = create_named_schedule_sampler("", diffusion, args)
    bs = int(sys.argv[2])

    print(len(np.ones([diffusion.num_timesteps])))
    def debug():
        print()
        i, w = inst.sample(bs, "cpu")
        print(f"i.shape{i.shape}, max(i){th.max(i)}, min(i){th.min(i)}, w.shape{w.shape}")
        if len(i.shape) < 3:
            print(i)
        else:
            print(i[:4, :4])
            t = i
            ts_list = [t[:, i] for i in range(0, t.shape[1])]
            if True:# and torch.rand(1).item() < 0.2:
                print("len(ts_list)", len(ts_list))
                print("ts_list[0]", ts_list[0])
                print("ts_list[-1]", ts_list[-1])
                print("th.max(ts_list)", th.max(th.cat(ts_list)))
                print("th.min(ts_list)", th.min(th.cat(ts_list)))
        print(w)
    for i in range(10):
        debug()
