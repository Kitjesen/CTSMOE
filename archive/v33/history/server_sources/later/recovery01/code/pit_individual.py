"""Per-environment, per-speed pit progression; exploration is not certification."""
import torch

PROFILE = 'individual_speed_v1'


class IndividualPitCourse:
    def __init__(self, levels, max_level=21):
        self.max_level = max_level
        self.levels = levels.long()[:, None].repeat(1, 4).clamp(0, max_level)
        self.wins = torch.zeros_like(self.levels)
        self.fails = torch.zeros_like(self.levels)
        self.bins = torch.randint(4, (len(levels),), device=levels.device)
        self.review = torch.zeros_like(levels, dtype=torch.bool)
        self.promotions = torch.zeros_like(self.levels)
        self.demotions = torch.zeros_like(self.levels)

    def record(self, ids, actual, success):
        bins = self.bins[ids]
        eligible = ~self.review[ids] & (actual == self.levels[ids, bins])
        ids, bins, success = ids[eligible], bins[eligible], success[eligible]
        self.wins[ids, bins] = torch.where(success, self.wins[ids, bins]+1, 0)
        self.fails[ids, bins] = torch.where(~success, self.fails[ids, bins]+1, 0)
        up, down = self.wins[ids, bins] >= 2, self.fails[ids, bins] >= 2
        previous = self.levels[ids, bins]
        current = (previous + up.long() - down.long()).clamp(0, self.max_level)
        self.levels[ids, bins] = current
        self.promotions[ids, bins] += (current > previous).long()
        self.demotions[ids, bins] += (current < previous).long()
        reset = up | down
        self.wins[ids[reset], bins[reset]] = 0
        self.fails[ids[reset], bins[reset]] = 0

    def sample(self, ids):
        self.bins[ids] = torch.randint(4, (len(ids),), device=ids.device)
        frontier = self.levels[ids, self.bins[ids]]
        self.review[ids] = (torch.rand(len(ids), device=ids.device) < .2) & (frontier > 0)
        lower = (torch.rand(len(ids), device=ids.device)*frontier).long()
        return torch.where(self.review[ids], lower, frontier)

    def initialize_ids(self, ids, levels):
        self.levels[ids] = levels[:, None].clamp(0, self.max_level)
        self.wins[ids] = 0
        self.fails[ids] = 0
        self.review[ids] = False
        self.bins[ids] = torch.randint(4, (len(ids),), device=ids.device)

    def state_dict(self):
        return dict(profile=PROFILE, max_level=self.max_level,
                    **{name: getattr(self, name) for name in
                       ('levels', 'wins', 'fails', 'bins', 'review', 'promotions', 'demotions')})

    def load(self, state):
        if state['profile'] != PROFILE or state['max_level'] != self.max_level:
            raise ValueError('Individual pit curriculum contract mismatch')
        for name in ('levels', 'wins', 'fails', 'bins', 'review', 'promotions', 'demotions'):
            tensor = getattr(self, name)
            if tensor.shape != state[name].shape:
                raise ValueError('Individual pit curriculum shape mismatch')
            tensor.copy_(state[name].to(tensor.device))
        if bool(((self.levels < 0) | (self.levels > self.max_level)).any()):
            raise ValueError('Individual pit height out of range')

    def snapshot(self, active, learners, actual):
        result = dict(profile=PROFILE, success_streak=2, failure_streak=2,
                      review_probability=.2, max_height_m=.15+.05*self.max_level,
                      interpretation='independent exploration levels, not certified capability', by_learner={})
        for index, name in enumerate(('teacher', 'student')):
            mask = active & (learners == index)
            levels = actual[mask]
            result['by_learner'][name] = dict(
                count=int(mask.sum()), actual_counts=torch.bincount(levels, minlength=22).cpu().tolist(),
                mean_height_m=float((.15+.05*levels.float()).mean()) if len(levels) else None,
                max_height_m=float(.15+.05*levels.max()) if len(levels) else None,
                target_mean_height_by_speed=(.15+.05*self.levels[mask].float()).mean(0).cpu().tolist(),
                promotions=int(self.promotions[mask].sum()), demotions=int(self.demotions[mask].sum()))
        return result
