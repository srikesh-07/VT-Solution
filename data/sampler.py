import random

from torch.utils.data import BatchSampler


class CategoryWiseSampler(BatchSampler):
    def __init__(
        self,
        categorical_indices: dict,
        batch_size: int,
        shuffle: bool = True,
        mixed_batch_sampling: bool = True,
        drop_last: bool = False,
    ):
        super().__init__(sampler=[], batch_size=batch_size, drop_last=drop_last)
        self.shuffle = shuffle
        self.categorical_indices = categorical_indices
        self.mixed_batch_sampling = mixed_batch_sampling

    def __len__(self):
        batches = 0
        for indices in self.categorical_indices.values():
            batches += len(indices) // self.batch_size
            if not self.drop_last and (len(indices) % self.batch_size) > 0:
                batches += 1
        return batches

    def _create_seq_batches(self):
        for indices in self.categorical_indices.values():
            start_idx = 0
            for index_factor in range(0, len(indices) // self.batch_size):
                end_idx = (index_factor + 1) * self.batch_size
                yield indices[start_idx:end_idx]
                start_idx = end_idx
            else:
                if not self.drop_last:
                    if start_idx == 0:
                        yield indices
                    elif len(indices) % self.batch_size:
                        yield indices[end_idx:]
                if self.shuffle:
                    random.shuffle(indices)

    def _create_mixed_batches(self):
        indices_dict = dict()
        break_flag = 0
        items = list(self.categorical_indices.items())
        while break_flag < len(self.categorical_indices):
            for name, indices in items:
                if indices_dict.get(name, None) is None:
                    indices_dict[name] = [0, False]
                    start_idx = 0
                else:
                    start_idx = indices_dict[name][0]
                if indices_dict[name][1]:
                    continue
                end_idx = start_idx + self.batch_size
                if end_idx > len(indices):
                    break_flag += 1
                    indices_dict[name][1] = True
                    continue
                yield indices[start_idx:end_idx]
                indices_dict[name][0] = end_idx
        else:
            if not self.drop_last:
                for name, indices in items:
                    if len(indices) % self.batch_size:
                        yield indices[indices_dict[name][0] :]
                    if self.shuffle:
                        random.shuffle(indices)
            else:
                if self.shuffle:
                    random.shuffle(items)
                    for _, indices in items:
                        random.shuffle(indices)

    def __iter__(self):
        if self.mixed_batch_sampling:
            return self._create_mixed_batches()
        else:
            return self._create_seq_batches()
