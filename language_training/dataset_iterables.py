from abc import ABC, abstractmethod

from datasets import load_dataset


class DatasetIterable(ABC):
    @abstractmethod
    def __iter__(self):
        """
        :return: An iterator over the dataset. Each iteration should return a tuple containing sample text.
        """
        pass

    @abstractmethod
    def __next__(self):
        """
        :return: The next sample text.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: The name of this dataset.
        """
        pass


class RedPajamaV2DatasetIterable(DatasetIterable):
    def __init__(self, use_sample_dataset: bool = False, languages: list[str] = ["en"]):
        self.dataset = load_dataset("togethercomputer/RedPajama-Data-V2",
                                    name="sample" if use_sample_dataset else "default",
                                    split="train", streaming=True, languages=languages)

    def __iter__(self):
        self.iterator = iter(self.dataset)
        return self

    def __next__(self):
        return next(self.iterator)['raw_content']

    @property
    def name(self) -> str:
        return "RedPajamaV2"


class PileDatasetIterable(DatasetIterable):
    def __init__(self):
        self.dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)

    def __iter__(self):
        self.iterator = iter(self.dataset)
        return self

    def __next__(self):
        return next(self.iterator)['text']

    @property
    def name(self) -> str:
        return "Pile"
