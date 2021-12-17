import os
from glob import glob
from torch import Tensor
from typing import Tuple


from clmr.datasets import Dataset


class AUDIO(Dataset):
    """Create a Dataset for any folder of audio files.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        src_ext_audio (str): The extension of the audio files to analyze.
    """

    def __init__(
        self,
        root: str,
        src_ext_audio: str = ".wav",
        n_classes: int = 11,
    ) -> None:
        super(AUDIO, self).__init__(root)

        self._path = root
        self._src_ext_audio = src_ext_audio
        self.n_classes = n_classes

        self.fl = glob(
            os.path.join(self._path, "**", "*{}".format(self._src_ext_audio)),
            recursive=True,
        )

        if len(self.fl) == 0:
            raise RuntimeError(
                "Dataset not found. Please place the audio files in the {} folder.".format(
                    self._path
                )
            )
        self.nsynth_labelmap = {"bass":0, "brass":1, "flute":2, "guitar":3, "keyboard":4, "mallet":5, 
                   "organ":6, "reed":7, "string":8, "synth_lead":9, "vocal":10}

    def file_path(self, n: int) -> str:
        fp = self.fl[n]
        return fp

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple [Tensor, Tensor]: ``(waveform, label)``
        """
        audio, _ = self.load(n)
        class_name = self.file_path(n).split('/')[-1].split('_')[0]
        label = [self.nsynth_labelmap[class_name]]
        return audio, label

    def __len__(self) -> int:
        return len(self.fl)
