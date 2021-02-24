import pickle
import torch
import numpy as np

from .metadataset import Task
from .simple_functions import SimpleFunctionDataset


class CustomDataset(SimpleFunctionDataset):

  def __init__(
      self,
      file_path='./data.p',
      num_total_batches=200000,
      num_samples_per_function=5,
      num_val_samples=5,
      meta_batch_size=75,
      oracle=False,
      train=True,
      device='cpu',
      dtype=torch.float,
      **kwargs,
  ):
    super().__init__(
        num_total_batches=num_total_batches,
        num_samples_per_function=num_samples_per_function,
        num_val_samples=num_val_samples,
        meta_batch_size=meta_batch_size,
        oracle=oracle,
        train=train,
        device=device,
        dtype=dtype,
        **kwargs,
    )

    self.file_path = file_path
    with open(self.file_path, 'rb') as dataset_file:
      self.data = pickle.load(dataset_file)

    values = list(self.data.values())

    self.inputs, self.outputs = [], []
    for inputs, outputs in zip(values[::2], values[1::2]):
      self.inputs.append(inputs)
      self.outputs.append(outputs)

    self.inputs = np.concatenate(self.inputs).ravel()
    self.outputs = np.concatenate(self.outputs).ravel()
    self.infos = [None] * len(self.inputs)

    self.input_size = 1
    self.output_size = 1

  def _generate_batch(self):
    # print(self.inputs)
    outputs = np.zeros([self._meta_batch_size, self._num_total_samples, 1])
    inputs = np.zeros([self._meta_batch_size, self._num_total_samples, 1])
    # print(inputs.shape)
    for i in range(self._meta_batch_size):
      idx = np.random.randint(len(self.inputs), size=self._num_total_samples)
      inputs[i] = np.expand_dims(self.inputs[idx], axis=1)
      outputs[i] = np.expand_dims(self.outputs[idx], axis=1)

    return inputs, outputs, self.infos
