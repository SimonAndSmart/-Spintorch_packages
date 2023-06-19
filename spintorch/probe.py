import torch
import skimage


class WaveProbe(torch.nn.Module):
	def __init__(self, x, y):
		super().__init__()

		self.register_buffer('x', torch.tensor(x, dtype=torch.int64))
		self.register_buffer('y', torch.tensor(y, dtype=torch.int64))

	def forward(self, m):
		return m[:,0, self.x, self.y]

	def coordinates(self):
		return self.x.cpu().numpy(), self.y.cpu().numpy()

class WaveIntensityProbe(WaveProbe):
	def __init__(self, x, y):
		super().__init__(x, y)

	def forward(self, m):
		return super().forward(m).pow(2)

class WaveIntensityProbeDisk(WaveProbe):
	def __init__(self, x, y, r,probe_mesure_method='sum'): #changes here _sinan
		x, y = skimage.draw.disk((x, y), r)
		super().__init__(x, y)
		self.probe_mesure_method=probe_mesure_method

	def forward(self, m):
		readings=super().forward(m)
		if self.probe_mesure_method == 'mean':
			return (readings.sum().pow(2)/len(readings[0])).unsqueeze(0)
		elif self.probe_mesure_method == 'sum':
			return readings.sum().pow(2).unsqueeze(0)
		else:
			raise ValueError(f"'probe_mesure_method' from 'WaveIntensityProbeDisk' can only be 'mean' or 'sum',but get: {self.probe_mesure_method}")
