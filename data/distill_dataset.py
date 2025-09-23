import torch
import torch.utils.data as data
from source.args import args
 
class DistillDataset(data.Dataset):
    def __init__(self, data_np, data_recon_np):
        self.data_np = torch.from_numpy(data_np).float()
        self.data_recon_np = torch.from_numpy(data_recon_np).float()
        
        self.data_np = self.data_np.to(args.device)
        self.data_recon_np = self.data_recon_np.to(args.device)
        self._variance = None

    def __getitem__(self, index):
        return self.data_np[index], self.data_recon_np[index]

    def __len__(self):
        return self.data_np.shape[0]
    
    def calculate_variance(self, batch_size):
        if self._variance is None:
            with torch.no_grad():
                random_indices = torch.randint(0, len(self.data_recon_np), (batch_size,))
                if torch.cuda.is_available():
                    random_indices = random_indices.cuda()
                image_batch = self.data_recon_np[random_indices]
                self._variance = torch.var(image_batch).item()
        return self._variance


