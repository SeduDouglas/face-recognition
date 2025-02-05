import torch
from tqdm import tqdm

class Trainer:

    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        batch_loss_history = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            embeddings = self.model(data)
            
            loss, valid_triplets = self.criterion(embeddings, labels)

            if valid_triplets > 0:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            batch_loss_history.append(loss.item())
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'valid triplets' : f'{valid_triplets:.4f}'})
        
        return total_loss / num_batches, batch_loss_history
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        pairs_distance = []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                embeddings = self.model(data)
                loss, valid_triplets = self.criterion(embeddings, labels)
                for index_i in range(embeddings.size()[0] - 1):
                    for index_j in range(index_i + 1, embeddings.size()[0]):
                        pairs_distance.append(((embeddings[index_i] - embeddings[index_j]).pow(2).sum().sqrt(), labels[index_i] == labels[index_j]))
                
                total_loss += loss.item()
                num_batches += 1
        eq = [a for a in pairs_distance if a[1]] #Same Label distances
        dif = [a for a in pairs_distance if not a[1]] #Diferent label distances
        ta = len([a for a in eq if a[0] <= 1]) #N of True Acceptance
        fa = len([a for a in dif if a[0] <= 1]) #N of False Acceptance
        tn = len([a for a in dif if a[0] > 1]) #N of True Negative
        fn = len([a for a in eq if a[0] > 1]) #N of False Negative
        val = ta / len(eq) #Validation Rate
        far = fa / len(dif) #False Acceptance Rate
        acc = (ta + tn) / (ta + fa + tn + fn) #Accuracy

        return total_loss / num_batches, val, far, acc, pairs_distance
