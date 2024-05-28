import torch
import torch.nn as nn
import random



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        batch_sim = self.sim_func(emb1.unsqueeze(1), emb2.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, y_dim)
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /2./logvar.exp()).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound / 2.0

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MIEstimator(nn.Module):
    def __init__(self, args):
        super(MIEstimator, self).__init__()
        self.str_estimator = CLUBSample(args.dim, args.dim, args.dim)
        self.img_estimator = CLUBSample(args.img_dim, args.img_dim, args.img_dim)
        self.txt_estimator = CLUBSample(args.txt_dim, args.txt_dim, args.txt_dim)
        self.num = args.n_exp
    

    def forward(self, embeddings):
        strs, imgs, txts = embeddings
        idx1, idx2 = random.sample(range(self.num), k=2)
        str1, str2 = strs[idx1], strs[idx2]
        img1, img2 = imgs[idx1], imgs[idx2]
        txt1, txt2 = txts[idx1], txts[idx2]
        mi_loss = (self.str_estimator(str1, str2) + self.img_estimator(img1, img2) + self.txt_estimator(txt1, txt2)) / 3.0
        return mi_loss
    
    def train_estimator(self, embeddings):
        strs, imgs, txts = embeddings
        idx1, idx2 = random.sample(range(self.num), k=2)
        str1, str2 = strs[idx1], strs[idx2]
        img1, img2 = imgs[idx1], imgs[idx2]
        txt1, txt2 = txts[idx1], txts[idx2]
        est_loss = (self.str_estimator.learning_loss(str1, str2) + self.img_estimator.learning_loss(img1, img2) + self.txt_estimator.learning_loss(txt1, txt2)) / 3.0
        return est_loss

