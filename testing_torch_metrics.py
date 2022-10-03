from MedAI_code_segmentation_evaluation import calculate_metrics_torch
import numpy as np
import sklearn.metrics as sk
import torch
import time

device = torch.device('cuda:0')
batch = 10
true = torch.rand((batch,2,300,300)).round().to(device)
pred = torch.rand((batch,2,300,300)).round().to(device)

#--------------- sklearn metrics ------------------
start = time.time()

truenp = true.clone().detach().argmax(dim=1).view(batch, -1).cpu().numpy()
prednp = pred.clone().detach().argmax(dim=1).view(batch, -1).cpu().numpy()
sk_jaccard = [sk.accuracy_score(truenp[i], prednp[i]) for i in range(batch)]
sk_jaccard_mean = np.mean(sk_jaccard)

total_time = time.time() - start
print('SK timer {}'.format(total_time))
#-------------- Torch metrics ----------------
start = time.time()
torch_jaccard_mean = calculate_metrics_torch(true, pred)

total_time = time.time() - start
print('pytorch timer {}'.format(total_time))
print(sk_jaccard_mean, torch_jaccard_mean)
