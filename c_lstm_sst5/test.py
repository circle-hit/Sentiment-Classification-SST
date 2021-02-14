import torch
from train import *
model = 
print("Start testing")
model.load_state_dict(torch.load('model/trained.pth'))
_, test_acc = evaluate(model, x_test, device)
print('Test acc: {:.3f}'.format(test_acc))
with open("result.txt", 'a+') as f:
    f.write(str(test_acc))
    f.write('\n')
