import torch

print(torch.__version__)
print(torch.cuda.is_available())
x=torch.randn(3,4,requires_grad=True)
print(x)
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
print(device)
# model.to.device