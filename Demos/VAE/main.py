import os
import torch
import logging
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from model import VAE
logging.basicConfig(filename='train.log', format='%(asctime)s %(filename)s %(levelname)s %(message)s', datefmt='%a %d %b %Y %H:%M:%S', filemode='w', level=logging.INFO)

# prepare data
def get_dataloader(root_dir = './datasets'):
    train_data = CIFAR10(root=root_dir, train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test_data = CIFAR10(root=root_dir, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    train_loader = DataLoader(dataset=train_data, batch_size=4, num_workers=4, shuffle=True)
    return train_loader

# train function with fixed lr
def train(train_loader, device, model, lr, n_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    prev_loss = float('inf')
    for epoch in range(n_epochs):
        loss_sum = 0
        for batch_id, (data, target) in enumerate(train_loader):
            data = data.to(device)
            _,_,_,loss = model(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= len(train_loader.dataset)
        if loss_sum < prev_loss:
            torch.save(model.state_dict(), 'Demos/VAE/model.pth')
            logging.info('save model weights!')
        prev_loss = loss_sum
        logging.info(f'epoch {epoch} complete: loss {loss_sum}.')

def main():
    model = VAE().cuda()
    device = 'cuda:0'
    lr = 0.005; n_epochs=10
    train_loader = get_dataloader()

    if os.path.exists('.Demos/VAE/model.pth'):
        model.load_state_dict(torch.load('.Demos/VAE/model.pth', 'cuda:0'))
    
    train(train_loader, device, model, lr, n_epochs)


if __name__ == '__main__':
    main()