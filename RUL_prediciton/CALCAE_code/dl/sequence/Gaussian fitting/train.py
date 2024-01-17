from dl.sequence.function import *
from torch.autograd import Variable

lr = 1e-2  # learning rate
k = 3
stop = 1e-3
epochs = 100000


class Net(nn.Module):
    def __init__(self, k=3, eps=1e-6):
        super(Net, self).__init__()
        self.eps = eps
        self.w = torch.nn.Parameter(torch.randn((3, k)), requires_grad=True)

    def forward(self, x):
        out = 0
        for i in range(k):
            out += self.w[0, i] * torch.exp(-torch.pow((x - self.w[1, i]) / (self.w[2, i] + self.eps), 2))
        return out


def train(data, k=3, lr=1e-4, stop=1e-3, epochs=100000, device=device):
    model = Net(k=k)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_list = []
    epoch = 0
    while True:
        train_data = np.random.permutation(data)
        x, y = train_data[:, 0], train_data[:, 1]
        X, y = np.reshape(x, (-1, 1)).astype(np.float32), np.reshape(y, (-1, 1)).astype(np.float32)
        X, y = Variable(torch.from_numpy(X)).to(device), Variable(torch.from_numpy(y)).to(device)
        y_ = model(X)
        loss = criterion(y_, y)
        # print(loss.detach().numpy())
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if epoch % 100 == 0:
            loss_list.append(loss.detach().numpy())

        if (loss.detach().numpy() < stop) or (epoch > epochs):
            break

        epoch += 1

    return model, loss_list


def predict(model, x, device=device):
    x = np.array(x).astype(np.float32)
    x = torch.from_numpy(x).to(device)
    model.eval()
    torch.no_grad()
    y_ = model(x)
    return y_.detach().numpy()


if __name__ == '__main__':
    seed = 0
    print('seed: ', seed)
    setup_seed(seed=seed)

    i = 0
    name = Battery_list[i]
    df = Battery[name]
    x, y = np.array(df['cycle']), np.array(df['capacity'])

    model, loss_list = train(data=np.c_[x, y], k=k, lr=lr, stop=stop, epochs=epochs, device=device)
    print('Optimal parameters: ', model.w)
