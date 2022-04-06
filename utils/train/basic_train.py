import torch

def eval_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.shape[0]
            count += data.shape[0]

    test_loss /= len(test_loader.dataset)
    return test_loss


def train_step(model, data, target, optimizer, criterion):
    optimizer.zero_grad()
    y_pred = model(data)
    loss = criterion(y_pred, target)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(model, train_loader, criterion, device, optimizer, scheduler=None):
    model.train()
    losses = []
    for data, target in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = train_step(model, data, target, optimizer, criterion)
        losses.append(loss)
        if scheduler is not None:
            scheduler.step()

    return losses
