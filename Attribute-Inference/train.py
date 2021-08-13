import torch

def train(epochs, dataloader, optimizer, criterion, net, path):
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.values()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    torch.save(net.state_dict(), path)
    print('Finished Training')

def test(testloader, net, is_attack:bool):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data.values()
            # calculate outputs by running images through the network
            if is_attack:
                outputs, _ = net(images)
            else:
                outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(correct)
    print(total)
    print('Accuracy: %d %%' % (
    100 * correct / total))