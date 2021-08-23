#  Based on https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

import torch

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(epochs, dataloader, optimizer, criterion, net, path, is_target:bool):
    for epoch in range(epochs):

        running_loss = 0.0
        print('Epoch ' + str(epoch + 1))
        for i, data in enumerate(dataloader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.values()
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if is_target:
                outputs, _ = net(inputs)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 batches
                #print('[%d, %5d] loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    torch.save(net.state_dict(), path)
    print('Finished Training')

def test(testloader, net, is_target:bool):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:

            images, labels = data.values()
            # images, labels = images.to(device), labels.to(device)

            # calculate outputs by running images through the network
            if is_target:
                outputs, _ = net(images)
            else:
                outputs = net(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    print('Total test samples: ' + str(total))
    print('Correct test samples: ' + str(correct))
    print('Accuracy: %d %%' % (
    100 * correct / total))


def test_class(testloader, net, is_target):
    classes = ['0', '1', '2', '3', '4']

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:

            images, labels = data.values()
            # images, labels = images.to(device), labels.to(device)
            
            if is_target:
                outputs, _ = net(images)
            else:
                outputs = net(images)   

            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / (total_pred[classname] + 0.000001)
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))