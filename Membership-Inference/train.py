import torch
import time
import copy
import os
import numpy as np

def prepare_attack_data(model,train_loader,device):

    attack_X = []
    attack_Y = []

    model.eval()
    #Collect attack data
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #Top 3 posterior probabilities(high to low) for train samples
            topkvalues, _ = torch.topk(outputs.data, 3, dim=1)
            attack_X.append(topkvalues)
            attack_Y.append(torch.ones(len(labels)))

    return attack_X, attack_Y
    
def train_one_epoch(model,
                    train_iterator,
                    criterion,
                    optimizer,
                    device):
    train_loss = 0
    train_acc = 0
    correct = 0
    total = 0

    model.train()
    for i, (images, labels) in enumerate(train_iterator):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Record Loss
        train_loss += loss.item()

        #Predictions for accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    #Per epoch valdication accuracy calculation
    train_acc = correct / total
    train_loss = train_loss / total

    return train_loss, train_acc

def evaluate(model,
            val_iterator,
            criterion,
            device):

    val_loss = 0
    val_acc = 0
    correct = 0
    total =0

    model.eval()
    with torch.no_grad():
        for images, labels in val_iterator:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            #Caluclate the loss
            loss = criterion(outputs,labels)
            #record the loss
            val_loss += loss.item()

            #Predictions for accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        #Per epoch valdication accuracy calculation
        val_acc = correct / total
        val_loss = val_loss / total
    
    return val_loss, val_acc

###############################
# Training Attack Model
###############################
def train_attack_model(shadowX, 
                    shadowY,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    device,
                    epochs=10,
                    batch_size=20,
                    n_hidden=100):
        
        print(shadowX.shape)
        print(shadowY.shape)

        correct = 0
        total = 0
        train_loss = 0

        total = shadowX.shape[0]
        iteractions_per_epoch = max(total / batch_size, 1)

        #Training and Validation split for Attack Model
        valid_ratio = 0.95
        n_train = int(total * valid_ratio)
        n_val_samples = total - n_train
        train_data, valid_data = torch.utils.data.random_split(shadowX, 
                                                               [n_train, n_val_samples])

        model.train()
        for i in range(epochs):
            for it in range(iteractions_per_epoch):
                #Indices list for random subset of training samples
                indx = np.random.choice(n_train, batch_size, replace=False)

                #Training batch and their lablels
                X_batch = train_data[indx].to(device)
                Y_batch = shadowY[indx].to(device)

                #Forward Pass
                outputs = model(X_batch)

                #Caluclate the loss
                loss = criterion(outputs,Y_batch)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()       
            
                #Record Loss
                train_loss += loss.item()

                #Predictions for accuracy calculation
                _, predicted = torch.max(outputs.data, 1)
                total += Y_batch.shape[0]
                correct += (predicted == Y_batch).sum().item()

            scheduler.step()

            #Per epoch valdication accuracy calculation
            train_acc = correct / total
            train_loss = train_loss / total

###################################
# Training Target and Shadow Model
# #################################            
def train_model(model,
                train_loader,
                val_loader,
                test_loader,
                loss,
                optimizer,
                scheduler,
                device,
                model_path,
                verbose=False,
                num_epochs=50,
                is_target=False):
    
    best_valacc = 0
    patience = 5 # Early stopping
    train_loss_hist = []
    valid_loss_hist = []
    
    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(model, train_loader, loss, optimizer, device)
        valid_loss, valid_acc = evaluate(model, val_loader, loss, device)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)

        scheduler.step()

        print ('Epoch [{}/{}], Train Loss: {:.3f}, Train Acc: {:.2f}% | Val. Loss: {:.3f}, Val. Acc: {:.2f}%'
                   .format(epoch+1, num_epochs, train_loss, train_acc*100, valid_loss, valid_acc*100))

        if best_valacc<=valid_acc:
            best_valacc = valid_acc
            #Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            target_path = os.path.join(model_path,'best_target_model.ckpt')
            torch.save(best_model, target_path)
            stop_count = 0
        else:
            stop_count+=1
            if stop_count >=patience: #early stopping check
                if verbose:print('End Training after [{}] Epochs'.format(epoch+1))
                #Save the best model
                print('SAVING the best model')
                if is_target:
                    target_path = os.path.join(model_path,'best_target_model.ckpt')
                    torch.save(best_model, target_path)
                else:
                    shadow_path = os.path.join(model_path,'best_shadow_model.ckpt')
                    torch.save(best_model, shadow_path)

                break

    if is_target:
        print('Validation Accuracy for the Best Target Model is: {:.2f} %'.format(100* best_valacc))
    else:
        print('Validation Accuracy for the Best Shadow Model is: {:.2f} %'.format(100* best_valacc))

    #Test the model
    if is_target:
        print('LOADING the best Target model for Test')
        model.load_state_dict(torch.load(target_path))
    else:
        print('LOADING the best Shadow model for Test')
        model.load_state_dict(torch.load(shadow_path))
    
    #Feature vector and labels for attack model from target/shadow training samples
    dataX, dataY = prepare_attack_data(model,train_loader,device)

    # In test phase, we don't need to compute gradients (for memory efficiency)
    print("Test the Trained Network")
    model.eval() #Prepare for evaluation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            #Top 3 posterior probabilities(high to low) for test samples
            topkvalues, _ = torch.topk(outputs.data, 3, dim=1)
            dataX.append(topkvalues)
            dataY.append(torch.zeros(len(labels)))

            #Predictions for accuracy calculations
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if not is_target and total == 5000:
                #Using only 5000 test samples for Shadow Model 
                break
        
        if is_target:
            print('Test Accuracy of the Target model: {:.2f}%'.format(100 * correct / total))
        else:
            print('Test Accuracy of the Shadow model on {} images: {:.2f}%'.format(total, 100 * correct / total)) 
    
    # print(dataX)
    # print(dataY)
    # dataX = dataX.cpu().numpy()
    # dataY = dataY.cpu().numpy()
    dataX = torch.stack(dataX)
    dataY = torch.stack(dataY)
    
    return dataX, dataY

    

        
