import torch
from torch.utils.data.dataset import TensorDataset
import torch.nn.functional as F
import copy
import os


#Prepare data for Attack Model
def prepare_attack_data(model,
                        iterator,
                        device,
                        top_k=False,
                        test_dataset=False):
    
    attackX = []
    attackY = []
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in iterator:
            # Move tensors to the configured device
            inputs = inputs.to(device)
            
            #Forward pass through the model
            outputs = model(inputs)
            
            #To get class probabilities
            posteriors = F.softmax(outputs, dim=1)
            if top_k:
                #Top 3 posterior probabilities(high to low) for train samples
                topk_probs, _ = torch.topk(posteriors, 3, dim=1)
                attackX.append(topk_probs.cpu())
            else:
                attackX.append(posteriors.cpu())

            #This function was initially designed to calculate posterior for training loader,
            # but to handle the scenario when trained model is given to us, we added this boolean
            # to different if the dataset passed is training or test and assign labels accordingly    
            if test_dataset:
                attackY.append(torch.zeros(posteriors.size(0),dtype=torch.long))
            else:
                attackY.append(torch.ones(posteriors.size(0), dtype=torch.long))
        
    return attackX, attackY
    
def train_per_epoch(attack_type,
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    attribute = None,
                    bce_loss=False):
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0

    #testing
    count = 0
    
    model.train()
    for _ , data in enumerate(train_loader,0):

        # Move tensors to the configured device
        features = data[0].to(device)
        target = data[1].to(device)

        # Forward pass
        if attack_type == 'MemInf' or attribute == 'race':
            outputs = model(features)
        else:
            outputs, _ = model(features)
        
        
            
        if bce_loss:
            #For BCE loss
            loss = criterion(outputs, target.unsqueeze(1))
        else:
            loss = criterion(outputs, target)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Record Loss
        epoch_loss += loss.item()

        #Get predictions for accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    #Per epoch valdication accuracy calculation
    epoch_acc = correct / total
    epoch_loss = epoch_loss / total

    return epoch_loss, epoch_acc

def val_per_epoch(attack_type,
                model,
                val_loader,
                criterion,
                device,
                attribute=None,
                bce_loss=False):

    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total =0

    model.eval()
    with torch.no_grad():
        for _,data in enumerate(val_loader,0):
            # Move tensors to the configured device
            features = data[0].to(device)
            target = data[1].to(device)
           

            #Forward Pass
            if attack_type == 'MemInf' or attribute == 'race':
                outputs = model(features)
            else:
                outputs, _ = model(features)

            #Caluclate the loss
            if bce_loss:
                #For BCE loss
                loss = criterion(outputs, target.unsqueeze(1))
            else:
                loss = criterion(outputs,target)
                
            #record the loss
            epoch_loss += loss.item()
            
            #Check Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        #Per epoch valdication accuracy and loss calculation
        epoch_acc = correct / total
        epoch_loss = epoch_loss / total
    
    return epoch_loss, epoch_acc

###############################
# Training Attack Model
###############################
def train_attack_model(attack_type,
                    model,
                    dataset,
                    criterion,
                    optimizer,
                    lr_scheduler,
                    device,
                    model_path='./model',
                    epochs=10,
                    b_size=20,
                    num_workers=1,
                    verbose=False,
                    earlystopping=False):
        
    n_validation = 1000 # number of validation samples
    best_valacc = 0
    stop_count = 0
    patience = 5 # Early stopping

    path = os.path.join(model_path,'best_attack_model.ckpt')
        
    train_loss_hist = []
    valid_loss_hist = []
    val_acc_hist = []
    attr_label = None

    train_X, train_Y = dataset
    
    #Contacetnae list of tensors to a single tensor
    train_X = torch.cat(train_X)
    train_Y = torch.cat(train_Y)

    #Create Attack Dataset
    attackdataset = TensorDataset(train_X,train_Y)
 
        
    print('Shape of Attack Feature Data : {}'.format(train_X.shape))
    print('Shape of Attack Label Data : {}'.format(train_Y.shape))
    print('Total Train Samples for Attack Model : [{}]'.format(len(attackdataset)))
    print('Epochs [{}] and Batch size [{}] for Attack Model training'.format(epochs,b_size))

       
    #Create Train and Validation Split
    n_train_samples = len(attackdataset) - n_validation
    train_data, val_data = torch.utils.data.random_split(attackdataset, 
                                                        [n_train_samples, n_validation])
        

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=b_size,
                                                shuffle=True,
                                                num_workers=num_workers)
        
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                            batch_size=b_size,
                                            shuffle=False,
                                            num_workers=num_workers)

    if attack_type == 'AttrInf': #Attribute Inference
        attr_label = 'race'
    
    print('----Attack Model Training------')   
    for i in range(epochs):
            
        train_loss, train_acc = train_per_epoch(attack_type, model, train_loader, criterion, optimizer, device,attr_label)
        valid_loss, valid_acc = val_per_epoch(attack_type, model, val_loader, criterion, device, attr_label)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)
        
        lr_scheduler.step()
        
        print ('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
                 .format(i+1, epochs, train_loss, train_acc*100, valid_loss, valid_acc*100))

        if earlystopping: 
            if best_valacc<=valid_acc:
                print('Saving model checkpoint')
                best_valacc = valid_acc
                #Store best model weights
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, path)
                stop_count = 0
            else:
                stop_count+=1
                if stop_count >=patience: #early stopping check
                    print('End Training after [{}] Epochs'.format(epochs+1))
                    break
        else:#Continue model training for all epochs
            print('Saving model checkpoint')
            best_valacc = valid_acc
            #Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, path)
            
    return best_valacc
    


###################################
# Training Target and Shadow Model
###################################            
def train_model(attack_type,
                model,
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
                top_k=False,
                earlystopping=False,
                is_target=False
                ):
    
    best_valacc = 0
    patience = 5 # Early stopping
    stop_count= 0
    train_loss_hist = []
    valid_loss_hist = []
    val_acc_hist = []
    attr_label = None

    model_obj = model # Use it later to get feature extractions 
    
    if is_target:
        print('----Target model training----')
    else:
        print('---Shadow model training----')
    
    #Path for saving best target and shadow models
    if is_target:
        target_path = os.path.join(model_path,'best_target_model.ckpt')
    else:
        shadow_path = os.path.join(model_path,'best_shadow_model.ckpt')
    
    if attack_type == 'AttrInf':
        attr_label = 'gender'
    

    for epoch in range(num_epochs):
        
        train_loss, train_acc = train_per_epoch(attack_type, model, train_loader, loss, optimizer, device, attr_label)
        valid_loss, valid_acc = val_per_epoch(attack_type, model, val_loader, loss, device, attr_label)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)

        scheduler.step()

        print ('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
                   .format(epoch+1, num_epochs, train_loss, train_acc*100, valid_loss, valid_acc*100))
        
        if earlystopping:
            if best_valacc<=valid_acc:
                print('Saving model checkpoint')
                best_valacc = valid_acc
                #Store best model weights
                best_model = copy.deepcopy(model.state_dict())
                if is_target:
                    torch.save(best_model, target_path)
                else:
                    torch.save(best_model, shadow_path)
                stop_count = 0
            else:
                stop_count+=1
                if stop_count >=patience: #early stopping check
                    print('End Training after [{}] Epochs'.format(epoch+1))
                    break
        else:#Continue model training for all epochs
            print('Saving model checkpoint')
            best_valacc = valid_acc
            #Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            if is_target:
                torch.save(best_model, target_path)
            else:
                torch.save(best_model, shadow_path)
    
    
    if is_target:
        print('----Target model training finished----')
        print('Validation Accuracy for the Target Model is: {:.2f} %'.format(100* best_valacc))
    else:
        print('----Shadow model training finished-----')
        print('Validation Accuracy for the Shadow Model is: {:.2f} %'.format(100* best_valacc))

    if attack_type == 'AttrInf':
        #Get Feature extraxtions for Target 
        attack_X, attack_Y = get_feature_representation(model,model_path,test_loader,device,is_target)
    

    if is_target:
        print('----LOADING the best Target model for Test----')
        model.load_state_dict(torch.load(target_path))
    else:
        print('----LOADING the best Shadow model for Test----')
        model.load_state_dict(torch.load(shadow_path))
    
    
    if attack_type == 'MemInf':
        #As the model is fully trained, time to prepare data for attack model.
        #Training Data for members would come from shadow train dataset, and member inference from target train dataset respectively.
        attack_X, attack_Y = prepare_attack_data(model,train_loader,device,top_k)
        

    # In test phase, we don't need to compute gradients (for memory efficiency)
    print('----Test the Trained Network----')
    model.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        for _, data in enumerate(test_loader,0):
            inputs = data[0].to(device)

            if attack_type == 'MemInf':
                labels = data[1].to(device)
            elif attr_label == 'gender':
                labels = data[1].to(device)

            labels = labels.to(device)
            
            if attack_type == 'MemInf':
                test_outputs = model(inputs)
            else:
                test_outputs, _ =  model(inputs)
            
            #Predictions for accuracy calculations
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if attack_type == 'MemInf':
                # Posterior and labels for non-members
                probs_test = F.softmax(test_outputs, dim=1)
                if top_k:
                    #Take top K posteriors ranked high ---> low
                    topk_t_probs, _ = torch.topk(probs_test, 3, dim=1)
                    attack_X.append(topk_t_probs.cpu())
                else:
                    attack_X.append(probs_test.cpu())
                attack_Y.append(torch.zeros(probs_test.size(0), dtype=torch.long))

        if is_target:
            print('Test Accuracy of the Target model: {:.2f}%'.format(100 * correct / total))
        else:
            print('Test Accuracy of the Shadow model: {:.2f}%'.format(100 * correct / total)) 
            

    return attack_X, attack_Y

def get_feature_representation(model,
                            modelPath,
                            dataloader,
                            device,
                            is_target=False):

    file_path = os.path.join(modelPath,'best_target_model.ckpt')

    #Loaded the previous trained Target model to get Feature Representations
    if is_target:
        print('------Preparing Feature Data and Attr Labels for Attack Test ------')
    else:
        print('------Preparing Feature Data and Attr Labels for Attack Training ------')

    model.load_state_dict(torch.load(file_path))
    
    attackX = []
    attackY = []
    
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            # Move tensors to the configured device
            images = data[0].to(device)
            labels = data[2]
            
            #Forward pass through the model
            _ , outputs = model(images)
            
            attackX.append(outputs.cpu())
            attackY.append(labels)
      
    return attackX, attackY


        
