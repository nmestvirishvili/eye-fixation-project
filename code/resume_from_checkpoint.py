# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:02:01 2022

@author: Natia_Mestvirishvili
"""

#Resume from a checkpoint
eye_fixation_model = Eye_Fixation_CNN(resnet_model, center_bias)
checkpoint = torch.load(os.path.join(checkpoints_path, "Fixation_CNN_epoch4.pt"))
eye_fixation_model.load_state_dict(checkpoint['model_state_dict'])
opt = optim.SGD(eye_fixation_model.parameters(), lr=learning_rate)
opt.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
epochs_count = 30
log_file = "/Users/Natia_Mestvirishvili/Desktop/UHH/Computer Vision II/course_project/1654703714531"
for epoch_resumed in range(epoch+1, epoch+epochs_count+1):
    log_network_performance(log_file, "Epoch: " + str(epoch))
    eye_fixation_model.train()
    train_loss = 0
    for sample in train_data_loader:
        input_image = sample['image']
        pred = eye_fixation_model(input_image)
        loss =  F.binary_cross_entropy_with_logits(pred, sample["fixation"])
        train_loss += loss
        print(loss)
        #visualize_images(sample['image'], sample['fixation'], pred)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    log_network_performance(log_file, "Training loss: " + str(train_loss / len(train_data_loader)))
    print("Epoch:", epoch_resumed, "Training loss:", train_loss / len(train_data_loader))
    
    if (epoch%3 == 0):
        eye_fixation_model.eval()
        with torch.no_grad():
            valid_loss = sum(F.binary_cross_entropy_with_logits(eye_fixation_model(sample['image']), sample['fixation']) for sample in valid_data_loader)
    
        log_network_performance(log_file, "Validation loss: " + str(valid_loss / len(valid_data_loader)))
        print('Epoch:', epoch_resumed, 'Validation loss:', valid_loss / len(valid_data_loader))
    
    #save a checkpoint
    file_name = "MNIST_CNN_epoch"+str(epoch)+".pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': eye_fixation_model.state_dict(),
        'optimizer_state_dict': opt.state_dict()
        }, os.path.join(checkpoints_path, file_name))   
    
    log_network_performance(log_file, json.dumps(opt.state_dict()))