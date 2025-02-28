import torch
import torch.nn as nn
from torch.utils.data import TensorDataset , DataLoader
import torchvision.transforms as transforms 




def create_train_dataloader( data_file_path , batch_size = 32 , flag = True ):
    
    '''
        This Function takes the file path and returns the dataloader 
        
        Args : - 

            data_file_path :- File path for dataset
            batch_size :
            
            Flag : As D1 has both images and labels so need to build a separate dataloader for this. 
            
                Flag --> True --> D2 - D20
                Flag --> False --> D1
    
    
    
    
    '''
   
    # Load images
        
    data = torch.load(data_file_path , weights_only= False)
    
    if flag : 
    
        
        
        # extract images
        images  = torch.tensor(data['data'], dtype=torch.float32) # Converting numpy array to tensor
        images  = images.permute(0, 3, 1, 2) # Changing shape from (2500 , 32 ,32 , 3) --> (2500 , 3 , 32 , 32)
        
        
        # Building Dataloader
        dataset = TensorDataset(images) # Converting into Tensor Dataset
        dataloader = DataLoader(dataset , batch_size  = batch_size , shuffle=True) # Data Loader
        
       
        return dataloader 

    else :
        
        # Converting numpy array to tensor
        images , labels = torch.tensor(data['data'], dtype=torch.float32) , torch.tensor(data['targets'] ,       dtype=torch.long)
        
        images  = images.permute(0, 3, 1, 2) # Changing shape from (2500 , 32 ,32 , 3) --> (2500 , 3 , 32 , 32)
        
        # Building Dataloader
        dataset = TensorDataset(images , labels) # Converting into Tensor Dataset
        dataloader = DataLoader(dataset , batch_size  = batch_size , shuffle=True) # Data Loader
        
       
        
        return dataloader 
        
        
def create_val_dataloader(data_file_path , batch_size = 32):
    
    # Load images
        
    data = torch.load(data_file_path , weights_only= False)
    
    # Converting numpy array to tensor
    images , labels = torch.tensor(data['data'], dtype=torch.float32) , torch.tensor(data['targets'] , dtype=torch.long)
    
    images  = images.permute(0, 3, 1, 2) # Changing shape from (2500 , 32 ,32 , 3) --> (2500 , 3 , 32 , 32)
    
    # Building Dataloader
    dataset = TensorDataset(images , labels) # Converting into Tensor Dataset
    dataloader = DataLoader(dataset , batch_size  = batch_size , shuffle = False) # Data Loader
    
    
    
    return dataloader 
    
    
    