import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Optimizer
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from typing import Dict, Any


# ----------------------------
# Training Class
# ----------------------------
class Trainer:
    """
    Trainer class for handling training state updates, checkpoint saving, and training loop.

    Methods
    -------
    make_train_state(learning_rate: float, model_state_version: str) -> Dict[str, Any]
        Create an initial training state dictionary.
    
    update_train_state(early_stopping_criteria: int, model: nn.Module, train_state: Dict[str, Any], 
                       optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler) -> Dict[str, Any]
        Update training state with early stopping and model checkpointing.
    
    save_state_and_model(train_state: Dict[str, Any], model: nn.Module, optimizer: Optimizer, 
                         scheduler: lr_scheduler._LRScheduler, json_filename: str, model_version: str) -> None
        Save training state and model checkpoint.
    
    training_and_validation(train_dl: DataLoader, val_dl: DataLoader, num_epochs: int, device: torch.device) -> None
        Train and validate the model over a specified number of epochs.

    training(train_dl: DataLoader, criterion: nn.Module, optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler, epoch: int, device: torch.device, bar: tqdm) -> None
        Performs a single epoch of training.
    
    validate(val_dl: DataLoader, criterion: nn.Module, epoch: int, device: torch.device, bar: tqdm) -> None
        Evaluates the model on the validation set.
    
    save_the_time(param: str, file: str, seconds: float) -> None
        Saves elapsed time in a formatted manner to a JSON file.
    """

    def __init__(self, model: nn.Module, device: torch.device, early_stopping_criteria: int, model_state_version: str,save_dir: str, lr: float = 0.001) -> None:
        """
        Trainer class for managing model training, early stopping, and state saving.

        Attributes:
            model (nn.Module): The neural network model.
            device (torch.device): The device to run training on (CPU or GPU).
            early_stopping_criteria (int): Number of epochs with no improvement before stopping.
            model_state_version (str): Filename to save model state.
            train_state (Dict): Dictionary storing training state information.
            lr (float): Learning rate for training.
        """
        super().__init__()

        self.model = model
        self.cuda = device
        self.early_stopping_criteria = early_stopping_criteria
        self.train_state = Trainer.make_train_state(learning_rate = lr, model_state_version = model_state_version, save_dir = save_dir)
        self.lr = lr

    # ----------------------------
    # Create state dictionary
    # ----------------------------
    @staticmethod
    def make_train_state(learning_rate: float, model_state_version: str, save_dir: str):
        """
        Create an initial training state dictionary.

        Parameters
        ----------
        learning_rate : float
            Learning rate used for training.
        model_state_version : str
            File path for saving the model state.

        Returns
        -------
        train_state : dict
            Dictionary containing initial training state values.
        """
        return {
            "trainable_parameters" : 0,
            "batch_size" : None,
            "optimizer" : None,
            "scheduler" : None,
            "criterion" :   None,
            "stop_early_criteria": None,
            "stop_early": False,
            "early_stopping_step": 0,
            "early_stopping_best_val": 1e8,
            "learning_rate": learning_rate,
            "epoch_index": 0,
            "model_version": model_state_version,
            "save_dir": save_dir,
            "device" : None,
            "total_time" : None,
            "time_per_epoch" : None,
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

    # ----------------------------
    # Update the state dictionary
    # ----------------------------
    @staticmethod
    def update_train_state(early_stopping_criteria: int, model: nn.Module, train_state: dict, optimizer: Optimizer, 
                           scheduler:lr_scheduler._LRScheduler) -> Dict[str, Any]:
        """
        Update training state with early stopping and model checkpointing.

        Parameters
        ----------
        early_stopping_criteria : int
            Number of epochs without improvement before stopping training.
        model : nn.Module
            The neural network model being trained.
        train_state : dict
            Dictionary containing training state information.
        optimizer : Optimizer
            Optimizer used for model training.
        scheduler : lr_scheduler._LRScheduler
            Learning rate scheduler.

        Returns
        -------
        train_state : dict
            Updated training state dictionary.
        """

        if train_state['epoch_index'] == 0 or train_state['val_loss'][-1] < train_state['early_stopping_best_val']:
            
            ## Save cheeckpoints the dictionary to a JSON file and the best model to a pt file
            Trainer.save_state_and_model(train_state, model, optimizer, scheduler, 
                                         json_filename=train_state['model_version'] + '_train_state.json', 
                                         model_version=train_state['model_version'] + '.pt', save_dir = train_state['save_dir'])

            #Restart stopping_step
            train_state['early_stopping_step'] = 0
            #update the best value loss
            train_state['early_stopping_best_val'] = train_state['val_loss'][-1]
        else:
            
            #Save the actual train state for consulting during the training
            with open(train_state['model_version'] +'_train_state_temporary.json', 'w') as json_file:
                json.dump(train_state, json_file, indent=4)

            train_state['early_stopping_step'] += 1

        train_state['stop_early'] = train_state['early_stopping_step'] >= early_stopping_criteria
        return train_state

    # ----------------------------
    # Save the state dictionary
    # ----------------------------
    @staticmethod
    def save_state_and_model(train_state: dict, model: nn.Module, optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler,
                             json_filename: str, model_version: str, save_dir: str) -> None:
        """
        Save training state and model checkpoint.

        Parameters
        ----------
        train_state : dict
            Dictionary containing training state information.
        model : nn.Module
            The neural network model being trained.
        optimizer : Optimizer
            Optimizer used for model training.
        scheduler : lr_scheduler._LRScheduler
            Learning rate scheduler.
        json_filename : str
            Filename to save training state as JSON.
        model_version : str
            Filename to save model checkpoint.
        save_dir : str
            Path to save all checkpoints

        Returns
        -------
        None
        """
        os.makedirs(save_dir, exist_ok=True)

        # path's
        json_filepath = os.path.join(save_dir, json_filename)
        model_filepath = os.path.join(save_dir, f"{model_version}.pt")
        optimizer_filepath = os.path.join(save_dir, f"{model_version}_optimizer_state.pth")
        scheduler_filepath = os.path.join(save_dir, f"{model_version}_scheduler_state.pth")

        #Save Json
        with open(json_filepath, 'w') as json_file:
            json.dump(train_state, json_file, indent=4)

        # Salvar estados do otimizador e do scheduler
        torch.save(optimizer.state_dict(), optimizer_filepath)
        torch.save(scheduler.state_dict(), scheduler_filepath)

        # Save the PyTorch model to a .pt file
        model.save_checkpoint(path = save_dir, version = model_version)


    # ----------------------------
    # Training and Validation Loop
    # ----------------------------
    def training_and_validation(self, train_dl: DataLoader, val_dl: DataLoader, num_epochs: int, device: torch.device) -> None:
        """
        Train and validate the model.

        Parameters
        ----------
        train_dl : DataLoader
            Training data loader.
        val_dl : DataLoader
            Validation data loader.
        num_epochs : int
            Number of epochs to train.
        device : torch.device
            Device to use for training (CPU/GPU).
        
        Returns
        -------
        None
        """
        # Loss Function, Optimizer and Scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                      steps_per_epoch=int(len(train_dl)),
                                                      epochs=num_epochs,
                                                      anneal_strategy='linear')
        
        print("\n***** PARAMETERS AND ARGUMENTS *****\n", flush = True)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model Parameters", flush=True)
        print(f'The model has {num_params:,} trainable parameters \n', flush=True)
        print("Models Arguments", flush=True)
        print(
            f"Called: model = {type(self.model).__name__}, opt = {type(optimizer).__name__} (lr = {self.lr:.6f}), "
            f"epochs = {num_epochs}, device = {device}\n"
            f"        stop_criteria = {self.early_stopping_criteria}, criterion = {type(criterion).__name__}, "
            f"scheduler = {type(scheduler).__name__}",
            flush=True)
        
        self.train_state.update({"optimizer": type(optimizer).__name__, 
                                 "scheduler": type(scheduler).__name__, 
                                 "criterion": type(criterion).__name__,
                                 "device": str(self.cuda),
                                 "batch_size": train_dl.batch_size,
                                 "trainable_parameters": num_params,
                                 "stop_early_criteria": self.early_stopping_criteria})

        print("\n***** STARTING TRAINING *****\n", flush = True)

        train_bar = tqdm(desc=f'UrbanSound_Classifier_training',
                              total=len(train_dl), 
                              position=1, 
                              leave=False)
        val_bar = tqdm(desc=f'Validating_UrbanSound_Classifier',
                      total=len(val_dl), 
                      position=1, 
                      leave=False)

       
        start_time_sec = time.time()
        # Repeat for each epoch
        for epoch in range(num_epochs):
            train_bar.reset()
            val_bar.reset()            

            self.training(train_dl, criterion, optimizer, scheduler, epoch, device, bar = train_bar)
            self.validate(val_dl, criterion, epoch, device, bar = val_bar)    
            self.update_train_state(self.early_stopping_criteria, self.model, self.train_state, optimizer, scheduler)    
            
            if self.train_state['stop_early']:
                print(f"Early stopping at epoch {epoch}", flush = True)
                break

        print("\n***** TRAINING COMPLETED *****\n", flush = True)
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / num_epochs
        print(f'\nTime total: {total_time_sec:.2f} sec', flush=True)
        print(f'Time per epoch: {time_per_epoch_sec:.2f} sec', flush=True)
        
        file = os.path.join(self.train_state['save_dir'], self.train_state['model_version'] + '_train_state.json')
        Trainer.save_the_time('total_time',file , total_time_sec)
        Trainer.save_the_time('time_per_epoch',file, time_per_epoch_sec)
        
        temp_file = self.train_state['model_version'] +'_train_state_temporary.json'
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        print(f'\nModel and state saved: {self.train_state['save_dir']}', flush = True)

    # ----------------------------
    # Training Loop
    # ----------------------------         
    def training(self, train_dl: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, device: torch.device, 
                 bar: tqdm) -> None:
        """
        Perform a single epoch of training.
        
        Parameters
        ----------
        train_dl : DataLoader
            DataLoader for the training dataset.
        criterion : nn.Module
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer for training the model.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.
        epoch : int
            Current training epoch.
        device : torch.device
            Device on which the model is trained.
        bar : tqdm
            Progress bar for tracking training.
        """
                
        running_loss, total_prediction, correct_prediction = 0.0, 0.0, 0.0
        # Repeat for each batch in the training set

        self.model.train()
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Reset the gradients. This is done so that the gradients from the previous batch
            # are not used in the next step.
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            
            #Compute the Losses
            # The lossess is compute on the model output an the target
            loss = criterion(outputs, labels)

            #  Backpropagate the loss.
            loss.backward()
            
            # Update the model parameters. This is done by taking a step in the direction of the gradient.
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            bar.set_postfix({'loss': round(running_loss/(bar.n + 1),3), "epoch" : epoch + 1})
            bar.update()

        # update the stats at the end of the epoch
        loss = running_loss / len(train_dl)
        acc = correct_prediction/total_prediction

        # Append the loss to the list of losses, so that the loss can be computed for this epoch.
        self.train_state['train_loss'].append(loss)
        self.train_state['train_accuracy'].append(acc)

    # ----------------------------
    # Validation Loop
    # ----------------------------
    def validate(self, val_dl: DataLoader, criterion: nn.Module, epoch: int, device: torch.device, bar: tqdm) -> None:
        """
        Evaluate the model on the validation set.
        
        Parameters
        ----------
        val_dl : DataLoader
            DataLoader for the validation dataset.
        criterion : nn.Module
            Loss function.
        epoch : int
            Current validation epoch.
        device : torch.device
            Device on which the model is evaluated.
        bar : tqdm
            Progress bar for tracking validation.
        """
        
        running_loss, total_prediction, correct_prediction = 0.0, 0.0, 0.0
        
        # Disable gradient updates
        self.model.eval()
        with torch.no_grad():

            for data in val_dl:
                # Get the input features and target labels, and put them o"""  """n the GPU
                inputs, labels = data[0].to(device), data[1].to(device)
                
                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s
                
                # Get predictions
                outputs = self.model(inputs)
                
                # calculate validation loss
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs,1)
                
                # Count of predictions that matched the target label
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]
                
                bar.set_postfix({'loss': round(running_loss / (bar.n + 1),3),
                                   "epoch" : epoch + 1})
                bar.update()

        loss = running_loss / len(val_dl)    
        acc = correct_prediction/total_prediction
        self.train_state['val_loss'].append(loss)
        self.train_state['val_accuracy'].append(acc)

    # ----------------------------
    # Convert and Save time
    # ----------------------------
    @staticmethod
    def save_the_time(param: str, file: str, seconds: float) -> None:
        """
        Save the formatted elapsed time in a JSON file.

        Parameters
        ----------
        param : str
            The key under which the time will be stored in the JSON file.
        file : str
            The path to the JSON file where the time will be saved.
        seconds : float
            The elapsed time in seconds.

        Returns
        -------
        None
        """
        SECONDS_IN_MINUTE = 60
        SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
        SECONDS_IN_DAY = SECONDS_IN_HOUR * 24
        SECONDS_IN_YEAR = SECONDS_IN_DAY * 365

        years, seconds = divmod(seconds, SECONDS_IN_YEAR)
        days, seconds = divmod(seconds, SECONDS_IN_DAY)
        hours, seconds = divmod(seconds, SECONDS_IN_HOUR)
        minutes, seconds = divmod(seconds, SECONDS_IN_MINUTE)
        
        result = []
        if years: result.append(f"{round(years,2)} years")
        if days: result.append(f"{round(days,2)} days")
        if hours: result.append(f"{round(hours,2)} hours")
        if minutes: result.append(f"{round(minutes,2)} minutes")
        if seconds: result.append(f"{round(seconds,2)} seconds")


        with open(file, 'r+') as f:
            data = json.load(f)
            data[str(param)] = ', '.join(result)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
