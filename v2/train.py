from comet_ml import Experiment
import torch, torchaudio
from torch import nn, optim
import numpy as np
from tqdm import tqdm

from neuralnet import Model
from utils import *


class SpeechRecognition:
    # Main speech recognition class

    def __init__(self):
        # Number of residual CNN layers
        self.n_cnn_layers = 3 
        # Number of bidirectional GRU layers
        self.n_rnn_layers = 5 
        # Number of neurons for recurrent layer
        self.rnn_dim = 512
        # Length of model ouput vector
        self.n_classes = 29 
        # Number of features for the CNN
        self.n_feats = 128 
        # Size of gap between convolutions
        self.stride = 2 
        # Percent of neurons to disables
        self.dropout = 0.1 
        # Backpropogation step size
        self.learning_rate = 5e-4
        # Amount of iterations over the entire dataset
        self.epochs = 1
        # Number of samples seen at a time
        self.batch_size = 10 
        # Training dataset
        self.train_url = "train-clean-100"
        # Validation dataset  
        self.test_url = "test-clean" 
        
        self.model_path = "model.pt"

        # Instantiate model with defined hyper-parameters
        self.model = Model(n_cnn_layers=self.n_cnn_layers, n_rnn_layers=self.n_rnn_layers,
                           rnn_dim=self.rnn_dim, n_class=self.n_classes, n_feats=self.n_feats, 
                           stride=self.stride, dropout=self.dropout
        )
        
        # Adam optimiser to perform backpropogation
        self.optimiser = optim.AdamW(self.model.parameters(), self.learning_rate)
        
        # Loss function to calculate model error
        # Connectionist Temporal Classification loss 
        # Able to align audio frames to characters and then form words (or phonemes)
        # CTC loss is used since it doesnt require that the predicted vector 
        # And the label vector to be the same length, which is ideal since otherwise
        # The labels would need to be padded to match the number of time steps
        # `blank` character corresponds to index 28 in the character map
        self.blank_index = 28
        self.loss_fn = nn.CTCLoss(blank=self.blank_index) 

        # Training dataset
        self.train_dataset = torchaudio.datasets.LIBRISPEECH(
            "C:\\Users\\User\\Desktop", url=self.train_url, download=True
        )

        # Testing/validation dataset
        self.test_dataset = torchaudio.datasets.LIBRISPEECH(
            "C:\\Users\\User\\Desktop", url=self.test_url, download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, collate_fn=lambda x: preprocess(x, data_type="train")
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, 
            shuffle=False, collate_fn=lambda x: preprocess(x, data_type="valid")
        )

    def train_model(self, try_load=True):
        # Train the model on the LibriSpeech dataset

        def _train():
            # Update model's weights (train) for a single epoch 

            # Log the epoch with comet
            with self.logger.train():
                print(f"\nBeginning epoch {epoch}/{self.epochs}...")
                self.train_loader = iter(self.train_loader)
                for index in tqdm(range(self.train_data_len)):
                    try:
                        batch = next(self.train_loader)
                    except OSError:
                        continue
                    # Unpack each batch
                    spectrograms, labels, x_lengths, y_lengths = batch
                    # Set current gradients to zero
                    self.optimiser.zero_grad()
                    output = self.model(spectrograms)
                    # Swap first and second dimension (batch, time)
                    output = output.transpose(0, 1)

                    # Compute loss
                    loss = self.loss_fn(output, labels, x_lengths, y_lengths)
                    # Perform a backwards pass and update the gradients
                    loss.backward()

                    # Update the epoch's loss on comet
                    self.logger.log_metric("Loss", loss.item(), step=self.iterations)
                print(f"Completed training for epoch {epoch}/{self.epochs}")

        def _test():
            # Used to evaluate the loss of the model per batch across a single epoch
            self.model.eval()
            # Average test_loss over the epoch
            test_loss = 0
            test_cer, test_wer = [], []
            with self.logger.test(), torch.no_grad():
                for index, data in enumerate(self.test_loader):
                    # Similar steps to training mode
                    spectrograms, labels, x_lengths, y_lengths = batch
                    output = self.model(spectrograms)
                    output.transpose(0, 1)
                    loss = self.loss_fn(output, labels, x_lengths, y_lengths)
                    # Calculate the average loss over the epoch
                    test_loss += loss / self.test_data_len
                    # Get the highest probability for each time step
                    decoded_preds, decoded_targets = self._greedy_decoder(
                        output.transpose(0, 1), labels, y_lengths
                    )
                    # Calculate WER and CER
                    for x, y in zip(decoded_preds, decoded_targets):
                        test_cer.append(cer(y, x))
                        test_wer.append(wer(y, x))
            # Calulate the average metrics
            avg_cer = sum(test_cer) / len(test_cer)
            avg_wer = sum(test_wer) / len(test_wer)
            # Update metrics on comet
            self.logger.log_metric("Test-Loss", test_loss, step=self.iteration)
            self.logger.log_metric("CER", avg_cer, step=self.iteration)
            self.logger.log_metric("WER", avg_wer, step=self.iteration)
            
            print(f"""Across validation data:
                        - Average loss: {test_loss},
                        - Average CER: {avg_cer},
                        - Average WER: {avg_wer}
            """)

        # Set the model into training mode
        self.model.train()
        self.train_data_len = len(self.train_loader.dataset)
        self.test_data_len = len(self.test_loader.dataset)
        self.iterations = 0

        # Set up experiment with `comet.ml` to display loss and accuracy graphically
        comet_api_key = "oLcP8rq23uQf8MkMYIhYFnnXc" 
        project_name = "speechrecognition"
        experiment_name = "speechrecognition-colab"

        self.logger = Experiment(
            api_key=comet_api_key, project_name=project_name, parse_args=False
        )
        self.logger.set_name(experiment_name)
        self.logger.display()

        if try_load:
            self.model.load_state_dict(torch.load(self.model_path))

        for epoch in range(1, self.epochs + 1):
            _train()
            _test()

            # Each epoch is a model save checkpoint 
            # `state_dict` stores the state of the model, optimiser and the current epoch    
            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "epoch": epoch
            }

            torch.save(state_dict, self.model_path)

    def _greedy_decoder(self, 
                        output, # Softmax outputs from the model
                        labels, # Target labels
                        label_lengths, # Length of each target label
                        blank_label=None, # index of the `blank` character in CTC
                        collapse_repeated=True # Remove repeated characters (i.e. `hellllo` -> `helo`)
    ):
        # Get highest probabilites for each time step (3rd dimension)
        if blank_label is None:
            blank_label = self.blank_index

        arg_maxes = torch.argmax(output, dim=2)
        print(arg_maxes.shape)
        decodes = []
        targets = []
        for i, args in enumerate(arg_maxes):
            decode = []
            targets.append(text_process.int_to_text(labels[i][:label_lengths[i]].tolist()))
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j-1]:
                        continue
                    decode.append(index.item())
            decodes.append(text_targets.append(text_process.int_to_text(decode)))
        return decodes, targets

if __name__ == "__main__":
    print("Starting...")
    sr = SpeechRecognition()
    sr.train_model()