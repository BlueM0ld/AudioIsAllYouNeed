import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import torch.utils
from torch.utils.data import DataLoader
import torchaudio.transforms
from tqdm import tqdm
from modelAudio.model import Whisper
# from transformers import GPT2Tokenizer
import numpy as np
import wandb
import torchaudio

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set HYPERPARAMS

batch_size = 16 
learning_rate = 0.0001
num_epochs = 30
seq_len = 128

# Load Models
print("Load models and tokeniser")
# Need to use embeddings etc
# From Whisper somehow



# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = Whisper(
    vocab_size= 128
    ) ### Need to do !
model.to(device)
# Optimiser/loss functions

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr = learning_rate)



# Initialize wandb
wandb.init(
    project="AudioIsAllYouNeed",
    config={
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "max_seq_len": seq_len,
    }
)

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("danavery/urbansound8K",cache_dir="./audioData")['train']

def pad_spectrogram(mel_spectrogram, target_len=200):
    """
    Pads or truncates the Mel-spectrogram to a fixed target length.
    mel_spectrogram: Tensor of shape [n_mels, n_frames]
    target_len: The desired number of frames (time steps) for the spectrogram.
    """
    if mel_spectrogram.shape[-1] < target_len:
        # If the number of frames (n_frames) is less than target_len, pad it
        mel_spectrogram = nn.functional.pad(mel_spectrogram, (0, target_len - mel_spectrogram.shape[-1]))
    else:
        # If the number of frames (n_frames) is greater than target_len, truncate it
        mel_spectrogram = mel_spectrogram[:, :target_len]
    
    return mel_spectrogram

def process_data(batch, batch_index, device="cuda"):
    processed_batch = []
    
    # Extract the audio, class, and classID fields
    audio_data = batch['audio']
    classes = batch['class']
    class_ids = batch['classID']
    
    for i, audio in enumerate(audio_data):
        # Extract waveform and sampling rate
        waveform = torch.tensor(audio['array']).float()  # Convert to tensor and ensure dtype is float32
        sample_rate = audio['sampling_rate']
        
        # Convert to Mel spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
        log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        # Padding
        log_mel_spectrogram = pad_spectrogram(log_mel_spectrogram, target_len=200)

        # Add batch dimension and permute to [batch_size, channels, seq_len]
        mel_spectrogram = log_mel_spectrogram.unsqueeze(0).permute(0, 2, 1)  # Change to [1, n_frames, n_mels]

        # Append the processed data
        processed_batch.append({
            "mel_spectrogram": mel_spectrogram.to(device),  # Move to device if needed
            "class": classes[i],  # Class label
            "class_id": class_ids[i]  # Numeric class ID
        })
    
    return processed_batch
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            return {
                "mel_spectrogram": item["mel_spectrogram"],  # Update to return mel spectrogram
                "class_id": item["class_id"],
                "class": item["class"]
            }
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        

# Store processed items in memory
processed_data = []
for i in tqdm(range(0, len(dataset), batch_size), desc="Processing dataset"):
    batch_end = min(i + batch_size, len(dataset))
    batch = dataset[i:batch_end]
    processed_data.extend(process_data(batch, i))
    
    
# Split data for train/val/test
indices = np.random.permutation(len(processed_data))
train_size = int(0.8 * len(processed_data))
val_size = int(0.1 * len(processed_data))

train_data = [processed_data[i] for i in indices[:train_size]]
val_data = [processed_data[i] for i in indices[train_size:train_size+val_size]]
test_data = [processed_data[i] for i in indices[train_size+val_size:]]


# Create datasets with in-mem data
train_dataset = AudioDataset(train_data)
val_dataset = AudioDataset(val_data)
test_dataset = AudioDataset(test_data)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # Training
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimiser.zero_grad()
        
        audio_spectrogram = batch["mel_spectrogram"].to(device)
        classID = batch["class_id"].to(device)  # Move classID to the same device as the model
        
        outputs = model(audio_spectrogram)
        loss = criterion(outputs, classID)
        
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item()
    
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == classID).float().mean()
        
        # Log metrics
        wandb.log({
            "batch_loss": loss.item(),
            "batch_accuracy": accuracy.item(),
        })
        
    avg_train_loss = total_loss / len(train_loader)

# Save the model
torch.save(model.state_dict(), "audio_model.pth")  # Save the model state

# Evaluation
model.eval()  # Set the model to evaluation mode
total_eval_loss = 0
correct_predictions = 0

with torch.no_grad():  # Disable gradient calculation
    for batch in tqdm(val_loader, desc="Evaluating"):
        audio_spectrogram = batch["mel_spectrogram"].to(device)
        classID = batch["class_id"].to(device)
        
        outputs = model(audio_spectrogram)
        loss = criterion(outputs, classID)
        total_eval_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == classID).sum().item()

avg_eval_loss = total_eval_loss / len(val_loader)
eval_accuracy = correct_predictions / len(val_dataset)

# Log evaluation metrics
wandb.log({
    "eval_loss": avg_eval_loss,
    "eval_accuracy": eval_accuracy,
})
