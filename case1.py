# Import necessary libraries
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
import warnings

# Load the dataset, drop missing values, and set the first row as column names
df = pd.read_excel("/kaggle/input/fimmgdataset/FIMMG dataset_ok 1.xlsx", sheet_name=0, header=0)
df.columns = df.iloc[0]  # First row becomes the header
df = df[1:]  # Remove the first row as it is now used as the header
df.dropna(inplace=True)  # Remove missing data

# Split the dataframe into features (X) and labels (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = y.astype('int')  # Ensure labels are integers

# Split the dataset into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert training and test sets back into pandas DataFrames for easier manipulation
df_train = pd.DataFrame(X_train, columns=df.columns[:-1])
df_train['label'] = y_train

df_test = pd.DataFrame(data=X_test, columns=df.columns[:-1])
df_test['label'] = y_test

df_train.reset_index(drop=True, inplace=True)  # Reset indices for consistency
df_test.reset_index(drop=True, inplace=True)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)

# Define hyperparameters
embedding_dims = 2
batch_size = 32
epochs = 50

# Custom Dataset class for handling triplet data (anchor, positive, negative)
class CustomDataset(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.is_train = train
        self.transform = transform

        if self.is_train:
            self.data = df.iloc[:,:-1].values.astype(np.float32)
            self.labels = df.iloc[:,-1].values.astype(np.int64)
            self.index = df.index.values
        else:
            self.data = df.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_data = self.data[idx]

        if self.is_train:
            anchor_label = self.labels[idx]

            # Select a positive sample with the same label
            positive_list = self.index[(self.index != idx) & (self.labels[self.index] == anchor_label)]

            # Handle case where no positive samples are available
            if len(positive_list) == 0:
                return self.__getitem__(np.random.randint(len(self)))  # Retry with random sample

            positive_idx = random.choice(positive_list)
            positive_data = self.data[positive_idx]

            # Select a negative sample with a different label
            negative_list = self.index[(self.index != idx) & (self.labels[self.index] != anchor_label)]
            if len(negative_list) == 0:
                return self.__getitem__(np.random.randint(len(self)))

            negative_idx = random.choice(negative_list)
            negative_data = self.data[negative_idx]

            # Convert to tensor and apply any transformations
            anchor_data = torch.from_numpy(anchor_data)
            positive_data = torch.from_numpy(positive_data)
            negative_data = torch.from_numpy(negative_data)

            if self.transform:
                anchor_data = self.transform(anchor_data)
                positive_data = self.transform(positive_data)
                negative_data = self.transform(negative_data)

            return anchor_data, positive_data, negative_data, anchor_label
        else:
            anchor_data = torch.from_numpy(anchor_data)

            if self.transform:
                anchor_data = self.transform(anchor_data)

            return anchor_data

# Create DataLoader for training and test sets
train_ds = CustomDataset(df_train, train=True, transform=transforms.Compose([]))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

test_ds = CustomDataset(df_test, train=False, transform=transforms.Compose([]))
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# Define Triplet Loss for the network
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)  # Positive distance
        distance_negative = self.calc_euclidean(anchor, negative)  # Negative distance
        losses = torch.relu(distance_positive - distance_negative + self.margin)  # Compute loss

        return losses.mean()

# Define the neural network structure for learning embeddings
class Network(nn.Module):
    def __init__(self, input_dim, emb_dim=128):
        super(Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# Get input dimension and create the model
input_dim = train_ds.data.shape[1]  # Get the input dimension from the dataset
embedding_dims = 128  # Set embedding dimension

model = Network(input_dim, embedding_dims)  # Create the network
criterion = TripletLoss(margin=1.0)  # Set Triplet Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer

# Training loop
model.train()
model.to(device)
for epoch in tqdm(range(epochs), desc="Epochs"):
    running_loss = []
    for step, (anchor_data, positive_data, negative_data, anchor_label) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        anchor_data = anchor_data.to(device)
        positive_data = positive_data.to(device)
        negative_data = negative_data.to(device)

        optimizer.zero_grad()  # Zero the gradients
        anchor_out = model(anchor_data)  # Forward pass for anchor
        positive_out = model(positive_data)  # Forward pass for positive
        negative_out = model(negative_data)  # Forward pass for negative

        loss = criterion(anchor_out, positive_out, negative_out)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss.append(loss.cpu().detach().numpy())  # Track loss
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))

# Generate embeddings for evaluation
results = []
labels = []

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for anchor_data, _, _, label in tqdm(train_loader, desc="Generating Embeddings"):
        anchor_out = model(anchor_data.to(device)).cpu().numpy()  # Generate embedding
        results.append(anchor_out)  # Store embedding
        labels.append(label.numpy())  # Store label

# Convert embeddings and labels to numpy arrays
results = np.concatenate(results)
labels = np.concatenate(labels)

# Define a scoring function for model evaluation
def my_micro_macro(y_true, y_pred):
    macro = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return macro, precision, recall

# Modify AdaBoost to use the generated embeddings
def myAdaBoost(X_all2, Y2):
    rng = np.random.default_rng(seed=1)
    in_folds = 5  # Inner folds for cross-validation
    out_folds = 10  # Outer folds
    idx_ext = StratifiedKFold(n_splits=out_folds, shuffle=True, random_state=1)

    n_estimators_list = [50, 100, 200]

    results = {
        'f1': [],
        'pre': [],
        'recall': [],
        'acc': [],
        'auc': [],
        'Conf': []
    }

    # Perform cross-validation and evaluate the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_ext_idx, test_ext_idx in idx_ext.split(X_all2, Y2):
            train_ext, test_ext = X_all2[train_ext_idx], X_all2[test_ext_idx]
            labtrain_ext, labtest_ext = Y2[train_ext_idx], Y2[test_ext_idx]

            idx_int = StratifiedKFold(n_splits=in_folds, shuffle=True, random_state=1)

            best_pre = -np.inf
            best_n_estimators = None

            for train_int_idx, test_int_idx in idx_int.split(train_ext, labtrain_ext):
                train_int, test_int = train_ext[train_int_idx], train_ext[test_int_idx]
                labtrain_int, labtest_int = labtrain_ext[train_int_idx], labtrain_ext[test_int_idx]

                # Find the best AdaBoost model with cross-validation
                for n_estimators in n_estimators_list:
                    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=1)
                    ada.fit(train_int, labtrain_int)
                    y_pred = ada.predict(test_int)
                    macro, precision, recall = my_micro_macro(labtest_int, y_pred)
                    auc = roc_auc_score(labtest_int, ada.decision_function(test_int))
                    if precision > best_pre:
                        best_pre = precision
                        best_n_estimators = n_estimators

            # Train final AdaBoost model with best parameters
            ada = AdaBoostClassifier(n_estimators=best_n_estimators, random_state=1)
            ada.fit(train_ext, labtrain_ext)
            y_pred_ext = ada.predict(test_ext)
            k = classification_report(labtest_ext, y_pred_ext, output_dict=True)
            conf_matrix = confusion_matrix(labtest_ext, y_pred_ext)
            auc = roc_auc_score(labtest_ext, ada.decision_function(test_ext))

            # Store evaluation results
            results['f1'].append(k["weighted avg"]['f1-score'])
            results['pre'].append(k["weighted avg"]['precision'])
            results['recall'].append(k["weighted avg"]['recall'])
            results['acc'].append(k["accuracy"])
            results['auc'].append(auc)
            results['Conf'].append(conf_matrix)

    return results

# Use generated embeddings and labels for evaluation
k = myAdaBoost(results, labels)

# Print confusion matrices and evaluation metrics
conf = k["Conf"]
print(conf)
print(k)

# Calculate and print mean metrics
f1 = np.mean(k["f1"])
pre = np.mean(k["pre"])
recall = np.mean(k["recall"])
acc = np.mean(k["acc"])
auc = np.mean(k["auc"])
print('F1-SCORE IS ' + str(f1))
print('PRECISION IS ' + str(pre))
print('RECALL IS ' + str(recall))
print('ACCURACY IS ' + str(acc))
print('AUC SCORE IS ' + str(auc))
