import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from utils_humor import smooth_labels


def train(model, optimizer, criterion, train_dataloader, device, smoothing=0.1):
    model.train()
    epoch_loss = 0
    all_labels = []
    all_predictions = []

    for batch_idx, (features, labels) in enumerate(tqdm(train_dataloader)):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        model_output = model(features)
        model_output = torch.sigmoid(model_output).squeeze()  # Squeeze to match label shape
        smoothed_labels = smooth_labels(labels, smoothing).to(torch.float32).squeeze()
        # loss = criterion(model_output, labels.to(torch.float32).squeeze())  # Squeeze labels if needed
        loss = criterion(model_output, smoothed_labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        all_predictions.extend(model_output.detach().cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = epoch_loss / len(train_dataloader)
    auc_score = roc_auc_score(all_labels, all_predictions)
    binary_predictions = [1 if pred >= 0.1 else 0 for pred in all_predictions]
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions)
    recall = recall_score(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)

    return avg_loss, auc_score, accuracy, precision, recall, f1



def validation(model, criterion, dataloader, device):
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(dataloader)):
            features, labels = features.to(device), labels.to(device)
            model_output = model(features)
            model_output = torch.sigmoid(model_output).squeeze()  # Squeeze to match label shape
            loss = criterion(model_output, labels.to(torch.float32).squeeze())  # Squeeze labels if needed
            epoch_loss += loss.item()
            all_predictions.extend(model_output.detach().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = epoch_loss / len(dataloader)  # Correct dataloader length
    auc_score = roc_auc_score(all_labels, all_predictions)
    binary_predictions = [1 if pred >= 0.1 else 0 for pred in all_predictions]
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions)
    recall = recall_score(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)

    return avg_loss, auc_score, accuracy, precision, recall, f1



def test(model, dataloader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(dataloader)):
            features = features.to(device)
            model_output = model(features)
            model_output = torch.sigmoid(model_output).squeeze()
            all_predictions.extend(model_output.detach().cpu().numpy().flatten())

    binary_predictions = [1 if pred >= 0.1 else 0 for pred in all_predictions]

    return all_predictions, binary_predictions



def test_devel(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(dataloader)):
            features, labels = features.to(device), labels.to(device)
            model_output = model(features)
            model_output = torch.sigmoid(model_output).squeeze()
            all_predictions.extend(model_output.detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())

    binary_predictions = [1 if pred >= 0.1 else 0 for pred in all_predictions]

    return all_predictions, binary_predictions, all_labels



