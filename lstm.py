import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from transformers import AutoTokenizer
import numpy as np


def log_info(message, emoji="ℹ️"):
    print(f"\n{emoji} {message}")


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = []
        self.blanks = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Process sentences to create fill-in-the-blank examples
        for sentence in sentences:
            # Tokenize the sentence
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) < 6:  # Skip very short sentences
                continue

            # Choose a random word from the latter half to blank out
            mid_point = len(tokens) // 2
            blank_idx = random.randint(mid_point, len(tokens) - 1)
            blank_word = tokens[blank_idx]

            # Store the original sentence and blank word
            self.sentences.append({
                'full_text': sentence,
                'blank_word': blank_word,
                'blank_idx': blank_idx,
                'tokens': tokens
            })
            self.blanks.append(blank_word)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        item = self.sentences[idx]
        tokens = item['tokens']
        blank_idx = item['blank_idx']

        # Create forward sequence (before blank)
        forward_tokens = tokens[:blank_idx]
        forward_text = self.tokenizer.convert_tokens_to_string(forward_tokens)

        # Create backward sequence (after blank, reversed)
        backward_tokens = tokens[blank_idx + 1:][::-1]  # reverse the tokens
        backward_text = self.tokenizer.convert_tokens_to_string(backward_tokens)

        # Tokenize with padding
        forward_input = self.tokenizer(
            forward_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        backward_input = self.tokenizer(
            backward_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Get the target token ID
        target_tokens = self.tokenizer(item['blank_word'], return_tensors='pt')
        target_id = target_tokens['input_ids'][0, 1]  # Skip the CLS token

        return {
            'forward_ids': forward_input['input_ids'].squeeze(0),
            'backward_ids': backward_input['input_ids'].squeeze(0),
            'target_id': target_id,
            'blank_word': item['blank_word']
        }


class PredictionHandler:
    def __init__(self, forward_model, backward_model, tokenizer, device):
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.tokenizer = tokenizer
        self.device = device

    def get_prediction_confidence(self, output):
        """Calculate prediction confidence using softmax probabilities"""
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)
        return confidence.item()

    def predict_word(self, sentence, blank_position):
        """Predict missing word using both models"""
        # Split sentence at blank
        words = sentence.split()
        forward_context = ' '.join(words[:blank_position])
        backward_context = ' '.join(words[blank_position+1:])[::-1]  # Reverse for backward model

        # Tokenize contexts
        forward_tokens = self.tokenizer(forward_context,
                                      return_tensors='pt',
                                      padding=True,
                                      truncation=True).to(self.device)

        backward_tokens = self.tokenizer(backward_context,
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True).to(self.device)

        # Get predictions
        with torch.no_grad():
            forward_output = self.forward_model(forward_tokens['input_ids'])
            backward_output = self.backward_model(backward_tokens['input_ids'])

        # Get confidence scores
        forward_confidence = self.get_prediction_confidence(forward_output)
        backward_confidence = self.get_prediction_confidence(backward_output)

        # Get top 5 predictions from each model
        forward_probs = torch.nn.functional.softmax(forward_output, dim=1)
        backward_probs = torch.nn.functional.softmax(backward_output, dim=1)

        top_k = 5
        forward_values, forward_indices = torch.topk(forward_probs, top_k)
        backward_values, backward_indices = torch.topk(backward_probs, top_k)

        # Convert to words
        forward_words = [self.tokenizer.decode([idx.item()]).strip() for idx in forward_indices[0]]
        backward_words = [self.tokenizer.decode([idx.item()]).strip() for idx in backward_indices[0]]

        return {
            'forward_predictions': list(zip(forward_words, forward_values[0].tolist())),
            'backward_predictions': list(zip(backward_words, backward_values[0].tolist())),
            'forward_confidence': forward_confidence,
            'backward_confidence': backward_confidence,
            'selected_word': forward_words[0] if forward_confidence > backward_confidence else backward_words[0]
        }

class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        # Apply embedding
        embedded = self.dropout(self.embedding(input_ids))

        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1)

        # Process through LSTM
        output, (hidden, cell) = self.lstm(embedded)

        # Use only the last non-padded output for each sequence
        if attention_mask is not None:
            # Get the last non-padded position for each sequence
            lengths = attention_mask.sum(dim=1).long() - 1
            batch_size = output.size(0)

            # Gather the last relevant output for each sequence
            last_outputs = output[torch.arange(batch_size), lengths]
        else:
            last_outputs = output[:, -1]

        # Process through linear layer
        prediction = self.fc(last_outputs)
        return prediction
    
def evaluate_model(model, data_loader, criterion, tokenizer, device, model_type='forward'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch[f'{model_type}_ids'].to(device)
            target_ids = batch['target_id'].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).float().to(device)

            output = model(input_ids, attention_mask)
            loss = criterion(output, target_ids)

            predictions = output.argmax(dim=1)
            correct += (predictions == target_ids).sum().item()
            total += target_ids.size(0)
            total_loss += loss.item()

    return total_loss / len(data_loader), (correct / total * 100)


def train_model(model, train_loader, val_loader, criterion, optimizer, tokenizer,
                num_epochs=50, device='cpu', model_type='forward'):
    model.to(device)
    best_val_acc = 0
    patience = 5
    patience_counter = 0

    # Calculate total batches
    total_batches = len(train_loader)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader, 1):
            # Get the appropriate input based on model type
            input_ids = batch[f'{model_type}_ids'].to(device)
            target_ids = batch['target_id'].to(device)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (input_ids != tokenizer.pad_token_id).float().to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(input_ids, attention_mask)

            # Calculate loss and accuracy
            loss = criterion(output, target_ids)

            # Get predictions
            predictions = output.argmax(dim=1)
            train_correct += (predictions == target_ids).sum().item()
            train_total += target_ids.size(0)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # Print batch progress
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                current_loss = train_loss / batch_idx
                current_acc = (train_correct / train_total) * 100 if train_total > 0 else 0
                print(f"\r   Batch {batch_idx}/{total_batches} "
                      f"| Loss: {current_loss:.4f} "
                      f"| Acc: {current_acc:.2f}%", end="")

        print()  # New line after batch progress

        avg_train_loss = train_loss / total_batches
        train_accuracy = (train_correct / train_total) * 100 if train_total > 0 else 0

        # Validation phase
        val_loss, val_accuracy = evaluate_model(
            model, val_loader, criterion, tokenizer, device, model_type
        )

        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{model_type}_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Print epoch results
        log_info(f"Epoch {epoch+1}/{num_epochs} Results:", "📊")
        print(f"   Training Loss: {avg_train_loss:.4f}")
        print(f"   Training Accuracy: {train_accuracy:.2f}%")
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Validation Accuracy: {val_accuracy:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    return {
        'final_train_loss': avg_train_loss,
        'final_train_acc': train_accuracy,
        'final_val_loss': val_loss,
        'final_val_acc': val_accuracy,
        'best_val_acc': best_val_acc
    }


def process_race_data(dataset):
    """Extract sentences from RACE dataset articles."""
    sentences = []
    for article in dataset['article']:
        # Simple sentence splitting - you might want to use a better method
        article_sentences = article.split('.')
        sentences.extend([s.strip() + '.' for s in article_sentences if len(s.strip()) > 20])
    return sentences



 