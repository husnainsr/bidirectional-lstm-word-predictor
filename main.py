import torch
from lstm import process_race_data, SentenceDataset, LSTMPredictor, train_model, PredictionHandler
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

def log_info(message, emoji="ℹ️"):
    print(f"\n{emoji} {message}")

def main():
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_info(f"Using device: {device}", "🖥️")

    # Load RACE dataset
    log_info("Loading RACE dataset...", "📚")
    dataset = load_dataset("race", "all")
    train_sentences = process_race_data(dataset['train'])
    val_sentences = process_race_data(dataset['validation'])

    # Load tokenizer
    log_info("Loading tokenizer...", "🔤")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Create datasets
    log_info("Creating datasets...", "⚙️")
    train_dataset = SentenceDataset(train_sentences[:10000], tokenizer)  # Limit for testing
    val_dataset = SentenceDataset(val_sentences[:1000], tokenizer)

    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model parameters
    model_params = {
        'vocab_size': vocab_size,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3
    }

    # Initialize models
    log_info("Initializing models...", "🤖")
    forward_model = LSTMPredictor(**model_params)
    backward_model = LSTMPredictor(**model_params)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    num_epochs = 50

    # Train forward model
    log_info("Training Forward Model...", "⏩")
    forward_optimizer = torch.optim.Adam(forward_model.parameters(), lr=learning_rate)
    forward_results = train_model(
        model=forward_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=forward_optimizer,
        tokenizer=tokenizer,
        num_epochs=num_epochs,
        device=device,
        model_type='forward'
    )

    # Train backward model
    log_info("Training Backward Model...", "⏪")
    backward_optimizer = torch.optim.Adam(backward_model.parameters(), lr=learning_rate)
    backward_results = train_model(
        model=backward_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=backward_optimizer,
        tokenizer=tokenizer,
        num_epochs=num_epochs,
        device=device,
        model_type='backward'
    )

    # Save models
    log_info("Saving models...", "💾")
    torch.save(forward_model.state_dict(), 'forward_model.pth')
    torch.save(backward_model.state_dict(), 'backward_model.pth')

    # Print final results
    log_info("Final Results:", "🎯")
    print("\nForward Model:")
    print(f"Training Accuracy: {forward_results['final_train_acc']:.2f}%")

    print("\nBackward Model:")
    print(f"Training Accuracy: {backward_results['final_train_acc']:.2f}%")


    torch.save(forward_model.state_dict(), 'forward_model.pth')
    torch.save(backward_model.state_dict(), 'backward_model.pth')


    log_info("Loading trained models...", "📂")
    forward_model.load_state_dict(torch.load('forward_model.pth'))
    backward_model.load_state_dict(torch.load('backward_model.pth'))

    forward_model.eval()
    backward_model.eval()

    # Initialize prediction handler
    predictor = PredictionHandler(forward_model, backward_model, tokenizer, device)

    # Test sentences
    test_sentences = [
        ("The quick brown fox jumps over the lazy dog", 3),  # predict 'brown'
        ("The students studied hard for the final exam", 5),  # predict 'hard'
        ("She opened the window to let fresh air inside", 4),  # predict 'window'
    ]

    # Make predictions
    log_info("Making predictions...", "🎯")
    for sentence, blank_position in test_sentences:
        original_word = sentence.split()[blank_position]
        result = predictor.predict_word(sentence, blank_position)

        print(f"\nOriginal sentence: {sentence}")
        print(f"Word to predict: {original_word}")
        print("\nForward model predictions:")
        for word, conf in result['forward_predictions']:
            print(f"  {word}: {conf:.4f}")

        print("\nBackward model predictions:")
        for word, conf in result['backward_predictions']:
            print(f"  {word}: {conf:.4f}")

        print(f"\nSelected word: {result['selected_word']}")
        print(f"Forward confidence: {result['forward_confidence']:.4f}")
        print(f"Backward confidence: {result['backward_confidence']:.4f}")
        print("-" * 80)


