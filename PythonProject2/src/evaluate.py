import matplotlib.pyplot as plt

def evaluate_model(model, test_generator):
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
    return loss, accuracy

def plot_history(history, output_path='results/training_history.png'):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(output_path)
    plt.close()