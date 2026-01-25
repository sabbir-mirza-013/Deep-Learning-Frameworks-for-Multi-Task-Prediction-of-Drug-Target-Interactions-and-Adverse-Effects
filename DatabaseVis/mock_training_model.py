from training_monitor import TrainingMonitor
import time
import random

def start_training():
    # Initialize using our helper class
    monitor = TrainingMonitor(model_name='DeepLearn_Alpha_01')
    monitor.connect()
    
    headers = ["Batch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
    
    print("Starting training simulation...")
    
    best_val_acc = 0.0
    
    try:
        for epoch in range(1, 101):
            time.sleep(2) # Simulate training time
            
            # Generate dummy data for this epoch
            val_acc = random.uniform(0.7, 0.99)
            data = [
                [f"{epoch}001", f"{random.uniform(0.1, 0.9):.4f}", f"{random.uniform(0.7, 0.99):.4f}", f"{random.uniform(0.1, 0.9):.4f}", f"{val_acc:.4f}"],
                [f"{epoch}002", f"{random.uniform(0.1, 0.8):.4f}", f"{random.uniform(0.75, 0.99):.4f}", f"{random.uniform(0.1, 0.8):.4f}", f"{random.uniform(0.7, 0.99):.4f}"]
            ]
            
            # Check if this is the best epoch
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                print(f"⭐ New best validation accuracy: {val_acc:.4f}")
            
            monitor.log_epoch(epoch, headers, data, best=is_best)
            print(f"Sent update for Epoch {epoch}")
            
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        monitor.finish()

if __name__ == '__main__':
    start_training()
