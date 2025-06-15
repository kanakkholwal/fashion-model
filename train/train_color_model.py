
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.color_feature_extractor import ColorAwareFeatureExtractor
from models.color_loss import ColorTheoryLoss
from pipeline.dataset_loader import create_data_loaders


def train_color_model(
    data_dir: str,
    model_save_path: str = "saved_models/best_color_model.pt",
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorAwareFeatureExtractor().to(device)
    criterion = ColorTheoryLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loaders = create_data_loaders(data_dir, batch_size)
    writer = SummaryWriter()

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            targets = {
                "skin_tone_labels": batch["skin_tone"].to(device),
                "undertone_labels": batch["undertone"].to(device),
            }
            outputs = model(images)
            losses = criterion(outputs, targets)

            loss = losses["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loaders["train"])
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device)
                targets = {
                    "skin_tone_labels": batch["skin_tone"].to(device),
                    "undertone_labels": batch["undertone"].to(device),
                }
                outputs = model(images)
                losses = criterion(outputs, targets)
                val_loss += losses["total_loss"].item()

        avg_val_loss = val_loss / len(loaders["val"])
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss
            }, model_save_path)

    writer.close()
    print("Training complete. Best model saved to:", model_save_path)
    writer.close()
    print("Training complete. Best model saved to:", model_save_path)

    