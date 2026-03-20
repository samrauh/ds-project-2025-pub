import os

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(
        self, patience=25, min_delta=0.0001, path="checkpoint.pt", printing=True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path  # Path to save the best model
        self.printing = printing

        # remove model if path already exists
        if os.path.exists(path):
            os.remove(path)
            if self.printing:
                print(f"Removing existing model at: {path}")

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if val_loss < self.best_loss:
                self.save_checkpoint(val_loss, model)
                self.best_loss = val_loss
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        tmp_path = self.path + ".tmp"
        torch.save(model.state_dict(), tmp_path)
        try:
            os.replace(tmp_path, self.path)
        except PermissionError:
            # Windows: destination may be locked; remove it first, then rename
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            os.rename(tmp_path, self.path)
        if self.printing:
            print(
                f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ..."
            )


def train_model(
    epochs,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    early_stopper,
    device,
    printing=True,
):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for data in train_loader:
            # Ensure data is a Data object, not a list

            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1), data.y.float())
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * data.num_graphs

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                # Ensure data is a Data object, not a list
                if isinstance(data, list):
                    data = data[0]

                data = data.to(device)
                output = model(data)
                loss = criterion(output.view(-1), data.y.float())
                running_val_loss += loss.item() * data.num_graphs

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if printing:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
            )

        early_stopper(epoch_val_loss, model)

        if early_stopper.early_stop:
            if printing:
                print("Early stopping triggered")
            break

    return train_losses, val_losses


def train_model_mlp(
    epochs,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    early_stopper,
    device,
    printing=True,
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_train_losses = []

        for scalar, embedding, target, year in train_loader:
            scalar, embedding, target = (
                scalar.to(device),
                embedding.to(device),
                target.to(device),
            )
            optimizer.zero_grad()
            output = model(scalar, embedding)
            loss = criterion(output, target.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

        mean_train_loss = np.mean(batch_train_losses)
        train_losses.append(mean_train_loss)

        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for scalar, embedding, target, year in val_loader:
                scalar, embedding, target = (
                    scalar.to(device),
                    embedding.to(device),
                    target.to(device),
                )
                output = model(scalar, embedding)
                val_loss = criterion(output, target.view(-1, 1).float())
                batch_val_losses.append(val_loss.item())

        mean_val_loss = np.mean(batch_val_losses)
        val_losses.append(mean_val_loss)

        # Now, print and early stopping based on mean absolute percentage error
        if printing:
            print(
                f"Epoch: {epoch + 1}, Training Loss: {mean_train_loss:.4f}, Validation Loss: {mean_val_loss:.4f}"
            )

        early_stopper(mean_val_loss, model)
        if early_stopper.early_stop and printing:
            print("Early stopping triggered.")
            print(f"Best Validation loss: {early_stopper.best_loss:.4f}")
            break
        elif early_stopper.early_stop:
            break

    # Load the best model weights
    try:
        model.load_state_dict(torch.load(early_stopper.path))
    except Exception as e:
        if printing:
            print(f"Could not load best model state: {e}")

    return train_losses, val_losses


def train_model_gnn_temporal(
    epochs,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    early_stopper,
    device,
    printing=True,
):
    """
    Training loop for the temporal GNN that uses the ``node_available`` mask
    (produced by create_temp_graph_data_dict) to exclude zero-padded nodes from
    the loss computation.

    Each batch item is expected to be a PyG Data object with:
      - data.node_available : BoolTensor [num_nodes] — True for active IPC codes
      - data.y              : FloatTensor [num_nodes] — targets (0 for inactive nodes)
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        running_train_loss = 0.0
        num_train_nodes = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data).view(-1)  # [num_nodes_in_batch]
            mask = data.node_available  # BoolTensor [num_nodes_in_batch]

            loss = criterion(output[mask], data.y[mask].float())
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * mask.sum().item()
            num_train_nodes += mask.sum().item()

        epoch_train_loss = running_train_loss / max(num_train_nodes, 1)
        train_losses.append(epoch_train_loss)

        # ---- Validation ----
        model.eval()
        running_val_loss = 0.0
        num_val_nodes = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                output = model(data).view(-1)
                mask = data.node_available

                loss = criterion(output[mask], data.y[mask].float())
                running_val_loss += loss.item() * mask.sum().item()
                num_val_nodes += mask.sum().item()

        epoch_val_loss = running_val_loss / max(num_val_nodes, 1)
        val_losses.append(epoch_val_loss)

        if printing:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Val Loss: {epoch_val_loss:.4f}"
            )

        early_stopper(epoch_val_loss, model)
        if early_stopper.early_stop:
            if printing:
                print("Early stopping triggered.")
                print(f"Best Validation Loss: {early_stopper.best_loss:.4f}")
            break

    # Load best model weights
    try:
        model.load_state_dict(torch.load(early_stopper.path))
    except Exception as e:
        if printing:
            print(f"Could not load best model state: {e}")

    return train_losses, val_losses


def train_temporal_gnn(
    model,
    optimizer,
    criterion,
    data_dict,
    train_years,
    val_years,
    early_stopper,
    device,
    epochs=100,
    printing=True,
):
    """
    Stateful training loop for TemporalGATGRU.

    - GRU hidden state resets to zeros at the start of every epoch and is
      propagated **in chronological order** across all training years.
    - Truncated BPTT (1-step): gradients are zeroed and stepped after each
      year to keep memory bounded while still training the GRU.
    - Loss + evaluation use only active nodes (node_available == True).
    - Validation runs in order over val_years with a fresh hidden state.

    Args:
        model        : TemporalGATGRU instance
        optimizer    : torch optimizer
        criterion    : loss function (e.g. nn.MSELoss())
        data_dict    : dict mapping year -> PyG Data object
        train_years  : list of years to use for training
        val_years    : list of years to use for validation
        early_stopper: EarlyStopping instance
        device       : torch.device
        epochs       : maximum number of training epochs
        printing     : whether to print per-epoch progress

    Returns:
        train_losses : list of per-epoch training losses
        val_losses   : list of per-epoch validation losses
    """
    num_nodes = next(iter(data_dict.values())).x.shape[0]
    sorted_train = sorted(y for y in train_years if y in data_dict)
    sorted_val = sorted(y for y in val_years if y in data_dict)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        h = model.init_hidden(num_nodes, device)
        total_loss, total_nodes = 0.0, 0

        for year in sorted_train:
            data = data_dict[year].to(device)
            mask = data.node_available  # BoolTensor [N]

            optimizer.zero_grad()
            # Detach hidden state (TBPTT-1); supports both tensor and list-of-tensors
            h_in = [hi.detach() for hi in h] if isinstance(h, list) else h.detach()
            out, h = model(data, h_in)
            loss = criterion(out.view(-1)[mask], data.y[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_nodes += mask.sum().item()

        epoch_train_loss = total_loss / max(total_nodes, 1)
        train_losses.append(epoch_train_loss)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        h_val = model.init_hidden(num_nodes, device)
        val_loss_total, val_nodes = 0.0, 0

        with torch.no_grad():
            for year in sorted_val:
                data = data_dict[year].to(device)
                mask = data.node_available
                out, h_val = model(data, h_val)
                loss = criterion(out.view(-1)[mask], data.y[mask])
                val_loss_total += loss.item() * mask.sum().item()
                val_nodes += mask.sum().item()

        epoch_val_loss = val_loss_total / max(val_nodes, 1)
        val_losses.append(epoch_val_loss)

        if printing:
            print(
                f"Epoch {epoch + 1:>4}/{epochs}  "
                f"Train: {epoch_train_loss:.5f}  Val: {epoch_val_loss:.5f}"
            )

        early_stopper(epoch_val_loss, model)
        if early_stopper.early_stop:
            if printing:
                print(f"Early stopping — best val loss: {early_stopper.best_loss:.5f}")
            break

    # Restore best checkpoint
    try:
        model.load_state_dict(torch.load(early_stopper.path, map_location=device))
        if printing:
            print("Best model weights restored.")
    except Exception as e:
        if printing:
            print(f"Could not restore best model: {e}")

    return train_losses, val_losses
