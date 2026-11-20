import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import warnings

import cloudpickle
from loguru import logger
import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import typer

from scripts.config import (artifact_dir, artifact_path, batch_size, drift_baseline_stats_path,
                            early_stop_patience, experiment_name, image_size, inference_log_path,
                            learning_rate, mlflow_artifact_location, mlflow_state_dir,
                            mlflow_tracking_uri, model_name, model_results_path,
                            num_classes, num_epochs, project_root, pytorch_model_path, random_state,
                            scheduler_patience, threshold, training_run_name, transform_mean,
                            transform_std)

from scripts.features import (
    PROCESSED_DATA_DIR,
    create_dataloaders,
    define_transforms,
    load_split,
)

app=typer.Typer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MLflow setup
def setup_mlflow_tracking(experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    os.makedirs(mlflow_state_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary classification of glass vial images.

    Input: 64x64 RGB images (3 channels) - resized from native 1600x768
    Output: Single logit (positive = defective, negative = clean)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 2 max pooling layers: 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_cnn_model(train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device, random_seed=random_state):
    """Train the SimpleCNN model on glass vial images."""
    logger.info("Training CNN")
    logger.info(f"Device: {device} | Epochs: {num_epochs} | Learning rate: {learning_rate}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = SimpleCNN().to(device)

    # The dataset is imbalanced, so the positive (defective) class is
    # down-weighted by the negative/positive count ratio.
    train_labels = train_loader.dataset.labels
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    logger.info(f"Training class counts: clean(0)={num_negative}, defective(1)={num_positive}")
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    logger.info(f"Loss pos_weight: {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Halve the learning rate when validation macro-F1 plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=scheduler_patience)

    best_val_f1_macro = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs >= threshold).float().view(-1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.view(-1)).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_accuracy = train_correct / train_total
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                logits = model(images)
                loss = criterion(logits, labels)

                val_running_loss += loss.item()
                probs = torch.sigmoid(logits)
                predicted = (probs >= threshold).float().view(-1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.view(-1)).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='binary')
        epoch_val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro')

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}): "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} | "
            f"Train F1: {epoch_train_f1:.4f} | Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_accuracy:.4f} | Val F1: {epoch_val_f1:.4f} | "
            f"Val Macro F1: {epoch_val_f1_macro:.4f}"
        )

        if mlflow.active_run():
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
            mlflow.log_metric("train_f1_score", epoch_train_f1, step=epoch)
            mlflow.log_metric("val_f1_score", epoch_val_f1, step=epoch)
            mlflow.log_metric("val_f1_macro", epoch_val_f1_macro, step=epoch)

        scheduler.step(epoch_val_f1_macro)

        if epoch_val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = epoch_val_f1_macro
            # Deep-copy the weights so the "best" state does not keep changing
            best_model_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info(
                    f"Early stopping: no macro-F1 improvement for {early_stop_patience} consecutive epochs"
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logger.success(f"Best validation macro F1-score: {best_val_f1_macro:.4f}")

    return model


def tune_threshold(model, val_loader, device=device):
    """Tune the classification threshold to maximize macro F1 on the validation set.

    Macro-F1 is used because maximizing the positive-class F1 alone rewards
    predicting "defective" for every image (defective is the majority class).
    """
    # Fine grid: the model's probabilities can compress into a narrow band
    threshold_range = [round(t, 3) for t in np.arange(0.025, 0.976, 0.025)]

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)

            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)

    best_threshold = threshold
    best_f1 = 0.0

    for t in threshold_range:
        preds = (all_probs >= t).astype(int)
        f1_macro = f1_score(all_labels, preds, average='macro')
        f1_positive = f1_score(all_labels, preds, average="binary")

        logger.debug(
            f"  Threshold {t}: MacroF1={f1_macro:.4f}, F1(defective)={f1_positive:.4f}, "
            f"Acc={accuracy_score(all_labels, preds):.4f}"
        )
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = t
        
    logger.info(f"Optimal threshold: {best_threshold} (macro F1-score: {best_f1:.4f})")
        
    mlflow.log_param("optimal_threshold", best_threshold)
    mlflow.log_metric("optimal_threshold_macro_f1", best_f1)
        
    return best_threshold


def evaluate_cnn_model(model, test_loader, device=device, threshold=threshold):
    """Evaluate the model on the test set: accuracy, F1, precision, recall, ROC-AUC."""
    logger.info("Evaluating CNN")
    logger.info(f"Test batches: {len(test_loader)} | Threshold: {threshold}")

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(float).reshape(-1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probs)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)

    logger.info(
        f"Test set: Accuracy={accuracy:.4f} | F1={f1:.4f} | Precision={precision:.4f} | "
        f"Recall={recall:.4f} | ROC-AUC={roc_auc:.4f}"
    )
    logger.info(f"Confusion matrix [[TN, FP], [FN, TP]] = {conf_matrix.tolist()}")
    logger.info(f"Classification report:\n{classification_report(all_labels, all_preds)}")

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_roc_auc", roc_auc)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
    }

def compute_baseline_statistics(
    model, data_loader, device=device, feature_layer="fc1", threshold=threshold
):
    """Compute baseline statistics for drift detection from a dataset."""
    logger.info(f"Computing baseline statistics for drift detection (layer: {feature_layer})")

    model.eval()

    image_means = []
    image_stds = []
    class_counts = {0: 0, 1: 0}
    prediction_counts = {0: 0, 1: 0}
    confidences = []
    all_embeddings = []

    if not hasattr(model, feature_layer):
        raise ValueError(f"Model does not have layer '{feature_layer}'")
    feature_extractor = getattr(model, feature_layer)

    # Hook to capture embeddings
    embeddings = []

    def embedding_hook(module, inputs, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = feature_extractor.register_forward_hook(embedding_hook)

    try:
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                # Get predictions (logits -> probabilities)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= threshold).astype(int)

                # Image statistics (per channel)
                image_means.append(images.mean(dim=(0, 2, 3)).cpu().numpy())
                image_stds.append(images.std(dim=(0, 2, 3)).cpu().numpy())

                for label in labels.cpu().numpy().flatten():
                    class_counts[int(label)] += 1

                for pred in preds:
                    prediction_counts[int(pred)] += 1

                confidences.extend(probs)

                # Get embeddings from the hook
                if embeddings:
                    all_embeddings.extend(embeddings)
                    embeddings.clear()

        image_means = np.array(image_means)
        image_stds = np.array(image_stds)

        baseline_stats = {
            "image_statistics": {
                "mean_r": float(np.mean(image_means[:, 0])),
                "mean_g": float(np.mean(image_means[:, 1])),
                "mean_b": float(np.mean(image_means[:, 2])),
                "std_r": float(np.mean(image_stds[:, 0])),
                "std_g": float(np.mean(image_stds[:, 1])),
                "std_b": float(np.mean(image_stds[:, 2])),
            },
            "class_distribution": {
                "class_0_count": class_counts[0],
                "class_1_count": class_counts[1],
                "total": class_counts[0] + class_counts[1],
            },
            "prediction_distribution": {
                "pred_0_count": prediction_counts[0],
                "pred_1_count": prediction_counts[1],
                "total": prediction_counts[0] + prediction_counts[1],
            },
            "confidence_statistics": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
            },
            "embedding_statistics": {
                "mean": [],
                "std": [],
            },
        }

        if all_embeddings:
            all_embeddings_array = np.concatenate(all_embeddings, axis=0)
            baseline_stats["embedding_statistics"] = {
                "mean": np.mean(all_embeddings_array, axis=0).tolist(),
                "std": np.std(all_embeddings_array, axis=0).tolist(),
                "shape": int(all_embeddings_array.shape[1]),
            }

        logger.info(
            f"Baseline statistics computed: class 0={class_counts[0]}, 1={class_counts[1]}, "
            f"confidence mean={baseline_stats['confidence_statistics']['mean']:.4f}"
        )

        return baseline_stats

    finally:
        handle.remove()

def save_and_log_baseline(baseline_stats):
    """Write the drift baseline to disk and log it to the active MLflow run."""
    with open(drift_baseline_stats_path, "w") as f:
        json.dump(baseline_stats, f, indent=2)
    logger.info(f"Baseline statistics saved to {drift_baseline_stats_path}")

    image_stats = baseline_stats["image_statistics"]
    for key in ("mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"):
        mlflow.log_metric(f"baseline_image_{key}", image_stats[key])
    class_dist = baseline_stats["class_distribution"]
    mlflow.log_metric("baseline_class_0_count", class_dist["class_0_count"])
    mlflow.log_metric("baseline_class_1_count", class_dist["class_1_count"])
    confidence = baseline_stats["confidence_statistics"]
    mlflow.log_metric("baseline_confidence_mean", confidence["mean"])
    mlflow.log_metric("baseline_confidence_std", confidence["std"])
    mlflow.log_artifact(drift_baseline_stats_path, "drift_detection")


# Serving wrapper
class SimpleLoggingWrapper(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper that logs inference requests and returns predictions.

    NOTE: this class is pickled *by value* into the MLflow model (see
    ``log_pyfunc_model``), so the serving image does not need this package. Its
    methods therefore use ``print`` rather than the loguru logger, which is not
    picklable.
    """

    def __init__(
        self,
        model=None,
        model_path=None,
        log_path=Path(inference_log_path),
        threshold=threshold,
    ):
        self.model = model
        self.model_path = model_path
        self.log_path = Path(log_path)
        self.threshold = threshold

        # Transform for preprocessing images (same as training, no augmentation)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=transform_mean, std=transform_std),
            ]
        )

        # Device for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def get_raw_model(self):
        """Return the underlying PyTorch model for direct access."""
        return self.model

    def load_context(self, context):
        """Called by MLflow when the model is loaded."""
        print("Loading SimpleLoggingWrapper context...")
        if self.model is None:
            if self.model_path and Path(self.model_path).exists():
                checkpoint = torch.load(
                    self.model_path, map_location=self.device, weights_only=False
                )
                self.model = self._create_model_from_checkpoint(checkpoint)
            else:
                raise FileNotFoundError("Could not find model file")
        self.model.to(self.device)
        self.model.eval()
        print("SimpleLoggingWrapper context loaded successfully!")

    def _create_model_from_checkpoint(self, checkpoint):
        architecture = checkpoint.get("architecture", "SimpleCNN")
        if architecture == "SimpleCNN":
            model = SimpleCNN()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model

    def _get_timestamp_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _log_request(self, request_data):
        """Append a request log entry to the JSON log file."""
        try:
            logs = []
            if self.log_path.exists():
                with open(self.log_path, "r") as f:
                    try:
                        logs = json.load(f)
                        if not isinstance(logs, list):
                            logs = []
                    except json.JSONDecodeError:
                        logs = []

            logs.append(request_data)

            with open(self.log_path, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to log request: {e}")

    def preprocess_image(self, image_input):
        """Preprocess a single image: path, raw bytes or base64 string."""
        if isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            path = None
            try:
                candidate = Path(image_input)
                if candidate.exists():
                    path = candidate
            except OSError:
                path = None
            if path is not None:
                image = Image.open(path).convert("RGB")
            else:
                # Treat as base64-encoded image data (the portable contract
                # for containerized serving)
                try:
                    decoded_bytes = base64.b64decode(image_input)
                    image = Image.open(BytesIO(decoded_bytes)).convert("RGB")
                except Exception as e:
                    raise ValueError(
                        f"Cannot process string input '{image_input[:50]}...': "
                        f"not an existing file path and not valid base64 image data: {e}"
                    )
        elif isinstance(image_input, Path):
            if image_input.exists():
                image = Image.open(image_input).convert("RGB")
            else:
                raise FileNotFoundError(f"File not found: {image_input}")
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. Expected str, Path, or bytes."
            )

        return self.transform(image).unsqueeze(0)

    def predict(self, context, model_input):
        """Make predictions on input data and log the request."""
        if self.model is None:
            self.load_context(context)

        # Handle different input formats
        if isinstance(model_input, dict):
            if "instances" in model_input:
                image_inputs = model_input["instances"]
            elif "data" in model_input:
                image_inputs = model_input["data"]
            else:
                # Try to extract inputs from any key
                for value in model_input.values():
                    if isinstance(value, (list, str, np.ndarray, bytes, Path)):
                        single = isinstance(value, (str, Path, np.ndarray, bytes))
                        image_inputs = [value] if single else value
                        break
                else:
                    raise ValueError(f"Could not extract image inputs from input: {model_input}")
        elif isinstance(model_input, (list, tuple, np.ndarray)):
            image_inputs = model_input
        else:
            image_inputs = [model_input]

        if isinstance(image_inputs, np.ndarray):
            image_inputs = image_inputs.tolist()

        # Convert to flat list of inputs
        image_inputs_list = []
        for item in image_inputs:
            if isinstance(item, (list, tuple)):
                image_inputs_list.extend(item)
            else:
                image_inputs_list.append(item)
        image_inputs = image_inputs_list

        predictions = []
        confidences = []
        raw_outputs = []

        timestamp = self._get_timestamp_iso()

        for image_input in image_inputs:
            try:
                image_tensor = self.preprocess_image(image_input).to(self.device)

                with torch.no_grad():
                    logit = self.model(image_tensor)
                    prob = torch.sigmoid(logit).item()
                    pred = 1 if prob >= self.threshold else 0

                predictions.append(int(pred))
                confidences.append(round(float(prob), 4))
                raw_outputs.append(round(float(prob), 4))

                image_identifier = (
                    "<bytes>" if isinstance(image_input, bytes) else str(image_input)
                )
                self._log_request(
                    {
                        "timestamp": timestamp,
                        "image_path": image_identifier,
                        "prediction": int(pred),
                        "confidence": float(prob),
                        "raw_output": float(prob),
                    }
                )

            except Exception as e:
                print(f"Error processing image {image_input}: {e}")
                predictions.append(None)
                confidences.append(None)
                raw_outputs.append(None)

        return {
            "predictions": predictions,
            "confidences": confidences,
            "raw_outputs": raw_outputs,
            "num_samples": len(image_inputs),
        }


# registration
def save_model_artifacts(model, test_results, optimal_threshold):
    """Save the PyTorch checkpoint and the test results locally."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": SimpleCNN.__name__,
            "input_shape": (3, 64, 64),
            "output_shape": (1,),
            "threshold": optimal_threshold,
            "num_classes": num_classes,
        },
        pytorch_model_path,
    )
    logger.info(f"CNN model saved to {pytorch_model_path}")

    with open(model_results_path, "w") as results_file:
        json.dump(test_results, results_file, indent=2)
    logger.info(f"Model results saved to {model_results_path}")


def log_pyfunc_model(model, threshold):
    """Log the wrapped model as the pyfunc model artifact of the active run."""
    # In the notebook the classes lived in __main__, so cloudpickle embedded them in
    # the model. Registering this module keeps that behaviour when the pipeline is
    # imported instead of run as a script: the exported model stays self-contained.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    wrapper = SimpleLoggingWrapper(model=model, threshold=threshold)
    mlflow.pyfunc.log_model(artifact_path, python_model=wrapper)
    logger.info("Logged pyfunc model to the MLflow run")


def wait_until_ready(model_name, model_version, max_attempts=10):
    """Wait until the model version is READY in the registry."""
    client = MlflowClient()
    for attempt in range(max_attempts):
        try:
            details = client.get_model_version(name=model_name, version=model_version)
            status = ModelVersionStatus.from_string(details.status)
            if status == ModelVersionStatus.READY:
                logger.info(f"Model {model_name} version {model_version} is READY")
                return True
        except Exception as e:
            logger.warning(
                f"Error checking model status (attempt {attempt + 1}/{max_attempts}): {e}"
            )
        time.sleep(1)
    logger.warning(
        f"Model {model_name} version {model_version} did not become READY "
        f"after {max_attempts} attempts"
    )
    return False


def register_model(run_id, model_name=model_name):
    """Register the run's pyfunc model and move it to Staging with the `best` alias."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_until_ready(model_details.name, model_details.version)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(model_name, model_details.version, "deployed_to", "staging")
    client.set_model_version_tag(model_name, model_details.version, "task", "classification")
    client.set_registered_model_alias(
        name=model_name, alias="best", version=str(model_details.version)
    )

    logger.success(f"{model_name} version {model_details.version} registered in Staging")
    return model_details


# Entry point
@app.command()
def main(
    num_epochs: int = num_epochs,
    batch_size: int = batch_size,
    learning_rate: float = learning_rate,
):
    """Train, evaluate and register the vial image classifier.

    Expects the splits from `python -m scripts.features` in data/processed/.
    """
    warnings.filterwarnings("ignore")

    # All paths in the config are relative to the project root
    os.chdir(project_root)
    os.makedirs(artifact_dir, exist_ok=True)

    setup_mlflow_tracking(experiment_name)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Data: the splits made by the features step
    train_dataset, val_dataset, test_dataset = (
        load_split(PROCESSED_DATA_DIR / f"{name}.csv") for name in ("train", "val", "test")
    )
    train_transforms, val_transforms = define_transforms()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        train_transforms,
        val_transforms,
        batch_size=batch_size,
    )

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=training_run_name) as run:
        mlflow.log_param("model_architecture", "SimpleCNN")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("initial_threshold", threshold)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("image_size", f"{image_size[0]}x{image_size[1]}")
        mlflow.log_param("random_seed", random_state)

        cnn_model = train_cnn_model(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            random_seed=random_state,
        )
        mlflow.log_param("total_parameters", sum(p.numel() for p in cnn_model.parameters()))
        mlflow.log_param(
            "trainable_parameters",
            sum(p.numel() for p in cnn_model.parameters() if p.requires_grad),
        )

        # Tune threshold on the validation set, evaluate on the test set with it
        optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)
        test_results = evaluate_cnn_model(
            cnn_model, test_loader, device=device, threshold=optimal_threshold
        )

        # Drift baseline from the validation set
        baseline_stats = compute_baseline_statistics(
            cnn_model, val_loader, threshold=optimal_threshold
        )
        save_and_log_baseline(baseline_stats)

        # Artifacts and registration
        save_model_artifacts(cnn_model, test_results, optimal_threshold)
        mlflow.log_param("classification_threshold", optimal_threshold)
        log_pyfunc_model(cnn_model, optimal_threshold)
        model_details = register_model(run.info.run_id)

    logger.success("TRAINING JOB COMPLETE")
    logger.info(f"  Training samples: {len(train_loader.dataset)}")
    logger.info(f"  Validation samples: {len(val_loader.dataset)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")
    logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"  Test F1-score: {test_results['f1_score']:.4f}")
    logger.info(f"  Registered model: {model_name} version {model_details.version} (Staging)")


if __name__ == "__main__":
    app()