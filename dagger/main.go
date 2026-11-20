package main // Every Go file

import (
	"dagger/itu-sdse-project/internal/dagger"
) // Pulls other Dagger types (structs)

type ItuSdseProject struct {
	Source *dagger.Directory
} // The struct in question - way to group data

func New(
	// Project root. Local outputs are left out so every run starts clean and
	// downloads/trains for real. .git stays in: DVC needs it to fetch the
	// imported metadata CSVs.
	// +defaultPath="/"
	// +ignore=["data/images/raw", "data/test/*.jpg", "data/processed", "artifacts", "mlflow-state", "models/model", "model_pyfunc", ".dvc/cache", ".dvc/tmp", "**/__pycache__", ".ruff_cache", ".venv", "dagger"]
	source *dagger.Directory,
) *ItuSdseProject {
	return &ItuSdseProject{Source: source}
} // Constructor - runs before the other functions and establishes arguments for other functions, defaults path to root

// Base: Python container with the project and its dependencies installed.
func (m *ItuSdseProject) Base() *dagger.Container {
	return dag.Container().
		// pyproject.toml requires Python 3.13
		From("python:3.13-slim").
		WithMountedCache("/root/.cache/pip", dag.CacheVolume("pip")).
		// CPU-only PyTorch: the default Linux wheel pulls in ~2 GB of CUDA libraries
		WithEnvVariable("PIP_EXTRA_INDEX_URL", "https://download.pytorch.org/whl/cpu").
		WithWorkdir("/app").
		// The whole project must be in place before installing: requirements.txt
		// ends with `-e .`, which needs pyproject.toml and scripts/
		WithDirectory("/app", m.Source).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"})
}

// PullData: metadata via DVC, then the images from the object store.
func (m *ItuSdseProject) PullData() *dagger.Container {
	return m.Base().
		WithExec([]string{"python", "-m", "scripts.dataset", "pull"})
}

// Features: train/val/test splits and the dataset artifacts.
func (m *ItuSdseProject) Features() *dagger.Container {
	return m.PullData().
		WithExec([]string{"python", "-m", "scripts.features"})
}

// Train: trains, evaluates and registers the model on top of the features.
func (m *ItuSdseProject) Train() *dagger.Container {
	return m.Features().
		WithExec([]string{"python", "-m", "scripts.modeling.train"})
}

// Model: returns just the artifacts directory.
func (m *ItuSdseProject) Model() *dagger.Directory {
	return m.Train().Directory("/app/artifacts")
}
