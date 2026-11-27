package main // Every Go file

import (
	"dagger/itu-sdse-project/helpers"
	"dagger/itu-sdse-project/internal/dagger"
	context
)

const (
	pythonImage    = "python:3.13-slim"
	artifactsDir   = "/app/artifacts"
	// The servable MLflow model that the training step exports (config.export_dir)
	servingModelDir = "/app/models/model"
)

type ItuSdseProject struct {
	Source *dagger.Directory
} // The struct in question - way to group data

func New(
	// Project root. Local outputs are left out so every run starts clean and
	// downloads/trains for real. .git stays in: DVC needs it to fetch the
	// imported metadata CSVs.
	// +defaultPath="/"
	// +ignore=["data/images/raw", "data/test/*.jpg", "data/test/metadata.csv", "data/processed", "artifacts", "mlflow-state", "models/model", "model_pyfunc", ".dvc/tmp", "**/__pycache__", ".ruff_cache", ".venv", "dagger"]
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

// runStage adds one "python -m <module> <args>" step to a container.
func runStage(c *dagger.Container, module string, args ...string) (*dagger.Container, error) {
	cmd, err := helpers.PythonModuleCmd(module, args...)
	if err != nil {
		return nil, err
	}
	return c.WithExec(cmd), nil
}

// PullData: metadata via DVC, then the images from the object store.
func (m *ItuSdseProject) PullData() (*dagger.Container, error) {
	return runStage(m.Base(), helpers.DatasetModule, "pull")
}

// Features: train/val/test splits and the dataset artifacts.
func (m *ItuSdseProject) Features() (*dagger.Container, error) {
	c, err := m.PullData()
	if err != nil {
		return nil, err
	}
	return runStage(c, helpers.FeaturesModule)
}

// Train: trains, evaluates and registers the model on top of the features.
func (m *ItuSdseProject) Train() (*dagger.Container, error) {
	c, err := m.Features()
	if err != nil {
		return nil, err
	}
	return runStage(c, helpers.TrainModule)
}

// Model: the artifacts directory, plus the servable MLflow model in mlflow_model/
// (what the serving container needs).
func (m *ItuSdseProject) Model() (*dagger.Directory, error) {
	c, err := m.Train()
	if err != nil {
		return nil, err
	}
	return c.Directory(artifactsDir).
		WithDirectory("mlflow_model", c.Directory(servingModelDir)), nil
}

func (m *ItuSdseProject) PublishServe(ctx context.Context, src *dagger.Directory,
    owner string, tag string, token *dagger.Secret) (string, error) {
    return dag.Container().
        From("python:3.13-slim").
        WithWorkdir("/app").
        WithFile("requirements-serve.txt", src.File("docker/requirements-serve.txt")).
        WithExec([]string{"pip", "install", "--extra-index-url",
            "https://download.pytorch.org/whl/cpu", "-r", "requirements-serve.txt"}).
        WithDirectory("scripts", src.Directory("scripts")).
        WithDirectory("models/model", src.Directory("models/model")).
        WithFile("artifacts/drift_baseline_stats.json", src.File("artifacts/drift_baseline_stats.json")).
        WithExposedPort(5001).
        WithEntrypoint([]string{"python", "-m", "scripts.modeling.predict", "serve"}).
        WithRegistryAuth("ghcr.io", owner, token).
        Publish(ctx, "ghcr.io/"+owner+"/model-serve:"+tag)
}