package main // Every Go file

import (
	"dagger/itu-sdse-project/internal/dagger"
) // Pulls other Dagger types (structs)

type ItuSdseProject struct {
	Source *dagger.Directory
} // The struct in question - way to group data

func New(
	// +defaultPath="/"
	source *dagger.Directory,
) *ItuSdseProject {
	return &ItuSdseProject{Source: source}
} // Constructor - runs before the other functions and establishes arguments for other functions, defaults path to root

// Base: Python container with dependencies installed.
func (m *ItuSdseProject) Base() *dagger.Container {
	return dag.Container().
		From("python:3.12-slim").
		WithMountedCache("/root/.cache/pip", dag.CacheVolume("pip")).
		WithWorkdir("/app").
		WithFile("/app/requirements.txt", m.Source.File("requirements.txt")).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithDirectory("/app", m.Source)
}

// PullData: runs the dataset stage.
func (m *ItuSdseProject) PullData() *dagger.Container {
	return m.Base().
		WithExec([]string{"python", "-m", "itu_sdse_project.dataset"})
}

// Train: runs training on top of the pulled data.
func (m *ItuSdseProject) Train() *dagger.Container {
	return m.PullData().
		WithExec([]string{"python", "-m", "itu_sdse_project.modeling.train"})
}

// Model: returns just the artifacts directory.
func (m *ItuSdseProject) Model() *dagger.Directory {
	return m.Train().Directory("/app/artifacts")
}
