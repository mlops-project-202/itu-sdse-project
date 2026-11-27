package helpers

const (
	DatasetModule  = "scripts.dataset"
	FeaturesModule = "scripts.features"
	TrainModule    = "scripts.modeling.train"
)

// Stages lists the pipeline modules in execution order.
var Stages = []string{DatasetModule, FeaturesModule, TrainModule}