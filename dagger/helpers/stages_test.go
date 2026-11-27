package helpers

import (
	"strings"
	"testing"
)

func TestStagesAreValidModules(t *testing.T) {
	for _, m := range Stages {
		if _, err := PythonModuleCmd(m); err != nil {
			t.Errorf("stage %q is not a valid module name: %v", m, err)
		}
	}
}

func TestStagesBelongToScriptsPackage(t *testing.T) {
	for _, m := range Stages {
		if !strings.HasPrefix(m, "scripts.") {
			t.Errorf("stage %q is outside the scripts package", m)
		}
	}
}

func TestStagesOrder(t *testing.T) {
	want := []string{"scripts.dataset", "scripts.features", "scripts.modeling.train"}
	if len(Stages) != len(want) {
		t.Fatalf("expected %d stages, got %d", len(want), len(Stages))
	}
	for i := range want {
		if Stages[i] != want[i] {
			t.Errorf("stage %d: expected %q, got %q", i, want[i], Stages[i])
		}
	}
}