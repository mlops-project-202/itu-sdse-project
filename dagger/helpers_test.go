package main

import (
	"reflect"
	"testing"
)

func TestPythonModuleCmd(t *testing.T) {
	tests := []struct {
		name    string
		module  string
		args	[]string
		want    []string
		wantErr bool
	}{
		{"simple module", "dataset", nil, []string{"python", "-m", "dataset"}, false},
		{"dotted module", "scripts.modeling.train", nil ,[]string{"python", "-m", "scripts.modeling.train"}, false},
		{"valid module with argument", "scripts.dataset", []string{"pull"}, []string{"python", "-m", "scripts.dataset", "pull"}, false},
		{"empty name", "", nil, nil, true},
		{"contains space", "data set", nil, nil, true},
		{"shell metacharacters", "dataset; rm -rf /", nil, nil, true},
		{"starts with digit", "1dataset", nil, nil, true},
		{"trailing dot", "dataset.", nil, nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := pythonModuleCmd(tt.module)
			if (err != nil) != tt.wantErr {
				t.Fatalf("pythonModuleCmd(%q) error = %v, wantErr %v", tt.module, err, tt.wantErr)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("pythonModuleCmd(%q) = %v, want %v", tt.module, got, tt.want)
			}
		})
	}
}
