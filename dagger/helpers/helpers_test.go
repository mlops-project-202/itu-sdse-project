package helpers

import (
	"reflect"
	"testing"
)

func TestPythonModuleCmd(t *testing.T) {
	testCases := []struct {
		name        string
		value       string
		args        []string
		expectValue []string
		expectFail  bool
	}{
		{"Valid simple module", "dataset", nil, []string{"python", "-m", "dataset"}, false},
		{"Valid dotted module", "itu_sdse_project.modeling.train", nil, []string{"python", "-m", "itu_sdse_project.modeling.train"}, false},
		{"Valid module with argument", "scripts.dataset", []string{"pull"}, []string{"python", "-m", "scripts.dataset", "pull"}, false},
		{"Empty name", "", nil, nil, true},
		{"Name with space", "foo bar", nil, nil, true},
		{"Shell metacharacters", "foo; rm -rf /", nil, nil, true},
		{"Starts with digit", "1abc", nil, nil, true},
		{"Trailing dot", "foo.", nil, nil, true},

		// new cases
		{"Underscore and digits", "my_pkg2.mod_1", nil, []string{"python", "-m", "my_pkg2.mod_1"}, false},
		{"Multiple args", "scripts.dataset", []string{"pull", "--force"}, []string{"python", "-m", "scripts.dataset", "pull", "--force"}, false},
		{"Leading dot", ".foo", nil, nil, true},
		{"Double dot", "foo..bar", nil, nil, true},
		{"Hyphen in name", "foo-bar", nil, nil, true},
		{"Newline in name", "foo\nbar", nil, nil, true},
		{"Trailing newline", "foo\n", nil, nil, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := PythonModuleCmd(tc.value, tc.args...)

			if tc.expectFail {
				if err == nil {
					t.Errorf("expected an error for module: '%s', got: %v", tc.value, out)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v, for module: '%s'", err, tc.value)
			}
			if !reflect.DeepEqual(out, tc.expectValue) {
				t.Errorf("expected value: %v, got: %v, for module: '%s' with args: %v",
					tc.expectValue, out, tc.value, tc.args)
			}
		})
	}
}

func TestPythonModuleCmdReturnsIndependentSlices(t *testing.T) {
	a, _ := PythonModuleCmd("scripts.dataset", "pull")
	b, _ := PythonModuleCmd("scripts.dataset", "pull")
	a[0] = "changed"
	if b[0] != "python" {
		t.Errorf("results share memory: got %v", b)
	}
}
