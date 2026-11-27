package helpers

import (
	"fmt"
	"regexp"
)

var moduleNameRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$`)

// PythonModuleCmd builds the command that runs a Python module,
// e.g. ("pkg.dataset", "pull") -> ["python", "-m", "pkg.dataset", "pull"].
func PythonModuleCmd(module string, args ...string) ([]string, error) {
	if module == "" {
		return nil, fmt.Errorf("module name must not be empty")
	}
	if !moduleNameRe.MatchString(module) {
		return nil, fmt.Errorf("invalid Python module name %q", module)
	}
	cmd := []string{"python", "-m", module}
	return append(cmd, args...), nil
}
