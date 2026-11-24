//go:build !darwin && !linux

package main

import (
	"os"
	"strconv"
)

// termWidth is the terminal's width in columns: $COLUMNS when set, else 0 (unknown).
func termWidth(*os.File) int {
	n, _ := strconv.Atoi(os.Getenv("COLUMNS"))
	return n
}
