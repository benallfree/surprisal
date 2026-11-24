//go:build darwin || linux

package main

import (
	"os"
	"syscall"
	"unsafe"
)

// termWidth is the terminal's width in columns, or 0 when f isn't a terminal or the size is unknown.
func termWidth(f *os.File) int {
	var ws struct{ Row, Col, X, Y uint16 }
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), uintptr(syscall.TIOCGWINSZ), uintptr(unsafe.Pointer(&ws)))
	if errno != 0 {
		return 0
	}
	return int(ws.Col)
}
