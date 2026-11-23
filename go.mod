module github.com/benallfree/compress13k

go 1.27

require github.com/benallfree/nollama v0.0.0

require github.com/ebitengine/purego v0.11.3 // indirect

// nollama's integer models live on its intcodec branch until it is merged and tagged.
replace github.com/benallfree/nollama => /Volumes/Code/repos/nollama/.claude/worktrees/intcodec
