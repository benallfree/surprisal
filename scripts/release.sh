#!/bin/sh
# Builds the release archives for one version into dist/: scripts/release.sh 0.0.1-alpha.1
#   dist/surprisal_<v>_<os>_<arch>.tar.gz (.zip on Windows), each with the binary, README.md and LICENSE.md
#   dist/checksums.txt (SHA-256)
#   dist/surprisal.rb, the Homebrew formula for benallfree/homebrew-tap (Formula/surprisal.rb)
# Pure Go (CGO_ENABLED=0; Metal and CUDA load at run time), -trimpath so no local paths end up in the binaries.
set -e
V=${1:?usage: scripts/release.sh <version, e.g. 0.0.1-alpha.1>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
rm -rf dist && mkdir -p dist
for t in darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64; do
	os=${t%/*} arch=${t#*/} ext=
	[ "$os" = windows ] && ext=.exe
	d=surprisal_${V}_${os}_${arch}
	echo "$(date +%T) build $t"
	mkdir -p "dist/$d"
	GOOS=$os GOARCH=$arch CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=$V" -o "dist/$d/surprisal$ext" ./cmd/surprisal
	cp README.md LICENSE.md "dist/$d/"
	if [ "$os" = windows ]; then
		(cd "dist/$d" && zip -q -X "../$d.zip" ./*)
	else
		(cd dist && COPYFILE_DISABLE=1 tar -czf "$d.tar.gz" "$d")
	fi
	rm -rf "dist/$d"
done
(cd dist && shasum -a 256 ./*.tar.gz ./*.zip | sed 's| \./| |' > checksums.txt)
sum() { awk -v f="surprisal_${V}_$1.tar.gz" '$2 == f { print $1 }' dist/checksums.txt; }
URL=https://github.com/benallfree/surprisal/releases/download/v$V
cat > dist/surprisal.rb <<EOF
class Surprisal < Formula
  desc "Lossless compressor that codes with a language model, bit-identical on any hardware"
  homepage "https://github.com/benallfree/surprisal"
  version "$V"
  license "MIT"

  on_macos do
    on_arm do
      url "$URL/surprisal_${V}_darwin_arm64.tar.gz"
      sha256 "$(sum darwin_arm64)"
    end
    on_intel do
      url "$URL/surprisal_${V}_darwin_amd64.tar.gz"
      sha256 "$(sum darwin_amd64)"
    end
  end

  on_linux do
    on_arm do
      url "$URL/surprisal_${V}_linux_arm64.tar.gz"
      sha256 "$(sum linux_arm64)"
    end
    on_intel do
      url "$URL/surprisal_${V}_linux_amd64.tar.gz"
      sha256 "$(sum linux_amd64)"
    end
  end

  def install
    bin.install "surprisal"
  end

  def caveats
    <<~EOS
      The first use of a level downloads its model into ~/.surprisal (once):
        surprisal -1 ...     # levels 1-3 (-1 is the default): 0.5 GB
        surprisal -9 ...     # levels 7-9: 7.9 GB
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/surprisal --version")
  end
end
EOF
echo "$(date +%T) done:"
ls -la dist
