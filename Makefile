.PHONY: build install test vet fmt clean
.DEFAULT_GOAL := build

BINARY := mcsearch

# Default to the per-user XDG bin dir. Two reasons:
#  1. It's on most users' PATH ahead of /usr/local/bin, so plain
#     `make install` is actually visible to the next `mcsearch` call.
#  2. It's user-writable, so no sudo prompt for the common path.
# For a system-wide install: `sudo make install INSTALL_PATH=/usr/local/bin`.
INSTALL_PATH ?= $(HOME)/.local/bin
INSTALL_TMP  := $(INSTALL_PATH)/$(BINARY).new

build:
	@echo "Building $(BINARY)..."
	@go mod download
	@go build -trimpath -ldflags "-s -w" -o $(BINARY) ./cmd/mcsearch
	@echo "✓ Built ./$(BINARY)"

# Install via cp-then-rename. A direct cp over a running binary fails
# with ETXTBSY (the MCP server or `mcsearch watch` may be holding the
# inode open); rename(2) swaps the directory entry atomically while the
# running process keeps its old inode, and the next invocation picks up
# the new file.
install: build
	@echo "Installing $(BINARY) to $(INSTALL_PATH)..."
	@chmod +x $(BINARY)
	@mkdir -p $(INSTALL_PATH)
	@cp $(BINARY) $(INSTALL_TMP)
	@mv -f $(INSTALL_TMP) $(INSTALL_PATH)/$(BINARY)
	@echo "✓ Installed $(INSTALL_PATH)/$(BINARY)"

test:
	@go test ./...

vet:
	@go vet ./...

fmt:
	@gofmt -w .

clean:
	@rm -f $(BINARY)
	@echo "✓ Cleaned"
