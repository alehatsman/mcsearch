.PHONY: build install test vet fmt clean
.DEFAULT_GOAL := build

BINARY := mcsearch
INSTALL_PATH ?= /usr/local/bin

build:
	@echo "Building $(BINARY)..."
	@go mod download
	@go build -trimpath -ldflags "-s -w" -o $(BINARY) ./cmd/mcsearch
	@echo "✓ Built ./$(BINARY)"

install: build
	@echo "Installing $(BINARY) to $(INSTALL_PATH)..."
	@chmod +x $(BINARY)
	@sudo cp $(BINARY) $(INSTALL_PATH)/$(BINARY)
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
