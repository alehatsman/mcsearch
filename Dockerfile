# Build stage: compile a static `dex` binary.
#
# Tree-sitter grammars are wrapped in CGO, so we need a C toolchain.
# Alpine's musl + `-extldflags -static` gives us a single static binary
# that runs on distroless/scratch — no libc dependency at runtime.
FROM golang:1.26-alpine AS build
WORKDIR /src

RUN apk add --no-cache build-base

# Cache module downloads separately from source changes.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 GOOS=linux \
    go build -trimpath -tags 'sqlite_fts5' \
        -ldflags '-s -w -extldflags "-static"' \
        -o /out/dex ./cmd/dex

# Pre-create /cache owned by the nonroot uid so named volumes get
# correct ownership on first use. Distroless has no shell or `chown`.
RUN mkdir -p /out/cache && chown -R 65532:65532 /out/cache

# Runtime stage: distroless static, ~3 MB on top of the binary.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/dex /usr/local/bin/dex
COPY --from=build --chown=nonroot:nonroot /out/cache /cache

# Mount the project tree read-only at /work and the index cache at
# /cache. The MCP/CLI defaults to ~/.cache/dex, so override it.
ENV DEX_INDEX_DIR=/cache \
    DEX_EMBED_URL=http://host.docker.internal:8082 \
    DEX_EMBED_MODEL=Qwen/Qwen3-Embedding-4B

WORKDIR /work
USER nonroot:nonroot

# stdio MCP server is the canonical container entrypoint — `docker run -i`
# wires stdin/stdout to the host so Claude Code (or any MCP client) can
# talk to it directly. Override the command for CLI usage.
#
# Bind-mounted /cache: pass --user "$(id -u):$(id -g)" so the host owns
# the index files. Named volumes inherit the pre-created /cache
# ownership and work without --user.
#
#   # one-shot index using a named volume
#   docker run --rm -v "$PWD":/work:ro -v dex-cache:/cache \
#     -e DEX_EMBED_URL=http://host.docker.internal:8082 \
#     dex index /work
#
#   # as an MCP server over stdio
#   docker run --rm -i -v "$PWD":/work:ro -v dex-cache:/cache \
#     -e DEX_EMBED_URL=http://host.docker.internal:8082 \
#     dex
ENTRYPOINT ["/usr/local/bin/dex"]
CMD ["mcp"]
