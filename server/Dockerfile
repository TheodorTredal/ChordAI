FROM golang:1.25-alpine AS builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o chordai-server .

# ── final image ───────────────────────────────────────────────────────────────
FROM alpine:3.20

WORKDIR /app
COPY --from=builder /build/chordai-server .

EXPOSE 5555

CMD ["./chordai-server"]
