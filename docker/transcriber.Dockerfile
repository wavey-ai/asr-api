FROM rust:1.87-bookworm AS build

WORKDIR /app

COPY Cargo.toml Cargo.lock /app/
COPY src /app/src

RUN cargo build --release --locked

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/target/release/transcriber /usr/local/bin/transcriber

ENV RUST_LOG=transcriber=info,web_service=info

EXPOSE 8443

ENTRYPOINT ["transcriber"]
