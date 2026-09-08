#!/bin/bash
# Builds a minimal static libcurl (linked against a static OpenSSL) so the
# ipdb binary produced in the manylinux CI container has no runtime
# dependency on that container's libcurl.so / libssl.so sonames, which don't
# exist on the GitHub Actions runner the binary is later executed on.
#
# Usage: build_static_curl.sh <install-prefix>
# Requires: openssl-static + openssl-devel already installed (for libssl.a /
# libcrypto.a and headers) and a C compiler on PATH.
set -euo pipefail

CURL_VERSION="${CURL_VERSION:-8.10.1}"
PREFIX="${1:?usage: build_static_curl.sh <install-prefix>}"

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

curl -fsSL "https://curl.se/download/curl-${CURL_VERSION}.tar.gz" -o curl.tar.gz
tar xzf curl.tar.gz
cd "curl-${CURL_VERSION}"

# Disable protocols/features this project doesn't need, to avoid pulling in
# extra static dependencies (libidn2, libpsl, nghttp2, libssh, ...) that
# aren't available as static archives in the build container.
./configure \
  --prefix="$PREFIX" \
  --disable-shared \
  --enable-static \
  --with-openssl \
  --without-zlib \
  --without-libpsl \
  --without-libidn2 \
  --without-brotli \
  --without-zstd \
  --without-nghttp2 \
  --without-librtmp \
  --without-libssh2 \
  --without-libssh \
  --without-gssapi \
  --disable-ldap \
  --disable-ldaps \
  --disable-manual

make -j"$(nproc)"
make install
