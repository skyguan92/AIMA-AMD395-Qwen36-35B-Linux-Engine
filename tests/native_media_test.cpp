// SPDX-License-Identifier: Apache-2.0

#include "aima/native_media.h"

#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_media_test: " << message << '\n';
    std::exit(1);
  }
}

template <typename Function>
void require_invalid(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  require(false, message);
}

template <typename Function>
void require_invalid_redacted(Function&& function, std::string_view secret,
                              const char* message) {
  try {
    function();
  } catch (const std::invalid_argument& error) {
    require(std::string_view(error.what()).find(secret) == std::string_view::npos,
            message);
    return;
  }
  require(false, message);
}

std::filesystem::path temporary_directory() {
  std::string pattern = "/tmp/aima-native-media-test.XXXXXX";
  char* value = ::mkdtemp(pattern.data());
  if (value == nullptr) throw std::runtime_error("mkdtemp failed");
  return std::filesystem::path(value);
}

void write_bytes(const std::filesystem::path& path,
                 std::initializer_list<unsigned char> bytes) {
  std::ofstream output(path, std::ios::binary);
  for (unsigned char byte : bytes) output.put(static_cast<char>(byte));
  if (!output) throw std::runtime_error("fixture write failed");
}

void send_all(int socket, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t count = ::send(socket, bytes + offset, size - offset, 0);
    if (count <= 0) return;
    offset += static_cast<std::size_t>(count);
  }
}

void serve_http_connection(int client, std::uint16_t port) {
  std::string request;
  char buffer[1024];
  while (request.size() < 8192 && request.find("\r\n\r\n") == std::string::npos) {
    const ssize_t count = ::recv(client, buffer, sizeof(buffer), 0);
    if (count <= 0) return;
    request.append(buffer, static_cast<std::size_t>(count));
  }
  const std::size_t first_space = request.find(' ');
  const std::size_t second_space =
      first_space == std::string::npos
          ? std::string::npos
          : request.find(' ', first_space + 1);
  const std::string path =
      first_space == std::string::npos || second_space == std::string::npos
          ? std::string()
          : request.substr(first_space + 1, second_space - first_space - 1);
  if (path == "/redirect") {
    constexpr char response[] =
        "HTTP/1.1 302 Found\r\nLocation: /image\r\nContent-Length: 0\r\n"
        "Connection: close\r\n\r\n";
    send_all(client, response, sizeof(response) - 1);
    return;
  }
  if (path == "/redirect-body") {
    constexpr char response[] =
        "HTTP/1.1 302 Found\r\nLocation: /image\r\nContent-Length: 8\r\n"
        "Connection: close\r\n\r\nredirect";
    send_all(client, response, sizeof(response) - 1);
    return;
  }
  if (path == "/to-localhost") {
    const std::string response =
        "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://localhost:" +
        std::to_string(port) +
        "/image\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    send_all(client, response.data(), response.size());
    return;
  }
  if (path == "/cross-host") {
    constexpr char response[] =
        "HTTP/1.1 302 Found\r\nLocation: http://not-example.com/image.png\r\n"
        "Content-Length: 0\r\nConnection: close\r\n\r\n";
    send_all(client, response, sizeof(response) - 1);
    return;
  }
  if (path == "/loop") {
    constexpr char response[] =
        "HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\n"
        "Connection: close\r\n\r\n";
    send_all(client, response, sizeof(response) - 1);
    return;
  }
  if (path == "/slow") (void)::usleep(100000);
  constexpr unsigned char png[] = {0x89, 'P',  'N',  'G', 0x0d, 0x0a,
                                   0x1a, 0x0a, 0x00, 0x00, 0x00, 0x00};
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: " +
      std::to_string(sizeof(png)) + "\r\nConnection: close\r\n\r\n";
  send_all(client, response.data(), response.size());
  send_all(client, png, sizeof(png));
}

class TestHttpServer {
 public:
  TestHttpServer() {
    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) throw std::runtime_error("test socket failed");
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (::bind(listener, reinterpret_cast<sockaddr*>(&address),
               sizeof(address)) != 0 ||
        ::listen(listener, 16) != 0) {
      (void)::close(listener);
      throw std::runtime_error("test HTTP bind failed");
    }
    socklen_t address_size = sizeof(address);
    if (::getsockname(listener, reinterpret_cast<sockaddr*>(&address),
                      &address_size) != 0) {
      (void)::close(listener);
      throw std::runtime_error("test HTTP port lookup failed");
    }
    port_ = ntohs(address.sin_port);
    child_ = ::fork();
    if (child_ < 0) {
      (void)::close(listener);
      throw std::runtime_error("test HTTP fork failed");
    }
    if (child_ == 0) {
      (void)::signal(SIGPIPE, SIG_IGN);
      while (true) {
        const int client = ::accept(listener, nullptr, nullptr);
        if (client < 0) _exit(0);
        serve_http_connection(client, port_);
        (void)::close(client);
      }
    }
    (void)::close(listener);
  }

  TestHttpServer(const TestHttpServer&) = delete;
  TestHttpServer& operator=(const TestHttpServer&) = delete;
  ~TestHttpServer() {
    if (child_ > 0) {
      (void)::kill(child_, SIGTERM);
      (void)::waitpid(child_, nullptr, 0);
    }
  }

  std::uint16_t port() const { return port_; }

 private:
  pid_t child_ = -1;
  std::uint16_t port_ = 0;
};

void require_openssl(bool condition, const char* operation) {
  if (!condition) throw std::runtime_error(operation);
}

void add_certificate_extension(X509* certificate, int identifier,
                               const char* value) {
  X509_EXTENSION* extension =
      X509V3_EXT_conf_nid(nullptr, nullptr, identifier,
                          const_cast<char*>(value));
  require_openssl(extension != nullptr, "test certificate extension failed");
  const int added = X509_add_ext(certificate, extension, -1);
  X509_EXTENSION_free(extension);
  require_openssl(added == 1, "test certificate extension add failed");
}

class TestHttpsServer {
 public:
  TestHttpsServer(const std::filesystem::path& certificate_path,
                  std::uint16_t downgrade_port)
      : certificate_path_(certificate_path),
        downgrade_port_(downgrade_port) {
    EVP_PKEY_CTX* key_context = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    require_openssl(key_context != nullptr, "test TLS key context failed");
    const bool key_ready = EVP_PKEY_keygen_init(key_context) == 1 &&
                           EVP_PKEY_CTX_set_rsa_keygen_bits(key_context, 2048) ==
                               1 &&
                           EVP_PKEY_keygen(key_context, &key_) == 1;
    EVP_PKEY_CTX_free(key_context);
    require_openssl(key_ready, "test TLS key generation failed");

    certificate_ = X509_new();
    require_openssl(certificate_ != nullptr, "test certificate allocation failed");
    require_openssl(X509_set_version(certificate_, 2) == 1 &&
                        ASN1_INTEGER_set(X509_get_serialNumber(certificate_), 1) ==
                            1 &&
                        X509_gmtime_adj(X509_get_notBefore(certificate_), -60) !=
                            nullptr &&
                        X509_gmtime_adj(X509_get_notAfter(certificate_), 3600) !=
                            nullptr &&
                        X509_set_pubkey(certificate_, key_) == 1,
                    "test certificate fields failed");
    X509_NAME* name = X509_get_subject_name(certificate_);
    require_openssl(
        name != nullptr &&
            X509_NAME_add_entry_by_txt(
                name, "CN", MBSTRING_ASC,
                reinterpret_cast<const unsigned char*>("127.0.0.1"), -1, -1,
                0) == 1 &&
            X509_set_issuer_name(certificate_, name) == 1,
        "test certificate identity failed");
    add_certificate_extension(certificate_, NID_basic_constraints,
                              "critical,CA:TRUE");
    add_certificate_extension(certificate_, NID_key_usage,
                              "critical,keyCertSign,digitalSignature,keyEncipherment");
    add_certificate_extension(certificate_, NID_subject_alt_name,
                              "IP:127.0.0.1");
    require_openssl(X509_sign(certificate_, key_, EVP_sha256()) > 0,
                    "test certificate signing failed");
    FILE* certificate_file = std::fopen(certificate_path_.c_str(), "wb");
    require_openssl(certificate_file != nullptr,
                    "test certificate file open failed");
    const int written = PEM_write_X509(certificate_file, certificate_);
    const int closed = std::fclose(certificate_file);
    require_openssl(written == 1 && closed == 0,
                    "test certificate write failed");

    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    require_openssl(listener >= 0, "test TLS socket failed");
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (::bind(listener, reinterpret_cast<sockaddr*>(&address),
               sizeof(address)) != 0 ||
        ::listen(listener, 16) != 0) {
      (void)::close(listener);
      throw std::runtime_error("test HTTPS bind failed");
    }
    socklen_t address_size = sizeof(address);
    if (::getsockname(listener, reinterpret_cast<sockaddr*>(&address),
                      &address_size) != 0) {
      (void)::close(listener);
      throw std::runtime_error("test HTTPS port lookup failed");
    }
    port_ = ntohs(address.sin_port);
    child_ = ::fork();
    if (child_ < 0) {
      (void)::close(listener);
      throw std::runtime_error("test HTTPS fork failed");
    }
    if (child_ == 0) run_child(listener);
    (void)::close(listener);
  }

  TestHttpsServer(const TestHttpsServer&) = delete;
  TestHttpsServer& operator=(const TestHttpsServer&) = delete;
  ~TestHttpsServer() {
    if (child_ > 0) {
      (void)::kill(child_, SIGTERM);
      (void)::waitpid(child_, nullptr, 0);
    }
    X509_free(certificate_);
    EVP_PKEY_free(key_);
  }

  std::uint16_t port() const { return port_; }
  const std::filesystem::path& certificate_path() const {
    return certificate_path_;
  }

 private:
  [[noreturn]] void run_child(int listener) const {
    (void)::signal(SIGPIPE, SIG_IGN);
    SSL_CTX* context = SSL_CTX_new(TLS_server_method());
    if (context == nullptr ||
        SSL_CTX_set_min_proto_version(context, TLS1_2_VERSION) != 1 ||
        SSL_CTX_use_certificate(context, certificate_) != 1 ||
        SSL_CTX_use_PrivateKey(context, key_) != 1) {
      _exit(1);
    }
    while (true) {
      const int client = ::accept(listener, nullptr, nullptr);
      if (client < 0) _exit(0);
      SSL* session = SSL_new(context);
      if (session != nullptr && SSL_set_fd(session, client) == 1 &&
          SSL_accept(session) == 1) {
        serve_tls_connection(session);
        (void)SSL_shutdown(session);
      }
      SSL_free(session);
      (void)::close(client);
    }
  }

  void serve_tls_connection(SSL* session) const {
    std::string request;
    char buffer[1024];
    while (request.size() < 8192 &&
           request.find("\r\n\r\n") == std::string::npos) {
      const int count = SSL_read(session, buffer, sizeof(buffer));
      if (count <= 0) return;
      request.append(buffer, static_cast<std::size_t>(count));
    }
    const bool downgrade = request.find("GET /downgrade ") == 0;
    std::string response;
    if (downgrade) {
      response =
          "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:" +
          std::to_string(downgrade_port_) +
          "/image\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    } else {
      constexpr unsigned char png[] = {0x89, 'P',  'N',  'G', 0x0d, 0x0a,
                                       0x1a, 0x0a, 0x00, 0x00, 0x00, 0x00};
      response =
          "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: " +
          std::to_string(sizeof(png)) + "\r\nConnection: close\r\n\r\n";
      response.append(reinterpret_cast<const char*>(png), sizeof(png));
    }
    std::size_t offset = 0;
    while (offset < response.size()) {
      const int count = SSL_write(session, response.data() + offset,
                                  static_cast<int>(response.size() - offset));
      if (count <= 0) return;
      offset += static_cast<std::size_t>(count);
    }
  }

  EVP_PKEY* key_ = nullptr;
  X509* certificate_ = nullptr;
  std::filesystem::path certificate_path_;
  std::uint16_t downgrade_port_ = 0;
  pid_t child_ = -1;
  std::uint16_t port_ = 0;
};

}  // namespace

int main() {
  const std::filesystem::path root = temporary_directory();
  const std::filesystem::path outside = temporary_directory();
  const auto cleanup = [&]() {
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::remove_all(outside, error);
  };
  try {
    const std::filesystem::path png = root / "space image.png";
    write_bytes(png, {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a,
                      0x00, 0x00, 0x00, 0x00});
    const std::filesystem::path outside_png = outside / "outside.png";
    write_bytes(outside_png, {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a});
    std::filesystem::create_symlink(outside_png, root / "escape.png");

    aima::NativeMediaPolicy policy;
    policy.allowed_local_roots = {root};
    policy.allowed_media_domains = {"EXAMPLE.COM", "127.0.0.1", "localhost"};

    aima::NativeMediaPart local;
    local.kind = aima::NativeMediaKind::kImage;
    local.source = "file://" + root.string() + "/space%20image.png";
    const auto local_payload = aima::load_native_media_payload(local, policy);
    require(local_payload.transport == aima::NativeMediaTransport::kLocalFile &&
                local_payload.mime_type == "image/png" &&
                local_payload.bytes.size() == 12 &&
                local_payload.content_sha256.size() == 64,
            "local PNG load contract changed");

    aima::NativeMediaPart data = local;
    data.source = "data:image/png;base64,iVBORw0KGgoAAAAA";
    const auto data_payload = aima::load_native_media_payload(data, policy);
    require(data_payload.transport == aima::NativeMediaTransport::kDataUri &&
                data_payload.bytes == local_payload.bytes &&
                data_payload.content_sha256 == local_payload.content_sha256,
            "data/local content identity changed");

    aima::NativeMediaPart escaped = local;
    escaped.source = "file://" + (root / "escape.png").string();
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(escaped, policy); },
        "symlink escape left the local allowlist");

    aima::NativeMediaPart lexical_escape = local;
    lexical_escape.source =
        "file://" + (root / ".." / outside.filename() / "outside.png").string();
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(lexical_escape, policy); },
        "lexical parent traversal left the local allowlist");

    const std::filesystem::path swapped_path = root / "swapped.png";
    write_bytes(swapped_path,
                {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a});
    aima::NativeMediaPart swapped = local;
    swapped.source = "file://" + swapped_path.string();
    require(aima::validate_native_media_source(swapped, policy) ==
                aima::NativeMediaTransport::kLocalFile,
            "regular local file failed validation");
    std::filesystem::remove(swapped_path);
    std::filesystem::create_symlink(outside_png, swapped_path);
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(swapped, policy); },
        "path replacement after validation escaped the local allowlist");

    aima::NativeMediaPart mismatch = data;
    mismatch.kind = aima::NativeMediaKind::kVideo;
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(mismatch, policy); },
        "image bytes were admitted as video");

    aima::NativeMediaPolicy tiny = policy;
    tiny.maximum_image_bytes = 8;
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(data, tiny); },
        "data URI exceeded its byte limit");

    aima::NativeMediaPart noncanonical = data;
    noncanonical.source = "data:image/png;base64,iVBORw0KGgp=";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(noncanonical, policy); },
        "non-canonical base64 padding bits were admitted");

    aima::NativeMediaPart remote = local;
    remote.source = "https://example.com/path/image.png";
    require(aima::validate_native_media_source(remote, policy) ==
                aima::NativeMediaTransport::kHttpsUrl,
            "allowlisted HTTPS source was rejected");
    remote.source = "http://127.0.0.1:8080/image.png";
    require(aima::validate_native_media_source(remote, policy) ==
                aima::NativeMediaTransport::kHttpUrl,
            "allowlisted HTTP port was rejected");
    remote.source = "https://user@example.com/image.png";
    require_invalid_redacted(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "user@example.com",
        "remote URL credentials were admitted");
    remote.source = "https://not-example.com/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "unlisted remote domain was admitted");
    remote.source = "https://example.com:bogus/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "nonnumeric remote URL port was admitted");
    remote.source = "https://example.com:/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "empty remote URL port was admitted");
    remote.source = "https://example.com\\@not-example.com/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "backslash authority ambiguity was admitted");

    TestHttpServer server;
    const std::string loopback =
        "http://127.0.0.1:" + std::to_string(server.port());
    (void)::setenv("http_proxy", "http://127.0.0.1:1", 1);
    (void)::setenv("no_proxy", "", 1);
    remote.source = loopback + "/image";
    const auto remote_payload = aima::load_native_media_payload(remote, policy);
    require(remote_payload.transport == aima::NativeMediaTransport::kHttpUrl &&
                remote_payload.mime_type == "image/png" &&
                remote_payload.bytes == local_payload.bytes &&
                remote_payload.content_sha256 == local_payload.content_sha256,
            "bounded HTTP media load changed content identity");

    remote.source = loopback + "/redirect";
    require(aima::load_native_media_payload(remote, policy).bytes ==
                local_payload.bytes,
            "relative allowlisted HTTP redirect failed");
    aima::NativeMediaPolicy aggregate_limit = policy;
    aggregate_limit.maximum_image_bytes = local_payload.bytes.size();
    remote.source = loopback + "/redirect-body";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(remote, aggregate_limit); },
        "redirect bodies bypassed the aggregate download limit");
    remote.source = loopback + "/to-localhost";
    require(aima::load_native_media_payload(remote, policy).bytes ==
                local_payload.bytes,
            "allowlisted localhost redirect failed");

    remote.source = loopback + "/cross-host";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(remote, policy); },
        "redirect to an unlisted domain was followed");

    aima::NativeMediaPolicy one_redirect = policy;
    one_redirect.maximum_remote_redirects = 1;
    remote.source = loopback + "/loop";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(remote, one_redirect); },
        "remote redirect limit was ignored");

    aima::NativeMediaPolicy remote_tiny = policy;
    remote_tiny.maximum_image_bytes = 8;
    remote.source = loopback + "/image";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(remote, remote_tiny); },
        "remote payload exceeded its byte limit");

    aima::NativeMediaPolicy remote_timeout = policy;
    remote_timeout.maximum_image_fetch_milliseconds = 20;
    remote_timeout.maximum_remote_connect_milliseconds = 20;
    remote.source = loopback + "/slow";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(remote, remote_timeout); },
        "remote fetch timeout was ignored");

    aima::NativeMediaPolicy numeric_alias = policy;
    numeric_alias.allowed_media_domains.push_back("2130706433");
    remote.source = "http://2130706433:" + std::to_string(server.port()) +
                    "/image";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(remote, numeric_alias); },
        "noncanonical numeric host bypassed loopback restrictions");

    TestHttpsServer https_server(root / "test-ca.pem", server.port());
    const std::string secure_loopback =
        "https://127.0.0.1:" + std::to_string(https_server.port());
    remote.source = secure_loopback + "/image";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(remote, policy); },
        "untrusted HTTPS certificate was admitted");
    aima::NativeMediaPolicy trusted_tls = policy;
    trusted_tls.remote_tls_ca_bundle = https_server.certificate_path();
    const auto https_payload =
        aima::load_native_media_payload(remote, trusted_tls);
    require(https_payload.transport == aima::NativeMediaTransport::kHttpsUrl &&
                https_payload.bytes == local_payload.bytes &&
                https_payload.content_sha256 == local_payload.content_sha256,
            "verified HTTPS media load changed content identity");
    remote.source = secure_loopback + "/downgrade";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(remote, trusted_tls); },
        "HTTPS redirect downgraded to plaintext HTTP");

    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  std::cout << "native_media_test: PASS\n";
  return 0;
}
