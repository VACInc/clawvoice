"""Keep tests offline: any outbound connection to a non-loopback host fails."""
import socket

_real_connect = socket.socket.connect
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and host not in _LOOPBACK and not host.startswith("/"):
        raise OSError(f"network disabled in tests: {host}")
    return _real_connect(self, address)


def install():
    socket.socket.connect = _guarded_connect
