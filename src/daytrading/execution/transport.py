"""Bound SDK HTTP calls so a missing response cannot hang the trading loop."""

from requests import Session
from requests.adapters import HTTPAdapter


class BoundedHTTPAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (3.0, 5.0)
        return super().send(request, **kwargs)


def configure_client_timeouts(client) -> None:
    # alpaca-py 0.44.0 owns this Session but does not supply request timeouts.
    session = getattr(client, "_session", None)
    if isinstance(session, Session):
        session.mount("https://", BoundedHTTPAdapter())
        session.mount("http://", BoundedHTTPAdapter())
