import structlog


def get_logger(name: str = __name__):
    """Return a configured structlog logger."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )
    return structlog.get_logger(name)
