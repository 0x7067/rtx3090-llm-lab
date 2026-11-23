"""Only fixed, redacted errors may cross the subprocess protocol."""

MESSAGES = {
    "invalid_document": "The original document cannot be decoded safely.",
    "unsupported_media": "This document format is unsupported.",
    "encrypted_document": "Encrypted documents are unsupported.",
    "resource_limit": "The document exceeds the processing resource limit.",
    "provider_unavailable": "The OCR inference service is unavailable.",
    "provider_failure": "The OCR inference service failed.",
    "invalid_provider_output": "The OCR inference service returned invalid output.",
    "timeout": "Document processing exceeded the configured time limit.",
}

# Operator-only classifications. Never derive these values from a document,
# dependency exception, path or model response text.
RESOURCE_LIMITS = frozenset({
    "page_dimensions", "region_dimensions", "pixel_budget", "page_count", "region_count",
    "token_limit", "block_text", "total_text", "memory",
    "warning_count", "input_bytes", "input_protocol", "output_protocol",
})


class WorkerError(Exception):
    def __init__(self, code: str, retryable: bool = False, *, resource_limit: str | None = None):
        if code not in MESSAGES:
            code, retryable = "provider_failure", False
        self.code = code
        self.retryable = retryable
        self.resource_limit = (
            resource_limit
            if code == "resource_limit" and isinstance(resource_limit, str)
            and resource_limit in RESOURCE_LIMITS
            else None
        )
        super().__init__(MESSAGES[code])

    def envelope(self):
        output = {
            "protocolVersion": 1,
            "error": {
                "code": self.code,
                "message": MESSAGES[self.code],
                "retryable": self.retryable,
            },
        }
        if self.resource_limit is not None:
            output["error"]["resourceLimit"] = self.resource_limit
        return output
