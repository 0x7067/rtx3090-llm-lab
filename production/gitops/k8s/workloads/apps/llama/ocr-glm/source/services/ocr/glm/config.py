from dataclasses import dataclass
import os

from ..paddle.errors import WorkerError


@dataclass(frozen=True)
class GlmWorkerConfig:
    layout_model_dir: str
    native_base_url: str
    served_model_name: str
    pdf_scale: int = 2
    cpu_model_dir: str | None = None


def parse_config(value: object) -> GlmWorkerConfig:
    required = {
            "configVersion", "layoutDevice", "layoutModelDir", "nativeBaseUrl", "servedModelName", "pdfScale"}
    if (not isinstance(value, dict) or set(value) not in [required, required | {"cpuReader"}]
            or type(value["configVersion"]) is not int or value["configVersion"] != 1
            or value["layoutDevice"] != "cpu" or type(value["pdfScale"]) is not int or value["pdfScale"] != 2
            or value["nativeBaseUrl"] != "http://127.0.0.1:8001/v1" or value["servedModelName"] != "glm-ocr"
            or not isinstance(value["layoutModelDir"], str) or not os.path.isabs(value["layoutModelDir"])
            or not os.path.isdir(value["layoutModelDir"])):
        raise WorkerError("provider_unavailable")
    cpu_dir = None
    if "cpuReader" in value:
        from .cpu_reader import READER_ID, reader_identity
        cpu = value["cpuReader"]
        if (not isinstance(cpu, dict) or set(cpu) != {"id", "fingerprint", "modelDir"}
                or cpu["id"] != READER_ID or cpu["fingerprint"] != reader_identity()["fingerprint"]
                or not isinstance(cpu["modelDir"], str) or not os.path.isabs(cpu["modelDir"])
                or not os.path.isdir(cpu["modelDir"]) or os.path.islink(cpu["modelDir"])):
            raise WorkerError("provider_unavailable")
        cpu_dir = cpu["modelDir"]
    return GlmWorkerConfig(value["layoutModelDir"],value["nativeBaseUrl"],value["servedModelName"],cpu_model_dir=cpu_dir)
