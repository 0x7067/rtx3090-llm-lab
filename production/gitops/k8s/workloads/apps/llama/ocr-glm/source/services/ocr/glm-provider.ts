import path from "node:path";
import { z } from "zod";
import { ocrCpuReaderIdentity } from "../../shared/ocr.js";
import {
  OcrWorkerProvider,
  type OcrWorkerProviderOptions,
} from "./provider.js";

const nativeBase = z.string().refine((value) => {
  try {
    const url = new URL(value);
    return (
      url.protocol === "http:" &&
      url.hostname === "127.0.0.1" &&
      url.port === "8001" &&
      url.pathname === "/v1" &&
      !url.username &&
      !url.password &&
      !url.search &&
      !url.hash
    );
  } catch {
    return false;
  }
}, "GLM inference requires its pinned private loopback endpoint.");

export const glmWorkerConfigSchema = z
  .object({
    configVersion: z.literal(1),
    layoutDevice: z.literal("cpu"),
    layoutModelDir: z.string().refine(path.isAbsolute),
    nativeBaseUrl: nativeBase,
    servedModelName: z.literal("glm-ocr"),
    pdfScale: z.literal(2),
    cpuReader: z.object({
      id: z.literal(ocrCpuReaderIdentity.id),
      fingerprint: z.literal(ocrCpuReaderIdentity.fingerprint),
      modelDir: z.string().refine(path.isAbsolute),
    }).strict().optional(),
  })
  .strict();

/** A completed Python process alone does not establish that GPU work drained. */
export async function checkGlmInferenceIdle(
  base: string,
  fetcher: typeof fetch = fetch,
): Promise<boolean> {
  if (!nativeBase.safeParse(base).success) return false;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2000);
  const read = async (endpoint: string) => {
    const response = await fetcher(new URL(endpoint, base), {
      redirect: "error",
      signal: controller.signal,
    });
    if (!response.ok || !response.body)
      throw Error("Native inference status unavailable.");
    const reader = response.body.getReader();
    let bytes = 0;
    const parts: Uint8Array[] = [];
    try {
      for (;;) {
        const next = await reader.read();
        if (next.done) break;
        bytes += next.value.length;
        if (bytes > 1_048_576) {
          controller.abort();
          await reader.cancel();
          throw Error("Native inference status exceeded its bound.");
        }
        parts.push(next.value);
      }
    } finally {
      reader.releaseLock();
    }
    return Buffer.concat(parts).toString("utf8");
  };
  try {
    const metrics = await read("/metrics");
    for (const name of ["requests_processing", "requests_deferred"]) {
      const matches = [
        ...metrics.matchAll(
          new RegExp(
            `^llamacpp:${name}(?:\\{[^\\n]*\\})?\\s+([0-9]+(?:\\.[0-9]+)?)\\s*$`,
            "gm",
          ),
        ),
      ];
      if (matches.length !== 1 || Number(matches[0][1]) !== 0) return false;
    }
    const slots: unknown = JSON.parse(await read("/slots"));
    return (
      Array.isArray(slots) &&
      slots.length === 1 &&
      slots[0]?.id === 0 &&
      slots[0]?.is_processing === false
    );
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** Native GLM owns its drain check; an auxiliary CPU reader must be explicitly declared. */
export class GlmOcrProvider extends OcrWorkerProvider {
  constructor(
    options: OcrWorkerProviderOptions,
    dependencies: { fetch?: typeof fetch } = {},
  ) {
    const config = glmWorkerConfigSchema.parse(options.workerConfig);
    super(
      { ...options, pythonLibraryPath: "/opt/py312/lib" },
      {
        inferenceIdle: () =>
          checkGlmInferenceIdle(config.nativeBaseUrl, dependencies.fetch),
      },
    );
    if (
      this.pipeline.ocrModel.id !== "ggml-org/GLM-OCR-GGUF" ||
      this.pipeline.inferenceEngine.name !== "llama-cpp"
    )
      throw Error(
        "GLM provider requires its pinned GGUF model and llama.cpp engine.",
      );
    if (Boolean(config.cpuReader) !== Boolean(this.pipeline.auxiliaryReader))
      throw Error("The auxiliary reader must be declared in both pipeline and worker configuration.");
  }
}
