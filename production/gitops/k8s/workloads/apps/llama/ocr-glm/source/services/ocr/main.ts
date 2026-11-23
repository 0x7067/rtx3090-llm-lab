import { spawn } from "node:child_process";
import { lstat, mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { ocrPipelineSchema } from "../../shared/ocr.js";
import { buildOcrHttpServer } from "./http.js";
import { PaddleOcrProvider } from "./provider.js";
import { GlmOcrProvider } from "./glm-provider.js";
import { OcrRuntime } from "./runtime.js";

const root = path.dirname(fileURLToPath(import.meta.url));
const preparedSchema = z
  .object({
    pipeline: ocrPipelineSchema,
    workerConfig: z.record(z.string(), z.unknown()),
  })
  .strict();

async function preflight(
  python: string,
  manifest: string,
  provider: "paddle" | "glm",
  dataDir: string,
) {
  return new Promise<z.infer<typeof preparedSchema>>((resolve, reject) => {
    const child = spawn(
      python,
      [
        path.join(
          root,
          provider === "glm"
            ? "deploy/glm_preflight.py"
            : "deploy/preflight.py",
        ),
        "--manifest",
        manifest,
        "--emit-runtime",
        ...(provider === "glm"
          ? ["--verified-cache", path.join(dataDir, "glm-verified-artifacts-v1.json")]
          : []),
      ],
      {
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          PATH: process.env.PATH,
          PYTHONNOUSERSITE: "1",
          PYTHONDONTWRITEBYTECODE: "1",
          ...(provider === "glm" ? { LD_LIBRARY_PATH: "/opt/py312/lib" } : {}),
        },
      },
    );
    let count = 0;
    const chunks: Buffer[] = [];
    const timer = setTimeout(() => child.kill("SIGKILL"), 120_000);
    child.stdout.on("data", (part: Buffer) => {
      count += part.length;
      if (count > 65536) child.kill("SIGKILL");
      else chunks.push(part);
    });
    child.stderr.resume();
    child.on("error", () => {
      clearTimeout(timer);
      reject(Error("OCR startup preflight could not execute."));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0 || count > 65536) {
        reject(
          Error(
            "OCR startup preflight failed; check the pinned runtime manifest.",
          ),
        );
        return;
      }
      try {
        resolve(
          preparedSchema.parse(
            JSON.parse(Buffer.concat(chunks).toString("utf8")),
          ),
        );
      } catch {
        reject(Error("OCR startup preflight returned invalid configuration."));
      }
    });
  });
}

/** This entrypoint is for the separate OCR container, never the finance app startup. */
async function start() {
  if (process.env.OCR_ENABLED !== "real") {
    console.info("OCR is disabled. No service, worker or model was started.");
    return;
  }
  const required = (name: string) => {
    const value = process.env[name];
    if (!value) throw Error(`Missing required ${name} configuration.`);
    return value;
  };
  const absolute = (name: string) => {
    const value = required(name);
    if (!path.isAbsolute(value))
      throw Error(`${name} requires an absolute path.`);
    return value;
  };
  const manifest = absolute("OCR_RUNTIME_MANIFEST");
  const tokenPath = absolute("OCR_TOKEN_FILE");
  const dataDir = absolute("OCR_DATA_DIR");
  const namespace = required("OCR_NAMESPACE");
  if (!/^[A-Za-z0-9_-]{1,100}$/.test(namespace))
    throw Error("Invalid OCR namespace configuration.");
  const python = process.env.OCR_PYTHON ?? "/usr/local/bin/python3";
  if (!path.isAbsolute(python))
    throw Error("OCR_PYTHON requires an absolute executable path.");
  const port = Number(process.env.OCR_PORT ?? "9417");
  const host = process.env.OCR_LISTEN_HOST ?? "127.0.0.1";
  if (
    !Number.isInteger(port) ||
    port < 1024 ||
    port > 65535 ||
    !["127.0.0.1", "0.0.0.0"].includes(host)
  )
    throw Error("Invalid private OCR listener configuration.");
  const tokenStat = await lstat(tokenPath);
  if (
    !tokenStat.isFile() ||
    tokenStat.size > 4097 ||
    tokenStat.isSymbolicLink()
  )
    throw Error("Invalid OCR credential file.");
  const token = (await readFile(tokenPath, "utf8")).trim();
  if (!/^[\x21-\x7e]{24,4096}$/.test(token))
    throw Error("Invalid OCR credential.");
  const providerName = process.env.OCR_PROVIDER ?? "paddle";
  if (providerName !== "paddle" && providerName !== "glm")
    throw Error("OCR provider must be explicitly pinned to paddle or glm.");
  const prepared = await preflight(python, manifest, providerName, dataDir);
  const Provider = providerName === "glm" ? GlmOcrProvider : PaddleOcrProvider;
  const provider = new Provider({
    pipeline: prepared.pipeline,
    workerConfig: prepared.workerConfig,
    python,
    worker: path.join(
      root,
      providerName === "glm" ? "glm/worker.py" : "paddle/worker.py",
    ),
    workRoot:
      providerName === "glm" && process.env.OCR_WORK_DIR
        ? absolute("OCR_WORK_DIR")
        : path.join(dataDir, "work"),
    onResourceLimit: (diagnostic) => console.warn(JSON.stringify(diagnostic)),
  });
  await mkdir(dataDir, { mode: 0o700, recursive: true });
  await provider.prepare();
  const runtime = new OcrRuntime({
    databasePath: path.join(dataDir, "ocr.sqlite"),
    allowReal: true,
    providers: [provider],
    documentConcurrency: 1,
    queueCapacity: 16,
  });
  const app = await buildOcrHttpServer({
    runtime,
    credentials: [{ token, namespace }],
  });
  let stopping = false;
  let inFlight: Promise<unknown> | undefined;
  const worker = setInterval(() => {
    if (stopping || inFlight) return;
    inFlight = runtime
      .processNext()
      .catch(() =>
        console.error(
          "OCR worker failed; durable jobs retain their recovery state.",
        ),
      )
      .finally(() => {
        inFlight = undefined;
      });
  }, 250);
  const maintenance = setInterval(() => {
    if (!stopping)
      try {
        runtime.maintain();
      } catch {
        console.error("OCR maintenance failed.");
      }
  }, 1000);
  let cleanupInFlight: Promise<unknown> | undefined;
  const cleanup = setInterval(() => {
    if (stopping || cleanupInFlight) return;
    cleanupInFlight = provider
      .cleanup()
      .catch(() => console.error("OCR temporary-file cleanup failed."))
      .finally(() => {
        cleanupInFlight = undefined;
      });
  }, 15_000);
  const shutdown = async () => {
    if (stopping) return;
    stopping = true;
    clearInterval(worker);
    clearInterval(maintenance);
    clearInterval(cleanup);
    runtime.close();
    provider.close();
    await inFlight;
    await cleanupInFlight;
    await app.close();
  };
  for (const signal of ["SIGINT", "SIGTERM"] as const)
    process.once(signal, () => {
      void shutdown();
    });
  try {
    await app.listen({ host, port });
  } catch {
    await shutdown();
    throw Error("OCR private listener could not start.");
  }
  console.info(
    `Private OCR service listening on port ${port}; extraction requires explicit authenticated jobs.`,
  );
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  start().catch((error) => {
    // Fixed operational error text only; do not dump exception objects/env/payloads.
    console.error(
      error instanceof Error && error.message.startsWith("OCR")
        ? error.message
        : "OCR startup failed.",
    );
    process.exitCode = 1;
  });
}
