import { z } from "zod";
import { accountingDate } from "./accounting.js";
import type { Expense } from "./types.js";

export const captureLimits = {
  files: 10,
  bytes: 10 * 1024 * 1024,
  xmlBytes: 1024 * 1024,
  totalBytes: 50 * 1024 * 1024,
} as const;
export const expenseCaptureSchema = z
  .object({
    batchId: z.string().uuid(),
    position: z
      .union([
        z.number().int().min(1).max(captureLimits.files),
        z.string().regex(/^(?:[1-9]|10)$/),
      ])
      .transform(Number),
    date: accountingDate,
    reason: z.string().trim().min(5).max(1500),
    requestKey: z.string().uuid(),
    // Optional for incumbent online clients. Offline commands pin the server's
    // current company/subject/member/authentication context, never credentials.
    expectedIdentity: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  })
  .strict();
export type ExpenseCaptureInput = z.infer<typeof expenseCaptureSchema>;
export const captureContextSchema = z.object({
  key: z.string().regex(/^[a-f0-9]{64}$/),
  companyId: z.string().uuid(),
  subjectId: z.string().uuid(),
  memberId: z.string().regex(/^[a-zA-Z0-9-]{1,80}$/),
  displayName: z.string().min(1).max(100),
  authentication: z.enum(["demo", "password"]),
}).strict();
export type CaptureContext = z.infer<typeof captureContextSchema>;
export const captureOriginSchema = z
  .object({
    batchId: z.string().uuid(),
    position: z.number().int().min(1).max(10),
    documentId: z.string().uuid(),
    originalName: z.string().min(1).max(200),
    initialDate: accountingDate,
    initialCategory: z.literal("Outros"),
    fieldsExtracted: z.literal(false),
    mode: z.literal("manual-original-capture"),
  })
  .strict();
export type CaptureOrigin = z.infer<typeof captureOriginSchema>;
export function expenseCaptureOrigin(
  expense: Pick<Expense, "history">,
): CaptureOrigin | null {
  const event = expense.history.find(
    (e) => e.action === "Original capturado em lote",
  );
  if (!event) return null;
  try {
    return captureOriginSchema.parse(JSON.parse(event.note).capture);
  } catch {
    return null;
  }
}
export interface CaptureExisting {
  kind: "duplicate-original" | "occupied-position";
  expenseId: string;
  documentName: string;
}
