import { z } from "zod";
import type { Journal } from "./accounting.js";
import { accountingVersion } from "./accounting.js";
import type { AuditEvent } from "./types.js";
import type { ActorContext } from "./actor.js";
import type { MileageDeclaration } from "./mileage.js";

export const ledgerExportSchema = z
  .object({
    label: z.string().trim().min(3).max(120),
    journals: z
      .array(
        z
          .object({ id: z.string().uuid(), version: accountingVersion })
          .strict(),
      )
      .min(1)
      .max(100),
  })
  .strict()
  .refine(
    (v) => new Set(v.journals.map((j) => j.id)).size === v.journals.length,
    "Não repita lançamentos no lote.",
  );
export type LedgerExportFields = z.infer<typeof ledgerExportSchema>;
export type ExportJournal = Pick<
  Journal,
  | "id"
  | "date"
  | "description"
  | "reference"
  | "lines"
  | "accounts"
  | "allocations"
  | "provenance"
  | "postedBy"
  | "postedAt"
  | "reversalOf"
  | "opening"
  | "settlement"
  | "payableSettlement"
  | "payableRefund"
  | "payableCreditNote"
  | "advanceSettlement"
  | "cardRefund"
  | "batchSettlement"
> & {
  dependencies?: string[];
  source: {
    kind: string;
    id: string;
    token: string;
    evidenceMode: string;
    type: string;
    mileage?: MileageDeclaration;
    delegatedCorrections?: import("./expense-corrections.js").ExpenseCorrectionRecord[];
  } | null;
};
export interface LedgerExport {
  id: string;
  label: string;
  version: number;
  state: "review" | "queued" | "void";
  journalVersions: { id: string; version: number }[];
  payload: {
    schema: "despesas-journals-v1";
    currency: "BRL";
    mode: "simulation";
    journals: ExportJournal[];
  };
  sha256: string;
  createdAt: string;
  reviewedAt?: string;
  createdBy: string;
  reviewedBy?: string;
  history: AuditEvent[];
  delivery: LedgerDelivery | null;
}
export type DeliveryOutcome = "success" | "transient-error" | "permanent-error";
export interface DeliveryJobView {
  workerId: string;
  fence: number;
  actor: ActorContext;
  state: "claimed" | "expired" | "finished" | "interrupted";
  startedAt: string;
  expiresAt: string;
  finishedAt?: string;
  outcome?: DeliveryOutcome;
  failure?: "timeout" | "adapter-error";
}
export interface LedgerDelivery {
  job?: DeliveryJobView;
  id: string;
  exportId: string;
  idempotencyKey: string;
  state: "pending" | "retry" | "dead-letter" | "simulated-ack";
  version: number;
  attempts: number;
  cycleAttempts: number;
  replayCount: number;
  provider: "local-mock-erp-v1";
  providerAcknowledged: false;
  simulatedReceipt: string | null;
  nextAttemptAt: string | null;
  history: AuditEvent[];
}
