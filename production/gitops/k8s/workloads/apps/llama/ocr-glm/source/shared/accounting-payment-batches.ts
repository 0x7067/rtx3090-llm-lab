import type { ActorContext } from "./actor.js";
import type { PaymentBatch } from "./payables.js";
import { z } from "zod";
import {
  accountingDate,
  type AccountingSource,
  type Journal,
} from "./accounting.js";

export const paymentBatchSettlementSchema = z
  .object({
    recognitionId: z.string().uuid(),
    batchId: z.string().uuid(),
    billId: z.string().uuid(),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    date: accountingDate,
    liabilityAccountId: z.string().uuid(),
    assetAccountId: z.string().uuid(),
    understood: z.literal(true),
  })
  .strict();
export type PaymentBatchSettlementFields = z.infer<
  typeof paymentBatchSettlementSchema
>;
export interface PaymentBatchSettlementBasis {
  kind: "simulated-payable-batch-item-v1";
  recognitionId: string;
  recognitionVersion: number;
  source: AccountingSource;
  batch: PaymentBatch;
  item: PaymentBatch["bills"][number];
  evidence: CompletedPaymentBatchEvidence;
  parent: Pick<
    Journal,
    | "id"
    | "date"
    | "version"
    | "lines"
    | "accounts"
    | "allocations"
    | "createdBy"
    | "postedBy"
  >;
  contributors: string[];
  amount: number;
  token: string;
  mode: "simulation";
  bankConfirmed: false;
}
export interface PaymentBatchSettlementContext {
  basis: PaymentBatchSettlementBasis;
  liabilityAccounts: { id: string; amount: number }[];
  reservations: {
    journalId: string;
    accountId: string;
    amount: number;
    kind: "batch-simulation" | "declared-payment";
  }[];
  claimedBy: string[];
}

export interface PaymentBatchStageProof {
  actor: ActorContext;
  version: number;
  at: string;
}
export interface CompletedPaymentBatchEvidence {
  kind: "completed-supplier-payment-simulation-v1";
  batchId: string;
  financialToken: string;
  preparation: PaymentBatchStageProof;
  authorization: PaymentBatchStageProof;
  simulation: PaymentBatchStageProof;
  bankConfirmed: false;
}
export interface PaymentBatchCloseEvidence {
  batch: PaymentBatch;
  item: PaymentBatch["bills"][number];
  evidence: CompletedPaymentBatchEvidence | null;
  mode: "simulation";
  bankConfirmed: false;
}
/** Read-only source inspection, not an accounting proposal or posted journal. */
export interface PaymentBatchAccountingSource {
  batch: PaymentBatch;
  evidence: CompletedPaymentBatchEvidence | null;
  issue: string | null;
  bills: {
    item: PaymentBatch["bills"][number];
    source: AccountingSource | null;
    issue: string | null;
  }[];
  basisToken: string;
  mode: "simulation";
  bankConfirmed: false;
  createsJournal: false;
}
