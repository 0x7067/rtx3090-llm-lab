import { z } from "zod";
import {
  accountingDate,
  type AccountingSource,
  type Journal,
} from "./accounting.js";
import { payableCreditAllocationsSchema } from "./accounting-payable-credits.js";
import type { CardMatch, CardRefund, CardTransaction } from "./cards.js";
import type { Document } from "./types.js";
import type { ActorContext } from "./actor.js";

export const cardRefundAccountingSchema = z
  .object({
    recognitionId: z.string().uuid(),
    transactionId: z.string().uuid(),
    refundId: z.string().uuid(),
    matchId: z.string().uuid(),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    date: accountingDate,
    allocations: payableCreditAllocationsSchema,
    understood: z.literal(true),
  })
  .strict();
export type CardRefundAccountingFields = z.infer<
  typeof cardRefundAccountingSchema
>;
export type CardRefundAccountingAllocation =
  CardRefundAccountingFields["allocations"][number];
export interface CardRefundAccountingOrigin {
  kind: "simulated-card-refund-v1";
  recognitionId: string;
  source: AccountingSource;
  parent: Pick<
    Journal,
    | "id"
    | "version"
    | "date"
    | "lines"
    | "accounts"
    | "allocations"
    | "createdBy"
    | "postedBy"
    | "postedAt"
  >;
  transaction: Pick<
    CardTransaction,
    | "id"
    | "cardId"
    | "owner"
    | "ownerMemberId"
    | "merchant"
    | "date"
    | "amount"
    | "origin"
    | "sourceRef"
    | "mode"
  >;
  receipt: CardMatch;
  refund: CardRefund;
  statement?: {
    importId: string;
    filename: string;
    sha256: string;
    size: number;
  };
  token: string;
  bankConfirmed: false;
}
export interface CardRefundAccountingBasis extends CardRefundAccountingOrigin {
  amount: number;
  allocations: CardRefundAccountingAllocation[];
}
export interface CardRefundAccountingContext {
  basis: CardRefundAccountingOrigin;
  claimedBy: string[];
  reservations: {
    journalId: string;
    refundId: string;
    matchId: string;
    recognitionId: string;
    amount: number;
    allocations: CardRefundAccountingAllocation[];
  }[];
  refundReserved: number;
  receiptReserved: number;
  liabilityReserved: number;
  refundRemaining: number;
  receiptRemaining: number;
  liabilityRemaining: number;
  debitLines: {
    index: number;
    line: Journal["lines"][number];
    reserved: number;
    remaining: number;
  }[];
}

export interface CardRefundAccountingChoices {
  recognitionId: string;
  recognitionVersion: number;
  choices: {
    transactionId: string;
    refundId: string;
    matchId: string;
    reference: string;
    merchant: string;
    date: string;
    refundAmount: number;
    receiptAmount: number;
    issue: string | null;
    journals: { id: string; state: Journal["state"]; amount: number }[];
  }[];
  scope: "simulated-card-refund-receipt-choices";
  bankConfirmed: false;
}

export interface CardRefundCloseEvidence {
  transaction: CardRefundAccountingOrigin["transaction"];
  refund: CardRefund;
  statement?: CardRefundAccountingOrigin["statement"];
  receipts: {
    matchId: string;
    expenseId: string;
    state: "active" | "released";
    label: string;
    amount: number;
    source: AccountingSource | null;
    documents: Document[];
    preparedBy: ActorContext | null;
    preparedAt: string;
    reason: string;
    issue: string | null;
  }[];
  postedAmount: number;
  otherPeriodAmount: number;
  draftAmount: number;
  remaining: number;
  bankConfirmed: false;
}
