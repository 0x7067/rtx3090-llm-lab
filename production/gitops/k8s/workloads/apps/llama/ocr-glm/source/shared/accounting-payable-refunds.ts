import { z } from "zod";
import {
  accountingDate,
  type Journal,
  type AccountingSource,
} from "./accounting.js";
import type { PayableRefund } from "./payable-refunds.js";

export const payableRefundAccountingSchema = z
  .object({
    settlementId: z.string().uuid(),
    refundId: z.string().uuid(),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    date: accountingDate,
    understood: z.literal(true),
  })
  .strict();
export type PayableRefundAccountingFields = z.infer<
  typeof payableRefundAccountingSchema
>;
export interface PayableRefundAccountingBasis {
  kind: "reviewed-payable-refund-v1";
  settlementId: string;
  settlementVersion: number;
  recognitionId: string;
  source: AccountingSource;
  parent: Pick<
    Journal,
    | "id"
    | "date"
    | "version"
    | "lines"
    | "accounts"
    | "createdBy"
    | "postedBy"
    | "postedAt"
    | "payableSettlement"
    | "history"
  >;
  refund: PayableRefund;
  amount: number;
  token: string;
  bankConfirmed: false;
}
export interface PayableRefundAccountingContext {
  basis: PayableRefundAccountingBasis;
  reservations: { journalId: string; amount: number }[];
  remaining: number;
  claimedBy: string[];
}
