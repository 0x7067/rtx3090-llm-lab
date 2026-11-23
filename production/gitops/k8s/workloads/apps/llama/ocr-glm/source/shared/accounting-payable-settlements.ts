import { z } from "zod";
import { accountingDate, type AccountingSource } from "./accounting.js";
import type { PayablePayment } from "./payable-payments.js";
export const payableSettlementSchema = z
  .object({
    recognitionId: z.string().uuid(),
    paymentId: z.string().uuid(),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    date: accountingDate,
    liabilityAccountId: z.string().uuid(),
    assetAccountId: z.string().uuid(),
    understood: z.literal(true),
  })
  .strict();
export type PayableSettlementFields = z.infer<typeof payableSettlementSchema>;
export interface PayableSettlementBasis {
  kind: "declared-payable-payment-v1";
  recognitionId: string;
  recognitionVersion: number;
  source: AccountingSource;
  payment: PayablePayment;
  amount: number;
  token: string;
  bankConfirmed: false;
}
export interface PayableSettlementContext {
  basis: PayableSettlementBasis;
  liabilityAccounts: { id: string; amount: number }[];
  reservations: { journalId: string; accountId: string; amount: number }[];
  claimedBy: string[];
}
