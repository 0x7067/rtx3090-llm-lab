import { z } from "zod";
import { accountingDate } from "./accounting.js";
import type { AccountingSource } from "./accounting.js";
import type { Reimbursement } from "./types.js";

export const reimbursementSettlementSchema = z
  .object({
    recognitionId: z.string().uuid(),
    paymentId: z.string().uuid().optional(),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    date: accountingDate,
    liabilityAccountId: z.string().uuid(),
    assetAccountId: z.string().uuid(),
    understood: z.literal(true),
  })
  .strict();
export type ReimbursementSettlementFields = z.infer<
  typeof reimbursementSettlementSchema
>;
export interface ReimbursementSettlementBasis {
  kind: "declared-reimbursement-v1" | "declared-reimbursement-installment-v1";
  installment?: import("./reimbursement-installments.js").ReimbursementInstallment;
  recognitionId: string;
  recognitionVersion: number;
  source: AccountingSource;
  payment: Reimbursement;
  amount: number;
  token: string;
  bankConfirmed: false;
}
export interface ReimbursementSettlementContext {
  basis: ReimbursementSettlementBasis;
  liabilityAccounts: { id: string; amount: number }[];
  claimedBy: string[];
  reservations?: {journalId:string;accountId:string;amount:number}[];
  payments?: {id:string;amount:number;date:string;reference:string}[];
}
