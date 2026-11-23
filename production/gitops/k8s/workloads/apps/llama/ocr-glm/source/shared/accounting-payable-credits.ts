import { z } from "zod";
import {
  accountingDate,
  type Journal,
  type AccountingSource,
} from "./accounting.js";
import type { PayableCreditNote } from "./payable-credit-notes.js";

export const payableCreditAllocationsSchema = z
  .array(
    z
      .object({
        recognitionLine: z.number().int().min(0).max(99),
        amount: z.number().int().positive().max(100_000_000),
      })
      .strict(),
  )
  .min(1)
  .max(99)
  .refine(
    (lines) =>
      new Set(lines.map((l) => l.recognitionLine)).size === lines.length,
    "Não repita a mesma linha do reconhecimento.",
  );
export const payableCreditAccountingSchema = z
  .object({
    recognitionId: z.string().uuid(),
    creditId: z.string().uuid(),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    date: accountingDate,
    allocations: payableCreditAllocationsSchema,
    understood: z.literal(true),
  })
  .strict();
export type PayableCreditAccountingFields = z.infer<
  typeof payableCreditAccountingSchema
>;
export type PayableCreditAllocation =
  PayableCreditAccountingFields["allocations"][number];
export interface PayableCreditAccountingOrigin {
  kind: "reviewed-payable-credit-v1";
  recognitionId: string;
  recognitionVersion: number;
  source: AccountingSource;
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
    | "postedAt"
    | "history"
  >;
  note: PayableCreditNote;
  amount: number;
  token: string;
  bankConfirmed: false;
}
export interface PayableCreditAccountingBasis extends PayableCreditAccountingOrigin {
  allocations: PayableCreditAllocation[];
}
export interface PayableCreditAccountingContext {
  basis: PayableCreditAccountingOrigin;
  reservations: {
    journalId: string;
    amount: number;
    allocations: PayableCreditAllocation[];
  }[];
  debitLines: {
    index: number;
    line: Journal["lines"][number];
    reserved: number;
    remaining: number;
  }[];
  claimedBy: string[];
}
