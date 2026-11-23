import { z } from "zod";
import type { ApprovedExpenseFundingSource } from "./expense-funding.js";
import type { ActorContext } from "./actor.js";
import type { Expense, ExpenseAllocation } from "./types.js";
import type { CardMatch, CardTransaction } from "./cards.js";
export interface CardReceiptSource extends ApprovedExpenseFundingSource {}
export interface CardReceiptCandidate {
  source: CardReceiptSource;
  version: number;
  reserved: number;
  reimbursementReserved?: number;
  available: number;
  maximum: number;
  parts: (ExpenseAllocation & { reserved: number; available: number })[];
}
export interface CardReceiptLinkView {
  match: CardMatch;
  source: CardReceiptSource | null;
  currentSource: CardReceiptSource | null;
  expenseVersion: number;
  current: boolean;
  issues: string[];
  legacy: boolean;
}
export interface CardReceiptReview {
  transaction: CardTransaction;
  basisToken: string;
  gross: number;
  reserved: number;
  covered: number;
  remaining: number;
  available: number;
  overReserved: boolean;
  stale: number;
  candidates: CardReceiptCandidate[];
  links: CardReceiptLinkView[];
  issues: string[];
  bankConfirmed: false;
}
const common = {
  version: z.number().int().positive(),
  expenseVersion: z.number().int().nonnegative(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
};
export const cardReceiptCommandSchema = z.discriminatedUnion("action", [
  z
    .object({
      ...common,
      action: z.literal("link"),
      expenseId: z.string().uuid(),
      amount: z.number().int().positive().max(100_000_000),
      parts: z
        .array(
          z
            .object({
              allocationId: z.string().min(1).max(80),
              amount: z.number().int().positive().max(100_000_000),
            })
            .strict(),
        )
        .min(1)
        .max(30)
        .optional(),
      understood: z.literal(true),
    })
    .strict(),
  z
    .object({
      ...common,
      action: z.literal("release"),
      matchId: z.string().uuid(),
    })
    .strict(),
]);
export type CardReceiptCommand = z.infer<typeof cardReceiptCommandSchema>;
export interface CardFundingComposition {
  reimbursementReserved?: number;
  gross: number;
  allocated: number;
  complete: boolean;
  current: boolean;
  links: {
    id: string;
    amount: number;
    transactionId: string | null;
    transactionGross: number | null;
  }[];
}
export interface CardReceiptAttribution {
  receiptVersion: 2;
  transaction: Pick<
    CardTransaction,
    | "id"
    | "cardId"
    | "owner"
    | "merchant"
    | "amount"
    | "date"
    | "origin"
    | "sourceRef"
  >;
  source: CardReceiptSource;
  parts: (Omit<ExpenseAllocation, "id" | "servicePeriod"> & {
    allocationId: string;
  })[];
  preparedBy: ActorContext;
  releasedBy?: ActorContext;
}
