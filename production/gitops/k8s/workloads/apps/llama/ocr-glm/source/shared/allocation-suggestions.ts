import { z } from "zod";
import type { Expense, ExpenseAllocation } from "./types.js";

export const allocationExampleId = z.string().regex(/^[a-zA-Z0-9-]{1,100}$/);
export const allocationSuggestionApply = z
  .object({
    sourceId: allocationExampleId,
    version: z.number().int().positive(),
    token: z.string().regex(/^[a-f0-9]{64}$/),
    reason: z.string().trim().min(5).max(1500),
    requestKey: z.string().uuid(),
  })
  .strict();
export type AllocationSuggestionApply = z.infer<
  typeof allocationSuggestionApply
>;
export interface AllocationExample {
  id: string;
  version: number;
  merchant: string;
  date: string;
  amount: number;
  approvalId: string;
  allocations: ExpenseAllocation[];
  documents: Expense["documents"];
  sameMerchant: boolean;
}
export interface AllocationSuggestion {
  method: "approved-example-ratio-v1";
  confidence: null;
  target: {
    id: string;
    version: number;
    amount: number;
    merchant: string;
    allocations: ExpenseAllocation[];
  };
  source: AllocationExample;
  rows: ExpenseAllocation[];
  token: string;
}

/** Exact centavo apportionment; stable source-order ties, no disappearing portion. */
export function scaleExampleAllocations(
  rows: ExpenseAllocation[],
  total: number,
): number[] {
  if (
    !Number.isSafeInteger(total) ||
    total < 1 ||
    total > 100_000_000 ||
    !rows.length ||
    rows.length > 30 ||
    rows.some(
      (r) =>
        !Number.isSafeInteger(r.amount) ||
        r.amount < 1 ||
        r.amount > 100_000_000,
    )
  )
    throw new Error("Confira o total salvo e as parcelas do exemplo.");
  const basis = rows.reduce((sum, row) => sum + BigInt(row.amount), 0n);
  const parts = rows.map((row, index) => {
    const product = BigInt(total) * BigInt(row.amount);
    return {
      index,
      amount: Number(product / basis),
      remainder: product % basis,
    };
  });
  const left = total - parts.reduce((sum, row) => sum + row.amount, 0);
  const ranked = [...parts].sort((a, b) =>
    a.remainder === b.remainder
      ? a.index - b.index
      : a.remainder > b.remainder
        ? -1
        : 1,
  );
  for (let index = 0; index < left; index++) ranked[index].amount++;
  if (parts.some((row) => row.amount === 0))
    throw new Error(
      "O total é pequeno demais para manter todas as partes do exemplo. Distribua manualmente.",
    );
  return parts.map((row) => row.amount);
}
