import { servicePeriodSchema } from "./service-periods.js";
import { z } from "zod";
import { directoryKey } from "./company.js";
import {
  categories,
  type ExpenseAllocation,
  type ExpenseFields,
  type Policy,
  type PolicyCheck,
} from "./types.js";

export const expenseAllocationSchema = z
  .object({
    id: z.string().uuid(),
    description: z.string().trim().min(1).max(300),
    category: z.enum(categories),
    costCenter: z.string().trim().min(1).max(100),
    amount: z.number().int().positive().max(100_000_000),
    servicePeriod: servicePeriodSchema.optional(),
  })
  .strict()
  .refine(
    (v) =>
      !v.servicePeriod ||
      v.servicePeriod.months.reduce((n, m) => n + m.amount, 0) === v.amount,
    {
      message: "Os valores mensais devem somar exatamente o valor desta parte.",
      path: ["servicePeriod"],
    },
  );
export const expenseAllocationsSchema = z
  .array(expenseAllocationSchema)
  .max(30)
  .refine(
    (rows) => new Set(rows.map((row) => row.id)).size === rows.length,
    "Cada parcela precisa de um identificador distinto.",
  );
type ClassifiedAmount = Pick<
  ExpenseFields,
  "amount" | "category" | "costCenter" | "allocations"
>;
const safeSum = (amounts: number[]) => {
  const total = amounts.reduce((sum, amount) => sum + amount, 0);
  if (!Number.isSafeInteger(total))
    throw new Error("O total das parcelas excede o limite seguro.");
  return total;
};

// Calculation-only fallback. Never persist this synthetic row in legacy records.
export function effectiveExpenseAllocations(
  fields: ClassifiedAmount,
): ExpenseAllocation[] {
  return fields.allocations?.length
    ? fields.allocations
    : [
        {
          id: "legacy",
          description: "Classificação única",
          category: fields.category,
          costCenter: fields.costCenter,
          amount: fields.amount,
        },
      ];
}
export function expenseCategoryAmounts(fields: ClassifiedAmount) {
  const rows = effectiveExpenseAllocations(fields);
  return categories
    .filter((category) => rows.some((row) => row.category === category))
    .map((category) => ({
      category,
      amount: safeSum(
        rows
          .filter((row) => row.category === category)
          .map((row) => row.amount),
      ),
    }));
}
export function expenseCenterAmount(
  fields: Pick<ExpenseFields, "amount" | "costCenter" | "allocations">,
  name: string,
): number {
  return fields.allocations?.length
    ? safeSum(
        fields.allocations
          .filter((row) => directoryKey(row.costCenter) === directoryKey(name))
          .map((row) => row.amount),
      )
    : directoryKey(fields.costCenter) === directoryKey(name)
      ? fields.amount
      : 0;
}
export function calculateExpensePolicy(
  fields: ClassifiedAmount,
  policies: Policy[],
): PolicyCheck {
  const header = policies.find((policy) => policy.category === fields.category);
  if (!header) throw new Error("Política de categoria não encontrada.");
  if (!fields.allocations?.length)
    return {
      limit: header.limit,
      excess: Math.max(0, fields.amount - header.limit),
      version: header.version,
      ...(header.enforcement ? { enforcement: header.enforcement } : {}),
    };
  const categoryChecks = expenseCategoryAmounts(fields).map(
    ({ category, amount }) => {
      const policy = policies.find((policy) => policy.category === category);
      if (!policy) throw new Error("Política de categoria não encontrada.");
      return {
        category,
        amount,
        limit: policy.limit,
        excess: Math.max(0, amount - policy.limit),
        version: policy.version,
        ...(policy.enforcement ? { enforcement: policy.enforcement } : {}),
      };
    },
  );
  return {
    limit: safeSum(categoryChecks.map((check) => check.limit)),
    excess: safeSum(categoryChecks.map((check) => check.excess)),
    version: header.version,
    categoryChecks,
  };
}

// An omitted edit retains the breakdown; [] explicitly returns to one classification.
export function normalizeExpenseAllocations<T extends ExpenseFields>(
  fields: T,
  previous?: ExpenseFields,
): T {
  const result = { ...fields };
  const allocations = expenseAllocationsSchema.parse(
    fields.allocations ?? previous?.allocations ?? [],
  );
  if (allocations.length) {
    if (safeSum(allocations.map((row) => row.amount)) !== fields.amount)
      throw Object.assign(
        new Error(
          "A soma das parcelas deve ser igual ao valor total da despesa. Redistribua as parcelas antes de alterar o total.",
        ),
        { statusCode: 400 },
      );
    result.allocations = allocations;
    result.category = allocations[0].category;
    result.costCenter = allocations[0].costCenter;
  } else delete result.allocations;
  return result;
}
