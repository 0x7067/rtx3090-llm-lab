import { z } from "zod";
import { categories } from "./types.js";
import { expenseAllocationsSchema } from "./expense-allocations.js";
export const expenseIdSchema = z
  .string()
  .regex(/^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$/);
export const expenseDateSchema = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(v + "T12:00:00Z");
    return Number.isFinite(+d) && d.toISOString().slice(0, 10) === v;
  }, "Data inválida");
export const expenseFieldsSchema = z
  .object({
    merchant: z.string().trim().max(160),
    amount: z.number().int().min(0).max(100_000_000),
    date: expenseDateSchema,
    category: z.enum(categories),
    costCenter: z.string().trim().max(100),
    purpose: z.string().trim().max(1500),
    allocations: expenseAllocationsSchema.optional(),
  })
  .strict();
