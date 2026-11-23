import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type { Vendor } from "./payables.js";
import { directoryKey } from "./company.js";

// Intentionally preserves accents, punctuation, suffixes and identifiers.
export const vendorNameKey = directoryKey;
export const aliasName = z
  .string()
  .trim()
  .min(2)
  .max(160)
  .refine(
    (v) => /[\p{L}\p{N}]/u.test(v) && !/[\p{Cc}\p{Cf}]/u.test(v),
    "Informe um nome legível, sem caracteres de controle.",
  );
const reason = z.string().trim().min(5).max(1500);
export const aliasProposal = z
  .object({
    vendorVersion: z.number().int().positive(),
    name: aliasName,
    basis: reason,
    reason,
    requestKey: z.string().uuid(),
  })
  .strict();
export const aliasDecision = z
  .object({
    version: z.number().int().positive(),
    action: z.enum(["accept", "reject", "withdraw", "revoke"]),
    reason,
    requestKey: z.string().uuid(),
  })
  .strict();
export type AliasState =
  "proposed" | "accepted" | "rejected" | "withdrawn" | "revoked";
export interface VendorAlias {
  id: string;
  vendorId: string;
  version: number;
  name: string;
  nameKey: string;
  basis: string;
  state: AliasState;
  supplier: Vendor;
  proposedBy: ActorContext;
  proposedAt: string;
  decidedBy?: ActorContext;
  decidedAt?: string;
  decisionReason?: string;
}
export interface AliasHistory {
  aliasId: string;
  before: VendorAlias | null;
  after: VendorAlias;
  actor: ActorContext;
  reason: string;
  createdAt: string;
}
export interface AliasView extends VendorAlias {
  available: boolean;
  warnings: string[];
  history: AliasHistory[];
}
export interface AliasWorkspace {
  vendor: Vendor;
  aliases: AliasView[];
}
export interface SupplierLookup {
  input: string;
  nameKey: string;
  status: "none" | "unique" | "ambiguous" | "unavailable";
  candidates: {
    vendor: Vendor;
    method: "registered-name" | "reviewed-alias";
    alias: AliasView | null;
    available: boolean;
    warnings: string[];
  }[];
}
