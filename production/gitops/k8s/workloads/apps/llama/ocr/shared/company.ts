import { z } from "zod";
import type { Persona } from "./types.js";

export const directoryId = z.string().regex(/^[a-zA-Z0-9-]{1,80}$/);
const name = z.string().trim().min(2).max(100);
const code = z
  .string()
  .trim()
  .regex(/^[A-Z0-9][A-Z0-9_-]{1,29}$/);
export const companyProfileSchema = z
  .object({
    name,
    legalName: z.string().trim().min(2).max(160),
  })
  .strict();
export const memberSchema = z
  .object({
    name,
    email: z.string().trim().toLowerCase().email().max(200),
    role: z.enum(["employee", "manager", "finance"]),
    active: z.boolean(),
    teamId: directoryId.nullable(),
    costCenterId: directoryId.nullable(),
  })
  .strict();
export const teamSchema = z
  .object({
    name,
    code,
    active: z.boolean(),
    leaderId: directoryId.nullable(),
  })
  .strict();
export const costCenterSchema = z
  .object({
    name,
    code,
    active: z.boolean(),
    teamId: directoryId.nullable(),
    ownerId: directoryId.nullable(),
  })
  .strict();
export const companySchemas = {
  profile: companyProfileSchema,
  members: memberSchema,
  teams: teamSchema,
  centers: costCenterSchema,
};
export type CompanyKind = keyof typeof companySchemas;
export type CompanyFields = {
  [K in CompanyKind]: z.infer<(typeof companySchemas)[K]>;
};
export type CompanyRecord<K extends CompanyKind> = CompanyFields[K] & {
  id: string;
  version: number;
  createdAt: string;
  updatedAt: string;
  // Immutable link to an existing local demo identity, never a login invitation.
  demoPersona: Persona | null;
};
export type CompanyHistory = {
  sequence: number;
  kind: CompanyKind;
  recordId: string;
  version: number;
  actor: string;
  reason: string;
  before: CompanyRecord<CompanyKind> | null;
  after: CompanyRecord<CompanyKind>;
  createdAt: string;
};
export interface CompanyDirectory {
  profile: CompanyRecord<"profile">;
  members: CompanyRecord<"members">[];
  teams: CompanyRecord<"teams">[];
  centers: CompanyRecord<"centers">[];
  history: CompanyHistory[];
  mode: "local-directory";
}
export interface CompanyOptions {
  centers: { id: string; name: string; code: string }[];
  teams: { id: string; name: string; code: string }[];
  assignment: { teamId: string | null; costCenterId: string | null };
  unregisteredCentersAllowed: true;
}

// Normalize case/spacing without collapsing distinct punctuation or accents.
export const directoryKey = (value: string) =>
  value.normalize("NFC").trim().replace(/\s+/g, " ").toLocaleLowerCase("pt-BR");
