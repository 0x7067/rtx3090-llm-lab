import type { ReimbursementFundingRecord } from "./expense-funding.js";
import type { ServicePeriod } from "./service-periods.js";
import type { PurchaseRequest } from "./purchasing.js";
import type { CompanyOptions } from "./company.js";
import type { MileageDeclaration } from "./mileage.js";
export const categories = [
  "Alimentação",
  "Transporte",
  "Hospedagem",
  "Materiais",
  "Outros",
] as const;
export type Category = (typeof categories)[number];
export type Status =
  "draft" | "submitted" | "needs_info" | "approved" | "rejected" | "cancelled";
export type Persona = "manager" | "ana" | "finance";
export const statusLabels: Record<Status, string> = {
  draft: "Rascunho",
  submitted: "Em análise",
  needs_info: "Precisa de ajuste",
  approved: "Aprovada",
  rejected: "Recusada",
  cancelled: "Cancelada",
};
export interface ExpenseFields {
  merchant: string;
  amount: number;
  date: string;
  category: Category;
  costCenter: string;
  purpose: string;
  allocations?: ExpenseAllocation[];
}
export interface ExpenseAllocation {
  servicePeriod?: ServicePeriod;
  id: string;
  description: string;
  category: Category;
  costCenter: string;
  amount: number;
}
export interface Policy {
  enforcement?: PolicyEnforcement;
  category: Category;
  limit: number;
  version: number;
}
export interface PolicyCheck {
  enforcement?: PolicyEnforcement;
  limit: number;
  excess: number;
  version: number;
  categoryChecks?: {
    enforcement?: PolicyEnforcement;
    category: Category;
    amount: number;
    limit: number;
    excess: number;
    version: number;
  }[];
}
export type PolicyEnforcement = "advisory" | "exception" | "block";
export interface Document {
  readingView?: import("./document-reading.js").DocumentReadingView;
  id: string;
  name: string;
  mime: string;
  size: number;
  sha256: string;
  demo: boolean;
  fiscal?: FiscalDocument;
}
export interface FiscalDocument {
  model: "55" | "65";
  key: string;
  number: string;
  issuer: string;
  issuerId: string;
  date: string;
  total: number;
  items: {
    description: string;
    quantity: string;
    unit: string;
    total: number;
  }[];
}
export interface AuditEvent {
  id: number;
  actor: string;
  action: string;
  note: string;
  createdAt: string;
}
export interface Expense extends ExpenseFields {
  mileage?: MileageDeclaration;
  id: string;
  employee: string;
  owner: string;
  status: Status;
  version: number;
  createdAt: string;
  updatedAt: string;
  demo: boolean;
  documents: Document[];
  history: AuditEvent[];
  policy: PolicyCheck;
  reimbursement?: Reimbursement;
  reimbursementLedger?: import("./reimbursement-installments.js").ReimbursementLedgerSummary;
}
export interface Reimbursement {
  amount?: number;
  funding?: ReimbursementFundingRecord;
  expenseId: string;
  state: "scheduled" | "paid" | "cancelled";
  date: string;
  reference: string;
  version: number;
  updatedAt: string;
}
export interface DuplicateCase {
  id: string;
  expenseIds: [string, string];
  basis: "document" | "fields" | "fiscal_key" | "mileage";
  amount: number;
  state: "open" | "dismissed" | "confirmed" | "recovered" | "prevented";
  note: string;
  recovered: number;
  reference: string;
  version: number;
}
export interface Budget {
  month: string;
  costCenter: string;
  limit: number;
  version: number;
}
export interface Operations {
  budgetPosition: {
    asOf: string;
    reports: import("./budget-consumption.js").BudgetConsumptionReport[];
  };
  cases: DuplicateCase[];
  budgets: Budget[];
  purchases: PurchaseRequest[];
}
export interface Bootstrap {
  directory: CompanyOptions;
  expenses: Expense[];
  policies: Policy[];
  cases?: DuplicateCase[];
  capabilities: {
    ocr: false;
    llm: false;
    authentication: "demo" | "password";
    audit: "local-mock-v1";
  };
}
