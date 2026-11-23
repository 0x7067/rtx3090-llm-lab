import type { PurchaseAmendmentSource } from "./purchase-amendments.js";
import type { PurchaseMatchingSnapshot } from "./payable-matching.js";

export interface PurchaseAuthorityVersion {
  owner: string;
  fingerprint: string;
  source: PurchaseAmendmentSource;
}
export interface PurchaseAuthorityApplication {
  before: PurchaseAuthorityVersion;
  after: PurchaseAuthorityVersion;
  at: string;
  reason: string;
  receiving: { id: string; token: string }[];
  orders: { id: string; token: string }[];
  claims: { kind: "allocation" | "reservation"; id: string; token: string }[];
  invoices: {
    id: string;
    token: string;
    purchase: PurchaseMatchingSnapshot["purchase"];
  }[];
  sha256: string;
}
