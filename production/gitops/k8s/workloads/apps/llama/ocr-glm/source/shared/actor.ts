import type { Persona } from "./types.js";

export type MemberRole = "employee" | "manager" | "finance";
/** Resolved by the server. Demo selection is still not authentication. */
export interface ActorContext {
  readonly subjectId: string;
  readonly memberId: string;
  readonly companyId: string;
  readonly role: MemberRole;
  readonly displayName: string;
  readonly authentication: "demo" | "password";
}
/** Legacy literals remain accepted by internal callers during the migration. */
export type Actor = Persona | ActorContext;
const legacy = {
  ana: { role: "employee", name: "Ana Costa" },
  manager: { role: "manager", name: "Marina Costa" },
  finance: { role: "finance", name: "Rafael Lima" },
} as const;
function legacyMember(actor: Persona) {
  const member = legacy[actor];
  if (!member) throw new Error("Unknown legacy demo actor");
  return member;
}
export function actorId(actor: Actor): string {
  if (typeof actor !== "string") return actor.memberId;
  legacyMember(actor);
  return actor;
}
export function actorRole(actor: Actor): MemberRole {
  return typeof actor === "string" ? legacyMember(actor).role : actor.role;
}
export function actorDisplayName(actor: Actor): string {
  return typeof actor === "string" ? legacyMember(actor).name : actor.displayName;
}
export function actorSnapshot(actor: Actor): ActorContext {
  if (typeof actor !== "string") return { ...actor };
  return {
    subjectId: `legacy-demo:${actor}`,
    memberId: actor,
    companyId: "legacy-demo",
    role: legacyMember(actor).role,
    displayName: legacyMember(actor).name,
    authentication: "demo",
  };
}
/** Preserve historical demo receipts; isolate new members' command UUIDs. */
export function actorCommandKey(actor: Actor, requestKey: string): string {
  const id = actorId(actor);
  if (id === "ana" || id === "manager" || id === "finance") return requestKey;
  if (typeof actor === "string") throw new Error("Unknown legacy demo actor");
  return `member:${actor.companyId}:${id}:${requestKey}`;
}
