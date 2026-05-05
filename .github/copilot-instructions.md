<!-- VAULT-THEMES-SUBMODULE:START -->
VaultWares guidance lives in `vault-themes/AGENTS.md` and `vault-themes/CONTEXT.md`.
Read those files before UI, branding, design-system, token, auth UX, encrypted
communication UX, Figma-derived, or agent-instruction work.
<!-- VAULT-THEMES-SUBMODULE:END -->

<!-- AGENT-LEDGER:START -->
## Shared Agent Ledger — Mandatory

Before answering the user, record your activity in the shared ledger:

```powershell
& "C:\Users\Administrator\Desktop\Github Repos\agent-ledger\scripts\record-agent-change.ps1" `
  -Project "<repo name or General Tasks>" `
  -Kind "<plan|commands|code-change|verification|handoff|general>" `
  -Summary "<1024-token max summary of code changed, commands run, or plan made>" `
  -Commands @("<important command 1>", "<important command 2>") `
  -Files @("<important file 1>", "<important file 2>")
```

Do not log secrets. If the ledger cannot be accessed, tell the user in your reply.
Use `C:\Users\Administrator\Desktop\Github Repos\CHANGES.md`, `agent-ledger\CHANGES.md`,
and project roadmap/todo files to maintain continuity.
<!-- AGENT-LEDGER:END -->
