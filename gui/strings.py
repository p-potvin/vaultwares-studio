"""UI strings (EN / QC French) + ``_t`` lookup and ``STATE_LABELS`` map.

The active language is module-level so widgets can swap it at runtime via
``set_active_lang('QC')`` without re-wiring every translator.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

from vaultwares_studio.pipeline import StageState


STATE_LABELS = {
    StageState.QUEUED.value: "Queued",
    StageState.RUNNING.value: "Running",
    StageState.NEEDS_INSTALL.value: "Needs Install",
    StageState.NEEDS_USER_INPUT.value: "Needs User Input",
    StageState.COMPLETE.value: "Complete",
    StageState.FAILED.value: "Failed",
}


_STRINGS: dict[str, dict[str, str]] = {
    "app_title":             {"EN": "Digital Twin Studio",                    "QC": "Studio de jumeau numérique"},
    "tab_studio":            {"EN": "Studio",                                  "QC": "Studio"},
    "tab_settings":          {"EN": "Settings",                                "QC": "Paramètres"},
    "current_job":           {"EN": "Current Job",                             "QC": "Travail actuel"},
    "source_video":          {"EN": "Source Video",                            "QC": "Vidéo source"},
    "choose_video":          {"EN": "Choose Video",                            "QC": "Choisir une vidéo"},
    "use_demo_video":        {"EN": "Use Demo Video",                          "QC": "Vidéo de démonstration"},
    "open_latest_job":       {"EN": "Open Latest Job",                         "QC": "Dernier travail"},
    "open_manifest":         {"EN": "Open Job Manifest",                       "QC": "Ouvrir le manifeste"},
    "job_steps":             {"EN": "Job Steps",                               "QC": "Étapes du travail"},
    "run_full_job":          {"EN": "Run Full Job",                            "QC": "Exécuter tout le travail"},
    "run_selected_step":     {"EN": "Run Selected Step",                       "QC": "Exécuter l'étape"},
    "open_job_folder":       {"EN": "Open Job Folder",                         "QC": "Ouvrir le dossier"},
    "run_state":             {"EN": "Run State",                               "QC": "État d'exécution"},
    "prompt_camera":         {"EN": "Prompt Camera Director",                  "QC": "Inviter le directeur de caméra"},
    "save_prompt":           {"EN": "Save Prompt",                             "QC": "Enregistrer l'invite"},
    "stage_previews":        {"EN": "Stage Previews",                          "QC": "Aperçus de l'étape"},
    "artifacts":             {"EN": "Artifacts",                               "QC": "Artéfacts"},
    "run_log":               {"EN": "Run Log",                                 "QC": "Journal d'exécution"},
    "clear_log":             {"EN": "Clear",                                   "QC": "Effacer"},
    "return_to_finish":      {"EN": "Return to Final Review",                  "QC": "Retour à la révision finale"},
    "final_review":          {"EN": "Final Review",                            "QC": "Révision finale"},
    "open_walkthrough":      {"EN": "Open Walkthrough Video",                  "QC": "Vidéo de visite guidée"},
    "open_3d_viewer":        {"EN": "Open Live 3D Viewer",                     "QC": "Visualiseur 3D en direct"},
    "open_output_folder":    {"EN": "Open Output Folder",                      "QC": "Dossier de sortie"},
    "integration":           {"EN": "VaultWares Integration",                  "QC": "Intégration VaultWares"},
    "api_url_label":         {"EN": "API Base URL",                            "QC": "URL de base de l'API"},
    "app_url_label":         {"EN": "App URL (Vault Flows)",                   "QC": "URL de l'application"},
    "bearer_label":          {"EN": "Bearer Token (optional)",                 "QC": "Jeton porteur (optionnel)"},
    "api_key_label":         {"EN": "API Key (optional)",                      "QC": "Clé API (optionnelle)"},
    "test_api":              {"EN": "Test API",                                "QC": "Tester l'API"},
    "export_workflow":       {"EN": "Export Workflow JSON",                    "QC": "Exporter le flux JSON"},
    "push_workflow":         {"EN": "Push Workflow",                           "QC": "Pousser le flux de travail"},
    "open_vault_flows":      {"EN": "Open Vault Flows",                        "QC": "Ouvrir Vault Flows"},
    "inspect_steps":         {"EN": "Inspect Previous Steps",                  "QC": "Inspecter les étapes précédentes"},
    "inspect_note":          {"EN": "The step rail remains live. Click any earlier step to reopen its viewer, previews, logs, and artifacts.",
                              "QC": "Le rail d'étapes reste actif. Cliquez sur une étape précédente pour rouvrir son visualiseur, ses aperçus, ses journaux et ses artéfacts."},
    "execution_mode_safe":   {"EN": "Execution mode: fallback-safe for local hardware.",
                              "QC": "Mode d'exécution : sécurisé pour le matériel local."},
    "execution_mode_strict": {"EN": "Execution mode: strict. Missing heavy tools fail the active stage.",
                              "QC": "Mode strict : les outils manquants font échouer l'étape active."},
    "enable_strict":         {"EN": "Enable Strict Tool Mode",                 "QC": "Activer le mode strict"},
    "disable_strict":        {"EN": "Disable Strict Tool Mode",                "QC": "Désactiver le mode strict"},
    "refresh_health":        {"EN": "Refresh Dependency Health",               "QC": "Vérifier les dépendances"},
    "health_title":          {"EN": "Dependency Health",                       "QC": "État des dépendances"},
    "theme_label":           {"EN": "Theme",                                   "QC": "Thème"},
    "preview_pending":       {"EN": "Preview pending",                         "QC": "Aperçu en attente"},
    "finish_summary":        {"EN": "The digital twin job completed. Open the final walkthrough video, launch the optional live 3D viewer, or inspect any previous step from the rail.",
                              "QC": "Le travail de jumeau numérique est terminé. Ouvrez la vidéo de visite finale, lancez le visualiseur 3D optionnel, ou inspectez une étape précédente."},
    "ready_to_execute":      {"EN": "Ready to execute the selected stage.",
                              "QC": "Prêt à exécuter l'étape sélectionnée."},
    "complete_earlier":      {"EN": "Complete earlier stages first, or use Run Full Job.",
                              "QC": "Terminez d'abord les étapes précédentes, ou utilisez Exécuter tout le travail."},
    "lang_switch_label":     {"EN": "FR",                                      "QC": "EN"},
    "no_video_yet":          {"EN": "No walkthrough video has been generated yet.",
                              "QC": "Aucune vidéo de visite n'a encore été générée."},
    "strict_off":            {"EN": "Strict mode: OFF",                        "QC": "Mode strict : OFF"},
    "strict_on":             {"EN": "Strict mode: ON",                         "QC": "Mode strict : ON"},
    "remote_title":          {"EN": "Remote Compute (Hugging Face Jobs)",      "QC": "Calcul à distance (Hugging Face Jobs)"},
    "hf_token_label":        {"EN": "HF Token (stored in OS keyring)",         "QC": "Jeton HF (stocké dans le trousseau)"},
    "hf_repo_label":         {"EN": "Artifact dataset repo (blank = auto)",    "QC": "Dépôt d'artéfacts (vide = auto)"},
    "hf_flavor_label":       {"EN": "Default GPU flavor",                      "QC": "Type de GPU par défaut"},
    "save_remote":           {"EN": "Save Remote Settings",                    "QC": "Enregistrer les paramètres"},
    "test_remote":           {"EN": "Run Echo Test Job (cpu-basic)",           "QC": "Tester avec un travail écho (cpu-basic)"},
    "remote_cost_title":     {"EN": "Confirm paid remote job",                 "QC": "Confirmer le travail payant"},
    "remote_cost_body":      {"EN": "This will launch a paid Hugging Face Job:\n{estimate}\n\nProceed?",
                              "QC": "Ceci lancera un travail Hugging Face payant :\n{estimate}\n\nContinuer ?"},
    "remote_saved":          {"EN": "Remote settings saved.",                  "QC": "Paramètres enregistrés."},
    "remote_no_token":       {"EN": "No HF token configured. Paste your token and save first.",
                              "QC": "Aucun jeton HF configuré. Collez votre jeton et enregistrez d'abord."},
    "preset_label":          {"EN": "Quality preset",                            "QC": "Préréglage de qualité"},
    "tab_viewport":          {"EN": "Viewport",                                  "QC": "Vue 3D"},
    "viewport_reload":       {"EN": "Reload Scene",                              "QC": "Recharger la scène"},
    "viewport_capture":      {"EN": "Capture Camera",                            "QC": "Capturer la caméra"},
    "viewport_loading":      {"EN": "Loading reconstruction…",                   "QC": "Chargement de la reconstruction…"},
    "viewport_no_scene":     {"EN": "No reconstruction yet — run a job, then reload.",
                              "QC": "Aucune reconstruction — exécutez un travail, puis rechargez."},
    "viewport_no_webengine": {"EN": "QtWebEngine is unavailable on this system. Use the Open Live 3D Viewer button instead.",
                              "QC": "QtWebEngine n'est pas disponible. Utilisez plutôt le visualiseur 3D en direct."},
    "viewport_captured":     {"EN": "Camera captured ({count} total) — saved to captured_cameras.json.",
                              "QC": "Caméra capturée ({count} au total) — enregistrée dans captured_cameras.json."},
    "viewport_cameras":      {"EN": "Captured cameras",                          "QC": "Caméras capturées"},
    "viewport_move_up":      {"EN": "Move Up",                                   "QC": "Monter"},
    "viewport_move_down":    {"EN": "Move Down",                                 "QC": "Descendre"},
    "viewport_delete":       {"EN": "Delete",                                    "QC": "Supprimer"},
    "viewport_preview_path": {"EN": "Preview Path",                              "QC": "Aperçu du trajet"},
    "viewport_need_two":     {"EN": "Capture at least 2 cameras to preview a path.",
                              "QC": "Capturez au moins 2 caméras pour prévisualiser un trajet."},
    "viewport_pattern_label":     {"EN": "Walk Pattern",
                                   "QC": "Trajectoire de marche"},
    "viewport_apply_pattern":     {"EN": "Apply + Preview Pattern",
                                   "QC": "Appliquer + Aperçu"},
    "viewport_pattern_no_preview": {"EN": "Need cloud_preview.ply before a pattern can be applied.",
                                    "QC": "cloud_preview.ply requis avant d'appliquer un motif."},
    "viewport_pattern_applied":    {"EN": "Pattern '{name}' applied — render path saved.",
                                    "QC": "Motif « {name} » appliqué — trajet de rendu enregistré."},
    "viewport_pattern_failed":     {"EN": "Pattern '{name}' failed: {error}",
                                    "QC": "Le motif « {name} » a échoué : {error}"},
    "viewport_view_top":           {"EN": "Top",        "QC": "Dessus"},
    "viewport_view_front":         {"EN": "Front",      "QC": "Avant"},
    "viewport_view_side":          {"EN": "Side",       "QC": "Côté"},
    "viewport_view_iso":           {"EN": "Iso",        "QC": "Iso"},
    "viewport_view_flip":          {"EN": "Flip Up",    "QC": "Inverser"},
    "viewport_section_view":       {"EN": "View",       "QC": "Vue"},
    "viewport_section_path":       {"EN": "Walk Path",  "QC": "Trajet"},
    "viewport_section_captures":   {"EN": "Captures",   "QC": "Captures"},
    "remote_declined":       {"EN": "Remote reconstruction declined; using the local quick path.",
                              "QC": "Reconstruction à distance refusée; utilisation du chemin local rapide."},
}


_active_lang: str = str(QSettings("VaultWares", "VaultwaresStudio").value("general/lang", "EN"))


def get_active_lang() -> str:
    return _active_lang


def set_active_lang(lang: str) -> None:
    global _active_lang
    _active_lang = lang


def t(key: str) -> str:
    """Return the localised UI string for the currently active language."""
    entry = _STRINGS.get(key, {})
    return entry.get(_active_lang, entry.get("EN", key))
