# CFG-Distillation und Turbo-Trajektorien-Erhalt fuer Z-Image in OneTrainer

Stand: 2026-08-25

Dieses Dokument beschreibt ein technisches Konzept, wie OneTrainer die zwei in `ai-toolkit` funktionierenden Mechanismen fuer `Tongyi-MAI/Z-Image-Turbo` abbilden kann:

1. Assistant-/Training-LoRA: temporare De-Distillation des Turbo-Transformers waehrend des Trainings.
2. Differential Guidance: Veraenderung des Trainingsziels, nicht der Loss-Funktion.

Ziel ist ein minimaler, pruefbarer Z-Image-MVP. Generische Teacher-Distillation, Caches, KL-Losses und LoRA-Weight-Scaling bleiben bewusst ausserhalb des ersten Schritts.

## Ergebnis der Code-Recherche

### ai-toolkit

Verifizierte Konfigurationsfelder in `ai-toolkit/toolkit/config_modules.py`:

- `do_differential_guidance`, Default `False`
- `differential_guidance_scale`, Default `3.0`
- `assistant_lora_path`, Kommentar: "mainly for decompression loras for distilled models"

Assistant-LoRA fuer Z-Image ist in `ai-toolkit/extensions_built_in/diffusion_models/z_image/z_image.py` modellnah implementiert:

- `assistant_lora_path` wird lokal oder aus Hugging Face geladen.
- Adapter-Keys wie `diffusion_model.layers...` werden auf `transformer.layers...` umgeschrieben.
- Es wird ein transformer-only LoRA-Netz erzeugt.
- Der Adapter wird mit `merge_weight=1.0` in den Transformer gemergt.
- Danach bleibt ein invertierbarer Adapter erhalten:
  - Training: Basis-Transformer enthaelt den gemergten Assistant-Effekt.
  - Sampling/Inference waehrend Training: Assistant wird mit negativer Wirkung aktiviert, um den Merge-Effekt wieder zu entfernen.

Differential Guidance ist in `ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py` als Zieltransformation implementiert:

```python
with torch.no_grad():
    target = noise_pred + guidance_scale * (target - noise_pred)
```

Das ist keine eigene Loss-Funktion. Fuer MSE/L1-artige Losses skaliert es den Residual-Vektor zwischen aktueller Vorhersage und normalem Trainingsziel. Der anschliessende Loss bleibt der normale bestehende Loss zwischen `predicted` und `target`.

### OneTrainer

Relevante bestehende Stellen:

- `modules/trainer/GenericTrainer.py`
  - ruft `model_setup.predict(...)` auf.
  - ersetzt fuer `ConceptType.PRIOR_PREDICTION` bereits Zielwerte durch ein untrainiertes Prior-Modell.
  - ruft danach `model_setup.calculate_loss(...)` auf.
- `modules/modelSetup/BaseZImageSetup.py`
  - erzeugt fuer Z-Image Flow-Matching `predicted` und `target`.
  - `target = latent_noise - scaled_latent_image`
  - `predicted = - transformer_output`
- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`
  - berechnet bestehende Losses aus `data["predicted"]` und `data["target"]`.
  - ist nicht der richtige Ort fuer Differential Guidance, wenn ai-toolkit aequivalent nachgebaut werden soll.
- `modules/modelLoader/ZImageModelLoader.py`
  - laedt Z-Image Transformer, VAE, Text Encoder und Scheduler.
  - unterstuetzt Quantisierung bereits ueber bestehende Loader-Mixins.
- `modules/modelSetup/ZImageLoRASetup.py`
  - erzeugt und hookt die zu trainierende Transformer-LoRA.
- `modules/module/LoRAModule.py`
  - LoRA-Hooks existieren, aber dauerhaftes Mergen in das Basismodul ist fuer `LoRAModule.apply_to_module()` derzeit noch TODO.

Aktuelle Z-Image-Presets in OneTrainer nutzen teils den Turbo-Base-Model-Namen plus einen separaten De-Turbo-Transformer. Das ist nicht aequivalent zu ai-toolkit. ai-toolkit trainiert auf dem Turbo-Modell mit gemergtem Training-Adapter und invertiert den Adapter fuer den normalen Turbo-Inferenzpfad.

### PR #1360

PR: `Nerogar/OneTrainer#1360`, Titel `Draft: Add LoRA weight scaling and CFG distillation functionality`, Status offen.

Wichtige bestaetigte Punkte aus Diff und Review:

- Der PR mischt mehrere Themen:
  - LoRA-Weight-Scaling.
  - CFG-Distillation.
  - Parent-Model/Teacher-Wrapper.
  - Distillation-Cache.
  - neue Loss-Typen und UI.
- Autor bezeichnet den Stand selbst als rushed und wollte Features spaeter in kleinere PRs trennen.
- Reviewer-Kernpunkt: Distillation ist hier Zielveraenderung, keine neue Loss-Funktion.
- Cache-Key `image_path + timestep` ist unzureichend, weil Datenvariationen und Batch-Kontext fehlen.
- `ParentModelWrapper` dupliziert bestehende Quantisierungs-/Loader-Logik.
- `prior_target` und `PRIOR_PREDICTION` werden mit Distillation-Zielen vermischt.
- Distillation-Loss-Code wirkt nur in masked-loss-Pfaden und kann bei unmasked Training wirkungslos sein.
- KL/Temperature-Scaling ist fuer MSE-basierte Diffusionsziele irrefuehrend.
- LoRA-Weight-Scaling sollte separat betrachtet werden.

Empfehlung: PR #1360 nicht als Ganzes fertigstellen. Er taugt als Erfahrungsquelle, aber der saubere Weg ist ein kleiner Neuansatz fuer die zwei in ai-toolkit belegten Mechanismen.

## Zielbild

OneTrainer bekommt zuerst eine Z-Image-spezifische Funktion fuer distillierte Turbo-Modelle:

- Training laeuft auf Turbo + gemergtem Assistant-LoRA.
- Training-LoRA ist die einzige trainierbare LoRA und die einzige exportierte LoRA.
- Interne Samples waehrend des Trainings laufen auf Turbo-Trajektorie, indem der Assistant-Effekt fuer Sampling invertiert wird.
- Differential Guidance veraendert `target`, bevor der bestehende Loss berechnet wird.
- Keine Teacher-Model-Kopie, kein Distillation-Cache, kein neuer Loss-Typ.

Das entspricht dem erfolgreichen ai-toolkit-Verhalten enger als PR #1360.

## Konfiguration

Minimal neue Felder in `modules/util/config/TrainConfig.py`:

```python
assistant_lora_model_name: str = ""
differential_guidance: bool = False
differential_guidance_scale: float = 3.0
```

Benennung:

- `assistant_lora_model_name` passt zu OneTrainers bestehendem `*_model_name`-Schema, auch wenn der Wert lokal oder Hugging-Face-artig sein kann.
- `differential_guidance` statt `cfg_distillation_loss`, weil keine neue Loss-Funktion eingefuehrt wird.

Validierung:

- `differential_guidance_scale > 0`, wenn `differential_guidance=True`.
- Assistant-LoRA im MVP nur fuer `ModelType.Z_IMAGE` und LoRA-Training erlauben.
- Assistant-LoRA und separater De-Turbo-Transformer sollten nicht zusammen aktiv sein. Sonst wird De-Distillation doppelt oder unklar angewendet.
- Assistant-LoRA darf nicht in Optimizer-Parameter, Saves oder Exporte gelangen.

Optional sauberer Datenfluss:

- `modules/util/ModelNames.py` erhaelt ein optionales `assistant_lora`-Feld.
- `TrainConfig.model_names()` mappt `assistant_lora_model_name` darauf.
- `ZImageModelLoader` bekommt den Pfad dadurch ohne Zugriff auf die komplette Trainingskonfiguration.

## Differential Guidance

### Algorithmus

Nach `BaseZImageSetup.predict(...)` liegt bereits alles vor:

- `model_output_data["predicted"]`
- `model_output_data["target"]`
- `batch["concept_type"]`

Die Zieltransformation lautet:

```python
target = predicted.detach() + scale * (target - predicted.detach())
```

Das muss unter `torch.no_grad()` passieren, wie in ai-toolkit. Danach nutzt OneTrainer unveraendert die existierenden Loss-Funktionen.

### Einbauort

MVP: `modules/modelSetup/BaseZImageSetup.py`, kurz vor dem Rueckgabedict aus `predict(...)`.

Vorteile:

- Z-Image ist der konkret benoetigte Fall.
- Keine generische Distillation-Abstraktion vor dem ersten funktionierenden Pfad.
- Validation kann denselben Zielpfad sehen, wenn sie `predict(...)` nutzt.
- `GenericTrainer` und `ModelSetupDiffusionLossMixin` bleiben weitgehend unangetastet.

Wichtig: `ConceptType.PRIOR_PREDICTION` nicht mit Differential Guidance vermischen. Fuer Samples mit `PRIOR_PREDICTION` sollte die Zieltransformation im MVP uebersprungen werden, weil `GenericTrainer` diese Ziele spaeter durch Prior-Predictions ersetzt. Das vermeidet dieselbe Architekturvermischung, die in PR #1360 kritisiert wurde.

Falls spaeter echte Teacher-Distillation noetig wird, bekommt sie einen separaten `DISTILLATION`-Pfad. Sie sollte nicht ueber `PRIOR_PREDICTION` oder masked prior preservation versteckt werden.

## Assistant-/Training-LoRA

### Algorithmus

Der Z-Image-Transformer wird so vorbereitet:

1. Turbo-Transformer laden.
2. Assistant-LoRA laden.
3. Assistant-LoRA mit Gewicht `+1.0` in die Transformer-Gewichte mergen.
4. Danach erst finale Dtype-Konvertierung/Quantisierung anwenden.
5. Zu trainierende User-LoRA normal hooken und optimieren.
6. Fuer Trainings-Samples den Assistant-Adapter mit negativer Wirkung aktivieren, sodass der gemergte Assistant-Effekt entfernt wird.
7. Nach dem Sample wieder deaktivieren.

Die Reihenfolge "Merge vor Quantisierung" ist wichtig. ai-toolkit macht fuer Z-Image denselben Sonderpfad, weil ein nachtraegliches Mergen in bereits quantisierte Layer fehleranfaelliger ist.

### Betroffene Module

#### `modules/modelLoader/ZImageModelLoader.py`

Hier sollte der Assistant geladen und gemergt werden, weil das zum Modell-Ladezustand gehoert, nicht zur Trainingsschleife.

Noetige Erweiterung:

- Wenn `assistant_lora` gesetzt ist:
  - Transformer zunaechst nicht direkt in quantisierter Form laden.
  - Assistant-Adapter laden.
  - Adapter-Keys normalisieren.
  - Adapter in Transformer mergen.
  - Danach bestehende `_convert_diffusers_sub_module_to_dtype(...)` / Quantisierungspfad wiederverwenden.
- Wenn kein Assistant gesetzt ist:
  - bestehender Ladepfad bleibt unveraendert.

#### `modules/model/ZImageModel.py`

Neue nicht-trainierbare Laufzeitfelder:

- `transformer_assistant_lora`
- `assistant_lora_merged`
- optional ein kleiner Context Manager fuer inverse Aktivierung beim Sampling.

Diese Felder duerfen nicht Teil von `model.adapters()` sein, sofern `adapters()` fuer Training-Saves und Optimizer verwendet wird. Der Assistant ist Infrastruktur des Basismodells, kein Trainingsartefakt.

#### `modules/module/LoRAModule.py`

OneTrainer braucht fuer den MVP kein generisches PEFT-Merge-Framework. Es reicht:

- `LoRAModule.apply_to_module(merge_weight=1.0)` fuer plain Linear-LoRA.
- Ein Multiplikator oder Enable-Flag fuer den Delta-Pfad, Default `1.0`.

Formel fuer plain Linear-LoRA:

```text
delta_weight = lora_up.weight @ lora_down.weight * (alpha / rank) * merge_weight
base.weight += delta_weight
```

Grenzen im MVP:

- nur klassische LoRA auf Linear-Layern.
- kein LoHa/LoKr/OFT-Merge.
- klare Fehlermeldung, wenn der Assistant-Adapter nicht in dieses Format passt.

Das deckt den konkreten Z-Image-Training-Adapter ab und vermeidet PR-#1360-artige Breite.

#### Z-Image-Sampler

Der Sampling-Pfad sollte um einen Context Manager laufen:

```python
with model.inverse_assistant_lora_for_sampling():
    # bestehendes Sampling
```

Semantik:

- Training: Assistant-Merge bleibt im Transformer, Assistant-Hook ist wirkungslos.
- Sampling: Assistant-Hook wirkt mit `-1.0` und entfernt den Merge-Effekt.
- Nach Sampling: Zustand wird immer wiederhergestellt, auch bei Fehlern.

Der konkrete Einbauort ist der zentrale Z-Image-Sampling-Pfad, nicht die UI. Dadurch profitieren Training-Samples automatisch.

### Adapter-Key-Normalisierung

Der bekannte Adapter `ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors` nutzt ai-toolkit-artige Keys, zum Beispiel:

```text
diffusion_model.layers.0.attention.to_k.lora_A.weight
diffusion_model.layers.0.attention.to_k.lora_B.weight
```

OneTrainer sollte zuerst bestehende LoRA-Normalisierung wiederverwenden, falls sie diese Form schon abdeckt. Falls nicht, reicht fuer den MVP eine enge Z-Image-Mapping-Regel:

```text
diffusion_model. -> transformer.
lora_A.weight   -> lora_down.weight
lora_B.weight   -> lora_up.weight
```

Rank und Alpha koennen aus den Tensorformen abgeleitet werden, analog ai-toolkit: `linear=dim`, `linear_alpha=dim`.

## Preset- und UI-Konzept

UI erst nach funktionierendem Config-Pfad.

MVP-Konfiguration fuer den erfolgreichen Anwendungsfall:

- Base: `Tongyi-MAI/Z-Image-Turbo`
- kein separater De-Turbo-Transformer-Override
- Assistant: `ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors`
- `differential_guidance=True`
- `differential_guidance_scale=3.0`

Ein spaeteres Preset kann etwa "Z-Image Turbo LoRA with Training Adapter" heissen. Es sollte sichtbar machen, dass der Assistant nur waehrend Training/Sampling-Verwaltung genutzt und nicht exportiert wird.

Die UI sollte keine grosse Distillation-Sektion wie PR #1360 bekommen. Fuer den ersten Schritt reichen:

- ein Assistant-LoRA-Pfadfeld im relevanten Modell/LoRA-Kontext.
- ein Toggle fuer Differential Guidance.
- ein numerisches Feld fuer Scale.
- Validierungsfehler fuer inkompatible Kombinationen.

## Umgang mit Prior Preservation

Prior Preservation bleibt getrennt:

- `ConceptType.PRIOR_PREDICTION` bleibt fuer bestehende Prior-Prediction-Targets.
- Differential Guidance veraendert Standard-Trainingsziele.
- Assistant-LoRA veraendert den Modellzustand, nicht Concept Types.

Keine Wiederverwendung von `prior_target` fuer Distillation. Falls spaeter Teacher-Distillation implementiert wird, braucht sie eigene Datenfelder wie `distillation_target` oder `teacher_target`.

## Implementierungsplan

### PR 1: Differential Guidance fuer Z-Image

Umfang:

- Config-Felder und Migration aus aktueller Config-Version.
- Zieltransformation in `BaseZImageSetup.predict(...)`.
- Skip fuer `PRIOR_PREDICTION`.
- kleine Tests mit festen Tensoren:
  - Scale `3.0` ergibt `pred + 3 * (target - pred)`.
  - `PRIOR_PREDICTION` bleibt unveraendert.
  - deaktivierte Option ist exakt alter Pfad.

Nicht enthalten:

- neue Loss-Typen.
- Teacher-Modelle.
- Caches.
- UI-Grossumbau.

### PR 2: Assistant-LoRA fuer Z-Image

Umfang:

- Config-/ModelNames-Durchleitung fuer `assistant_lora_model_name`.
- Z-Image-Loader-Sonderpfad: Assistant vor Quantisierung mergen.
- minimale plain-LoRA-Merge-Funktion.
- Assistant-Multiplikator oder Enable-Flag fuer inverse Sampling-Aktivierung.
- Ausschluss aus Optimizer, Save und Export.
- Tests:
  - synthetischer Linear-Layer: `merged + inverse_hook` entspricht Ausgangsausgabe.
  - `merged + inactive_hook` unterscheidet sich erwartbar.
  - Assistant-Parameter erscheinen nicht in Optimizer-Parametern.
  - unbekannter Adaptertyp bricht mit klarer Fehlermeldung ab.

### PR 3: Preset, UI und Dokumentation

Umfang:

- kleines Z-Image-Turbo-Preset fuer Assistant + Differential Guidance.
- UI-Felder und Validierung.
- README/Training-Hinweis:
  - Assistant ist Training-Infrastruktur.
  - Exportierte LoRA enthaelt nur die trainierte User-LoRA.
  - Nicht mit separatem De-Turbo-Transformer kombinieren.

### Spaeter, nur bei Bedarf

- generische Differential-Guidance-Helfer fuer weitere Modelltypen.
- echte Teacher-Distillation mit separatem `DISTILLATION`-Concept-Type.
- Cache nur fuer offline vorgerenderte Teacher-Ziele mit vollstaendigem Key oder manifestbasiertem Datensatz, nicht als Bildpfad/Timestep-Hitcache in der Trainingsschleife.
- LoRA-Weight-Scaling als unabhaengiges Feature.

## Validierung gegen den konkreten Trainingsfall

Minimaler Nachweis:

1. Training auf demselben Latex-Datensatz mit `Tongyi-MAI/Z-Image-Turbo`.
2. Assistant `ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors`.
3. Differential Guidance Scale `3.0`.
4. Vergleichs-Samples mit dem normalen Zielpfad: 9 Steps, CFG 1.
5. Kontrolle, dass die exportierte LoRA keine Assistant-Keys enthaelt.
6. Kontrolle, dass Training-Samples intern die inverse Assistant-Aktivierung nutzen.

Erwartung:

- Materialsignal sollte wie in ai-toolkit frueh sichtbar werden.
- Keine Notwendigkeit fuer hohe Inferenz-Staerke oder lange Sampling-Pfade.
- Weniger Risiko fuer Kompositions-/Anatomie-Kollaps als beim naiven Training auf dem distillierten Turbo-Pfad.

## Risiken und offene Punkte

- Adapter-Key-Formate muessen am echten `zimage_turbo_training_adapter_v2.safetensors` getestet werden. Die enge Mapping-Regel ist wahrscheinlich ausreichend, aber nicht bewiesen, bis der Loader damit laeuft.
- Merge vor Quantisierung muss mit OneTrainers Z-Image-Quantisierungspfad sauber zusammenspielen. Bestehende Loader-Helfer sollten wiederverwendet werden, statt PR #1360s `ParentModelWrapper`-Ansatz zu kopieren.
- OneTrainer hat aktuell LoRA-Merge-TODOs. Der MVP sollte nur plain Linear-LoRA implementieren und andere PEFT-Typen explizit ablehnen.
- Differential Guidance als Zieltransformation ist fuer den ai-toolkit-Fall korrekt. Ob derselbe Scale fuer alle OneTrainer-Lossvarianten sinnvoll ist, sollte nicht vorab verallgemeinert werden.
- Bestehende De-Turbo-Presets koennen Nutzer in die falsche Kombination fuehren. Preset/UI-Validierung ist deshalb wichtig, aber erst nach dem funktionierenden Config-Pfad.

## Kurzentscheidung

Nicht PR #1360 umbauen. Stattdessen die zwei belegten Mechanismen klein und getrennt implementieren:

1. Differential Guidance als Z-Image-Zieltransformation vor dem bestehenden Loss.
2. Assistant-LoRA als Z-Image-Loader-/Sampling-Mechanismus mit Merge `+1.0` und inverser Sampling-Aktivierung.

Das ist naeher an ai-toolkit, vermeidet die kritisierten PR-#1360-Probleme und reduziert den ersten Implementierungsumfang auf den nachweislich benoetigten Turbo-LoRA-Fall.
