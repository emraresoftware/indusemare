# INDUS-DEDUCT (`indusemare`) — Global Launch & Media Kit

- **Live Portal (TR / EN auto-detect, 7 tabs):** https://indus.emare.dev/ (English direct: `https://indus.emare.dev/?lang=en`)
- **Permanent CERN / Zenodo Concept DOI (always latest):** https://doi.org/10.5281/zenodo.24383347
- **Preprint PDF (4 embedded figures, Layers 0–13):** https://indus.emare.dev/assets/INDUS_DEDUCT_PREPRINT.pdf
- **GitHub Repository (`v1.1.0`, 12/12 tests PASS):** https://github.com/emraresoftware/indusemare
- **1-Page Referee Claim Ledger:** https://github.com/emraresoftware/indusemare/blob/main/CLAIM_LEDGER.md
- **Social Share Card (1200×630):** `portal/assets/og-indusemare.png`

---

## 1. Hacker News (`Show HN`) — Copy-Paste Ready

**Title:**
`Show HN: INDUS-DEDUCT – Calibrated computational benchmark for the Indus script (5,679 texts)`

**URL:**
`https://indus.emare.dev/?lang=en`

**First Comment (Post immediately after submission):**
> Hi HN — instead of another "I deciphered the Indus script!" claim, we built an anti-hallucination computational epigraphy benchmark across all 5,679 published Indus inscriptions (5,805 lines; 713 raw signs), strictly stripping all speculative upstream columns (`sanskrit`, `translation`, `notes`) and verifying 12/12 invariants bit-for-bit across macOS and Linux.
>
> Four things surprised us most when we calibrated our metrics against real **Ur III Sumerian (CDLI)**, **Linear B**, **Linear A**, **European & Chinese Folk Melodies (ABC)**, **South Indian Carnatic *Swara/Tāla***, **pure rhythmic durations**, and a **structured non-linguistic stamp negative control** (all matched to N=2,047 and exact Indus length distribution):
>
> 1. **We caught TWO major methodological traps (`z_H1` AND Kober `MI z` on music!):** Our rigid non-linguistic stamp negative control scored `z_H1 = -245.34σ` (more than 2× Indus's `-108.85σ`), retiring Rao-style conditional entropy; AND real melodic/rhythmic music scored `Kober MI z = +17.9σ` to `+151.1σ`! Combining **low within-sequence repetition** (`Indus 4.4%`, `Sumerian 9.3%` vs. `Music/Rhythm 52.6%–93.2%`), **asymmetric 2-affix stacking** (`Indus 207:1`, bottleneck `4.96×` vs. `Music/Stamps 0:0`), and **unseen-trigram generalization** (`Indus +6.38 pp`, `Sumerian +10.71 pp` vs. `Carnatic -12.05 pp`, `Stamps -2.95 pp`) cleanly separates Indus & natural scripts from *both* music and stamps.
> 2. **First quantitative BH-FDR paleographic drift chains (700 years):** After de-confounding Mohenjo-daro excavation trench datums (`n=1,253` within-area; single-datum `DKG(S)` `n=595`) and replicating against Harappa `Period 2–5` (`Spearman rho = +0.289, p = 0.0376`), two visually similar allograph chains shift chronologically (`q < 0.05`): Carrier `#156 → #154` (`q = 0.0083`) and a 3-stage sprouting leaf/seed sign `#838 → #803 → #806` (`q = 0.0339`, with `#806` independently #1 late sign 600 km away in Harappa, `q = 0.0106`). Core square-seal grammar stayed frozen for 700 years (`p > 0.09`).
> 3. **Why nobody has won Tamil Nadu's $1M prize yet (Known-Answer CSP proof):** On the 23-seal Gulf/Mesopotamia connected component (`DOF = 30`, `66%` singletons), our exact Max-CSP locks `0` signs. To prove it's not solver blindness, we ran the exact same solver on 23-seal closed graphs from real Linear B (`23/23`, `+22.75σ`, 100% signs locked) and Ur III Sumerian (`23/23`, `+7.00σ`). Under realistic 74% open-vocab Sumerian, `k = 0` anchors crashes to noise (`-0.11σ`), while **`k = 2` true anchors** jumps to `21/23` (`+4.97σ`, 68.2% shared signs locked). Expanding foreign Indus seals `24 → 55` actually *increases* `DOF` (`29 → 45`) — proving we need `≥2` bilingual anchors, not just a bigger neural net.
> 4. **Open 690-inscription damaged restoration catalog:** Under strict 12-city Leave-One-City-Out CV (zero local duplicate leakage), our visual-glyph SVD + neural MLP hybrid reaches `57.5%` Top-1 / `82.9%` Top-5 on `n=275` edge breaks (baseline `10.2%`) and `40.0%` / `60.0%` Top-3 on interior breaks.
>
> Everything is open-source (MIT / CC-BY-4.0), archived permanently on Zenodo (DOI: https://doi.org/10.5281/zenodo.24383348), with a 1-page Tier A (Proven) / Tier B (Leads) / Tier C (Retired/Null) Claim Ledger: https://github.com/emraresoftware/indusemare/blob/main/CLAIM_LEDGER.md and full 11-page PDF: https://indus.emare.dev/assets/INDUS_DEDUCT_PREPRINT.pdf. Zero-dependency verification (`python3 -m unittest tests/test_claim_ledger.py -v`) runs in 4 ms on a fresh clone. Brutal critiques welcome!

---

## 2. X / Twitter Thread (English — 6 Posts, Attach Figures 1–3!)

**Post 1 (Attach `portal/assets/og-indusemare.png`):**
> 99% of Indus Valley script posts claim "I deciphered it!"
>
> We did the opposite: a 13-layer anti-hallucination benchmark across all 5,679 inscriptions — retiring a famous entropy test, uncovering 700-year glyph evolution, restoring 690 broken seals & proving why the $1M prize is still locked.
>
> Live portal + code + PDF: https://indus.emare.dev/?lang=en 🧵👇

**Post 2 (Attach `reports/calibration_5way_chart.png`):**
> 1/ First shock: conditional entropy under shuffling (`z_H1`) does NOT prove natural language.
>
> Our structured Non-Linguistic Stamp negative control scored `-245.3σ` (2× Indus!).
>
> What *does* separate Indus + Ur III Sumerian + Linear B from non-linguistic stamps?
> • Kober bridge MI: `+53.7σ` vs `+0.07σ`
> • Unseen-trigram gain: `+6.38pp` vs `-2.95pp`
> • Blind 2-affix stacking: `207:1` vs `0:0`

**Post 3 (Attach `reports/layer12_time_evolution_plate.png`):**
> 2/ Second shock: 700 years of paleographic drift across Mohenjo-daro & Harappa (600 km apart, `rho = +0.289, p = 0.038`).
>
> Two visual allograph chains survive global BH-FDR (`q < 0.05`):
> • Carrier `#156 → #154` (`q = 0.0083`; 2.3m depth shift inside single trench `DKG(S)`)
> • A 3-stage sprouting sign `#838 (1 line) → #803 (3 branches) → #806 (5 branches)` (`q = 0.034`) — and `#806` is #1 late sign in Harappa too (`q = 0.011`)!

**Post 4 (Attach `reports/layer9_copper_rosetta_plate.png`):**
> 3/ Mohenjo-daro Copper Tablets (`229/230` city-exclusive) show a `96.2%` Text ↔ Icon lock (`+19.7σ` vs square seals `+2.8σ`) + 23 two-sided single-sign equations (e.g., quadruped pictograph `#341 ↔ 503 615 740`).
>
> Yet 4 blind tests — incl. a 4-language-family phonological null (`p > 0.30`) — prove they are compound ritual/institutional formulas, NOT bare animal nouns.

**Post 5:**
> 4/ Trade logistics & maritime split:
> • Harappan tablets (`98.9%` inland) carry freshwater river icons (`Gavi` gharial `62/66`, `Fish` `47/47`).
> • Maritime trade lives on clay cargo sealings (`65.3%` coastal) & round Gulf seals — anchored by Lothal's master customs seal co-signing Mohenjo-daro stone seal `#2239.1` on the same wet clay lump, and Twin-Humans `#091` (`0%` final, `42.4×` export corridor `q = 2.9e-4`)!

**Post 6:**
> 5/ Why is the $1M decipherment prize still open?
>
> Our exact Max-CSP locks `23/23` (`100%`) of closed Linear B (`+22.8σ`) & Ur III Sumerian (`+7.0σ`) at `k=0` anchors. Under realistic 74% open Sumerian, `k=0` crashes to noise (`-0.11σ`), and **`k=2` anchors** rescues it (`+4.97σ`).
>
> All code, 690 restored damaged seals, 11/11 tests & 11-page PDF:
> 🌐 https://indus.emare.dev/?lang=en
> 💻 https://github.com/emraresoftware/indusemare

---

## 3. X / Twitter Flood (Türkçe — 5 Gönderi)

**1/5 (Görsel: `og-indusemare.png`):**
> İnternetin %99'u "İndus yazısını çözdüm!" diyen temelsiz iddialarla dolu.
>
> Biz tam tersini yaptık: 5.679 İndus yazıtında 13 katmanlı, sıfır-halüsinasyonlu açık benchmark kurduk — 2009'un meşhur entropi testini çürüttük, 700 yıllık ilk paleografik glif evrimini yakaladık, 690 kırık mührü onardık.
>
> Canlı portal + PDF + açık kod: https://indus.emare.dev 🧵👇

**2/5 (Görsel: `calibration_5way_chart.png`):**
> 1) "Dil mi, damga mı?" tartışmasını gerçek Ur III Sümerce (CDLI), Linear B, Linear A ve Dil-Dışı Damgalarla eşit boyutlu 5'li kalibrasyona soktuk:
> • Meşhur `z_H1` entropi testi dil-dışı damgalarda `-245σ` vererek ÇÖKTÜ (emekli ettik).
> • Ama 3 sağlam test İndus'u gerçek dillerle aynı kümeye oturttu: Kober `+53.7σ` (vs `+0.07σ`), Görülmemiş trigram `+6.38 puan` (vs `-2.95`), 2-ek yığılması `207:1` (vs `0:0`).

**3/5 (Görsel: `layer12_time_evolution_plate.png`):**
> 2) Kazı derinliği sıfır-noktası tuzağını temizleyip Mohenjo-daro ve 600 km ötedeki Harappa'yı çapraz doğrulayınca (`rho = +0.289`) İndus'un 700 yıllık ilk nicel paleografik evrim zincirleri çıktı:
> • Taşıyıcı `#156 → #154` (`q = 0.0083`, aynı açmada 2.3 m derinlik farkı)
> • 3 aşamalı filizlenme `#838 (1 çizgi) → #803 (3 dal) → #806 (5 dal)` (`q = 0.034`) — `#806` Harappa'da da 1 numaralı geç dönem işareti!

**4/5 (Görsel: `layer9_copper_rosetta_plate.png`):**
> 3) Mohenjo-daro Bakır Tabletlerinde (`229/230`) metin ↔ ikon kilidi `%96.2` (`+19.7σ`); 23 çift-yüzlü tablette `#341` dört-ayaklı hayvan piktogramı ↔ `503 615 740` gibi tek-işaret denklemleri var.
> Lothal limanında aynı ıslak kil topağında gümrük ana mührü ile 700 km ötedeki Mohenjo-daro taş mührü yan yana! Ve Çift İnsan `#091` `%0` bitişle `42.4×` ihracat başlığı.

**5/5:**
> 4) 1 milyon dolarlık Tamil Nadu ödülüne neden kimse ulaşamadı?
> Aynı çözücümüz kapalı Linear B (`+22.8σ`) ve Sümercede (`+7.0σ`) %100 kilit açarken, %74 açık sözlüklü gerçek Sümercede `0` çapayla çöküp `2` iki-dilli çapayla (`+4.97σ`) açılıyor. Yani duvar algoritmada değil, en az 2 iki-dilli çapada.
>
> 690 kırık yazıt onarımı + 11/11 doğrulanmış açık kod:
> 🌐 https://indus.emare.dev
> 💻 https://github.com/emraresoftware/indusemare

---

## 4. LinkedIn Duyurusu (Türkçe & İngilizce Kısa Blok)

**Türkçe:**
> 4.500 yıllık çözülememiş İndus Vadisi yazısı için sıfır-halüsinasyonlu, uçtan uca tekrarlanabilir bir hesaplamalı epigrafi benchmark'ı yayımladık: **INDUS-DEDUCT v1.0** (`https://indus.emare.dev`).
>
> 5.679 yazıt, 13 katman, Mac ve Linux üzerinde bit-bit SHA-256 doğrulamalı **11/11 otomatik iddia testi** ile ne yaptık?
> ✅ **5'li Kalibrasyon:** Gerçek Ur III Sümerce (CDLI), Linear B, Linear A ve Dil-Dışı Damgalar karşısında meşhur `z_H1` entropi testini emekliye ayırdık; 3 geçerli ayırt ediciyi (`Kober MI +53.7σ`, `Görülmemiş trigram +6.38 puan`, `207:1 son-ek yığılması`) kanıtladık.
> ✅ **700 Yıllık Paleografik Evrim:** Mohenjo-daro ve 600 km ötedeki Harappa arasında çapraz doğrulamalı iki global BH-FDR glif evrim zinciri (`#156 → #154` ve `#838 → #803 → #806`).
> ✅ **690 Gerçek Kırık Yazıt Onarımı:** 12 şehri tamamen dışarıda bırakan (LOCO) görsel glif + nöral MLP hibrit motoruyla `%57.5` Top-1 / `%82.9` Top-5 onarım kataloğu.
> ✅ **Dürüst Bilimsel Sınır:** Tek bir uydurma sesli okuma yok; 23-mühürlü Bilinen-Cevap Max-CSP testiyle `k ≥ 2` iki-dilli çapa gereksinimini matematiksel olarak gösterdik.
>
> Canlı portal (TR/EN): https://indus.emare.dev
> Açık kod & hakem iddia defteri: https://github.com/emraresoftware/indusemare
> 11 sayfalık PDF makale: https://indus.emare.dev/assets/INDUS_DEDUCT_PREPRINT.pdf

---

## 5. Doğrudan Akademisyen & Kurul E-posta Şablonu (English)

**Suggested Recipients:**
- **Prof. Rajesh P. N. Rao** (Univ. of Washington — *Science* 2009 entropy/Markov benchmark)
- **Prof. Richard Sproat** (Google / Oregon Health & Science Univ. — non-linguistic control critique)
- **Dr. Bryan K. Wells & Dr. Andreas Fuls** (ICIT Corpus creators — direct upstream citation & paleography)
- **Prof. Massimo Vidale** (Univ. of Padova — specifically mention our blind convergence on Gulf sign `#091` Twins & `#090` Man!)
- **Prof. Asko Parpola** (Univ. of Helsinki — Copper Tablets `TAB:C`, `: X :` modifiers, Proto-Dravidian & honest Mode 1/2B bounds)
- **Tamil Nadu State Dept. of Archaeology** (Benchmark & falsification referee tool for incoming $1M decipherment submissions)

**Subject:** `INDUS-DEDUCT v1.0: Open Calibrated Benchmark for the Indus Script (5-Way Calibration, 700-yr Paleographic Drift & 690 LOCO Restorations)`

**Body:**
> Dear Prof. [Name],
>
> We have released **INDUS-DEDUCT (v1.0.1; Zenodo DOI: `10.5281/zenodo.24383348`)**, a fully reproducible, anti-hallucination computational epigraphy benchmark over 5,679 Indus inscriptions (derived from ICIT/CISI with all speculative upstream columns stripped), paired with an 11/11 automated claim-ledger test suite and interactive portal:
>
> - **Interactive Portal (EN):** https://indus.emare.dev/?lang=en
> - **Permanent Zenodo Archive (DOI):** https://doi.org/10.5281/zenodo.24383348
> - **11-Page Preprint PDF:** https://indus.emare.dev/assets/INDUS_DEDUCT_PREPRINT.pdf
> - **1-Page Claim Ledger (Tier A Proven / Tier B Leads / Tier C Retired):** https://github.com/emraresoftware/indusemare/blob/main/CLAIM_LEDGER.md
> - **Code & 690-Inscription Restoration Catalog:** https://github.com/emraresoftware/indusemare
>
> We claim **zero** phonetic decipherments. Instead, our core methodological contributions directly engage with your work:
> 1. **5-Way Matched Calibration** against real Ur III Sumerian (CDLI), Linear B, Linear A, and structured Non-Linguistic Stamps — retiring `z_H1` (`-245.3σ` on non-linguistic stamps) while validating three discriminators (`Kober MI +53.7σ`, unseen-trigram generalization `+6.38 pp`, `207:1` suffix stacking) plus **23-seal Known-Answer Max-CSP** proving `k ≥ 2` bilingual anchors are required under realistic 74% open-vocab attestation.
> 2. **De-confounded 700-Year Paleographic Drift** across Mohenjo-daro (`n=1,253` within-area; `DKG(S)` `n=595`) replicated in Harappa `Period 2–5` (`rho = +0.289, p = 0.0376`), identifying `#156 → #154` (`q = 0.0083`) and `#838 → #803 → #806` (`q = 0.0339`).
> 3. *(If writing to Prof. Vidale)* Independent blind BH-FDR convergence on Twin-Humans **`#091`** (`0%` final, `42.35×` in the foreign export corridor `q = 2.86e-4`; `0/12` on foreign square seals vs. `11/17` `#090/#091` on circular Gulf seals).
>
> We would deeply appreciate any methodological critique or corrections before formal journal submission.
>
> Best regards,  
> Emre & the INDUS-DEDUCT Team (`indus.emare.dev`)

---

## 6. Zenodo DOI Arşivi (TAMAMLANDI ✓)
- **Kalıcı Sürüm DOI (`v1.0.1`):** https://doi.org/10.5281/zenodo.24383348 (`https://zenodo.org/records/24383348`)
- **Konsept DOI (her zaman en güncel sürümü gösterir):** https://doi.org/10.5281/zenodo.24383347
- `README.md`, `CITATION.cff`, `indus.emare.dev` ve yukarıdaki tüm yayın metinlerine işlendi!
