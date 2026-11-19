# INDUS-DEDUCT Layer 9: İç Rosetta (`TAB:C`), Harappa Fişleri, Çok-İmzalı Kargo ve Meluhha Dış Ticaret Malları

## 1. Mohenjo-daro Bakır Tabletleri (`TAB:C`, 229/230 = %99.6) — İç Rosetta
- **Metin ↔ İkon Karşılıklı Bilgi (MI) Testi (1.000 permütasyon):**
  - `TAB:C` (tümü, n=136): `MI = 2.8209 bit` (**z = +18.39**, `p = 0.001`, ağırlıklı saflık = **%90.6** vs null `%45.4`)
  - `TAB:C` (sadece adlandırılmış hayvan/ikonlar, `Othr` hariç, n=79): `MI = 3.4348 bit` (**z = +19.66**, saflık = **%96.2**)
  - **Kontrol — Kare Mühürler (`SEAL:S`, n=1622):** `z = +2.79` (ağırlıklı saflık = `%54.4` vs null `%47.6` — yani kare mühürde metin hayvanın adı değil!).
- **Çift Yüzlü Bakır Tabletler (`23` tablet):** Arka yüzdeki hayvan figürünün yerine **tek bir İndus işareti** konmuş (`753 ↔ Tavşan`, `777/780/781/782 ↔ Boynuzlu Avcı & Keçi/Manda`, `341 ↔ 503 615 740`, `957 ↔ 798 231 705 033 904 100 740`).

## 2. Harappa Çift Yüzlü Fiş Sistemi (`TAB:I` + `TAB:B`, 772 çift/üç yüzlü tablet)
- `TAB:I` %96.7, `TAB:B` %84.7 Harappa merkezli; çok yüzlü tabletlerin %60.5–66.2'sinde bir yüzde **`[sayı + 700]`** (`032/033/034 + 700`), diğer yüzde kurum/isim + `400` damga eki var.
- **Kurum ↔ Miktar Kilit Testi:** `MI z = +9.55` (`p = 0.001`, ağırlıklı saflık `%77.6` vs null `%57.7`). Örnek: `003 861 || 003 156` üç-yüzlüsü **20/20 kez** hep `034 700` (4 ölçek) alıyor.
- **`: X : + 400` Ek Yığılması:** `20/21` (`%95.2`, `p = 1.157e-08`) — farklı dil değil, aynı gramerin sonuna Harappa `400` fiş eki eklenmiş.

## 3. Çok İmzalı Kargo Paketleri (`TAG*`) & Şehirlerarası Lojistik
- Toplam `26` kil kargo topağında ≥2 farklı mühür aynı anda basılmış (`{'Kalibangan': 3, 'Lothal': 12, 'Harappa': 3, 'Mohenjo-daro': 1, 'Rupar': 1, 'Dholavira': 6}`).
- **Lothal Ana Gümrük Karşı-İmzası (`705 500 741 001 055 220 740 090`):** `4` farklı kargo paketinde (`2112, 2113, 2115, 2117`) ayrı tüccar mühürleriyle ortak imza atıyor — bunlardan `#2113` paketindeki karşı imza (`861 002 235`), **700 km ötede Mohenjo-daro'da kazılan gerçek taş mühür (`2239.1`)**!
- Toplam `9` adet `len>=3` kargo mühür baskısı başka şehirdeki gerçek taş mühürle birebir eşleşiyor.

## 4. Meluhha Dış Ticaret Malları & İhracat Koridoru (Tekrarsız Fisher + Global BH-FDR)
- Global BH-FDR (`q < 0.05`) geçen tekrarsız işaret sayısı: **`14/327`**.
- **Ivory_Objects** (`n_uniq=32`, `raw=36`): FDR<0.05 geçen = **1** | yakalanan kopya-damga yanılsaması = `1`
  - İşaret **`400`**: `9/32` (`3.37x`, `p=1.02e-03`, `q=3.22e-02`)
- **Stoneware_Shell_Bangles** (`n_uniq=38`, `raw=50`): FDR<0.05 geçen = **2** | yakalanan kopya-damga yanılsaması = `1`
  - İşaret **`064`**: `3/38` (`23.04x`, `p=4.92e-04`, `q=3.22e-02`)
  - İşaret **`790`**: `4/38` (`9.13x`, `p=1.18e-03`, `q=3.22e-02`)
- **Copper_Gold_Implements** (`n_uniq=18`, `raw=23`): FDR<0.05 geçen = **3** | yakalanan kopya-damga yanılsaması = `6`
  - İşaret **`006`**: `2/18` (`89.47x`, `p=4.32e-04`, `q=3.22e-02`)
  - İşaret **`930`**: `2/18` (`71.58x`, `p=6.03e-04`, `q=3.22e-02`)
  - İşaret **`005`**: `3/18` (`14.91x`, `p=1.16e-03`, `q=3.22e-02`)
- **Cargo_Clay_Sealings_vs_Seals** (`n_uniq=171`, `raw=242`): FDR<0.05 geçen = **2** | yakalanan kopya-damga yanılsaması = `6`
  - İşaret **`326`**: `3/171` (`999.0x`, `p=9.09e-04`, `q=3.22e-02`)
  - İşaret **`700`**: `7/171` (`5.9x`, `p=9.26e-04`, `q=3.22e-02`)
- **Lothal_Port_vs_Core** (`n_uniq=146`, `raw=179`): FDR<0.05 geçen = **0** | yakalanan kopya-damga yanılsaması = `1`
- **Dholavira_Port_vs_Core** (`n_uniq=171`, `raw=202`): FDR<0.05 geçen = **0** | yakalanan kopya-damga yanılsaması = `0`
- **Chanhudaro_BeadHub_vs_Core** (`n_uniq=57`, `raw=58`): FDR<0.05 geçen = **3** | yakalanan kopya-damga yanılsaması = `0`
  - İşaret **`031`**: `13/57` (`3.83x`, `p=4.09e-05`, `q=6.69e-03`)
  - İşaret **`032`**: `15/57` (`2.6x`, `p=5.74e-04`, `q=3.22e-02`)
  - İşaret **`128`**: `2/57` (`999.0x`, `p=1.12e-03`, `q=3.22e-02`)
- **Export_Corridor_Foreign_vs_CoreSeals** (`n_uniq=39`, `raw=39`): FDR<0.05 geçen = **3** | yakalanan kopya-damga yanılsaması = `0`
  - İşaret **`091`**: `5/39` (`42.35x`, `p=8.74e-07`, `q=2.86e-04`)
  - İşaret **`004`**: `5/39` (`6.35x`, `p=1.43e-03`, `q=3.59e-02`)
  - İşaret **`090`**: `7/39` (`4.0x`, `p=1.93e-03`, `q=4.51e-02`)
