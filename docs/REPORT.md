# Skadi-Net: segmentazione del ghiaccio marino da SAR in banda C per l'elaborazione a bordo

*Relazione tecnica sul modello di intelligenza artificiale del progetto Skadi (Sezione 3 di `Skadi.pdf`).*

> **Stato del lavoro.** Questa relazione descrive in modo completo metodo, protocollo
> sperimentale e scelte di progetto. Il codice è stato verificato dall'inizio alla fine su
> scene sintetiche, ma **l'addestramento sulle scene reali AI4Arctic non è ancora stato
> eseguito**: la Sezione 6 non riporta quindi alcun numero di accuratezza. Le tabelle e le
> figure dei risultati vengono generate automaticamente in `docs/RESULTS.md` da
> `scripts/run_experiments.py`; finché quel file non esiste, non è possibile affermare che
> il modello superi TinyIceNet.

## Sommario

Skadi prevede di classificare il ghiaccio marino direttamente a bordo del satellite, a
partire da immagini SAR in banda C a doppia polarizzazione (HH, HV), e di trasmettere alle
navi una mappa vettoriale invece dell'immagine radar. Questo lavoro realizza la rete
descritta nel documento di missione: una U-Net con encoder MobileNetV3-Small, addestrata
con una loss che combina cross-entropy pesata e Dice, e quantizzata a 8 bit con
quantization-aware training. La rete ha 942 mila parametri, richiede 0,91 GMAC per una
scena 2×512×512 e occupa 1,27 MB una volta quantizzata. Il riferimento per il confronto è
TinyIceNet (Al Koutayni et al., 2026), che sullo stesso dataset e con gli stessi ingressi
dichiara un F1 del 75,2 % con 146,6 mila parametri e 2,97 GMAC. Il protocollo di
valutazione di TinyIceNet è stato replicato esattamente e la sua architettura è stata
reimplementata, in modo da poterla riaddestrare con la stessa procedura usata per la
nostra rete.

## 1. Introduzione

Il documento di missione (Sezione 3.2) definisce il compito come segmentazione semantica:
a ogni pixel dell'immagine SAR va assegnata una fra tre classi, acqua libera (OW), ghiaccio
del primo anno (FYI) e ghiaccio pluriennale (MYI). La distinzione ha una base fisica
(Sezione 2.2 del documento): l'acqua riflette specularmente e appare scura in entrambe le
polarizzazioni; il FYI, salino, produce scattering di superficie e un segnale medio in HH;
il MYI, desalinizzato e ricco di bolle d'aria, produce scattering di volume e un segnale
forte in HV. Per questo la rete riceve HH e HV come due canali separati.

I vincoli sono quelli di un acceleratore FPGA con meno di 30 W: pochi parametri, poche
operazioni e aritmetica a 8 bit. L'obiettivo di questo lavoro è ottenere, sotto questi
vincoli, un'accuratezza superiore a quella di TinyIceNet, che è il lavoro pubblicato più
vicino a Skadi per dati, ingressi e piattaforma di destinazione.

## 2. Lavori correlati

**AI4Arctic / AutoICE.** Il dataset AI4Arctic Sea Ice Challenge (Buus-Hinkler et al.,
2022) associa scene Sentinel-1 EW in HH/HV alle carte del ghiaccio dei servizi danese
(DMI) e canadese (CIS). La competizione AutoICE (Stokholm et al., 2024) ne ha fissato il
protocollo: 513 scene di addestramento, 20 di test, e per lo stadio di sviluppo (SOD) la
metrica F1 pesata sul numero di pixel di ogni classe.

**MMSeaIce** (Chen et al., 2023–2024), la soluzione vincitrice di AutoICE, usa una U-Net
multi-task con ingressi aggiuntivi (radiometro AMSR2, rianalisi ERA5, posizione, data).
Con i soli canali SAR il suo F1 sul SOD scende a circa 75,1 %, valore riportato da
TinyIceNet come riferimento.

**TinyIceNet** (Al Koutayni et al., 2026, arXiv:2603.03075) è progettata per l'inferenza a
bordo su FPGA Zynq UltraScale+, la stessa famiglia prevista da Skadi. Dal testo
dell'articolo:

| Aspetto | TinyIceNet |
|---|---|
| Dati | AI4Arctic *ready-to-train* (80 m), solo HH e HV |
| Ingresso | ogni scena ridimensionata a 512×512 (circa 800 m per pixel) |
| Architettura | 4 blocchi di due conv 3×3 con BN e ReLU (16, 32, 64, 64 filtri), 3 pooling, un solo upsampling ×8, conv 1×1; nessuna skip connection |
| Complessità | 146,6 k parametri, 2,97 GMAC |
| Addestramento | SGD, lr 0,001, cosine annealing, batch 32, 200 epoche × 500 iterazioni, sola cross-entropy; 10 scene di validazione |
| F1 sul test (20 scene, 512×512) | 75,168 % in FP32; 24,3 % con PTQ a 8 bit; 75,216 % con QAT a 8 bit sui pesi |
| FPGA ZCU102 | 7 fps, 113,6 mJ per scena |

L'articolo non precisa il tipo di media dell'F1; qui si assume la media pesata di AutoICE,
coerente con la frase «tiene conto dello sbilanciamento fra le classi» e con il confronto
diretto che gli autori fanno con MMSeaIce.

## 3. Dati e pre-elaborazione

**Sorgente.** Versione *ready-to-train* di AI4Arctic: 513 file netCDF di addestramento
(60,2 GB) e 20 scene di test con le relative carte di riferimento (2,6 GB). Ogni file
contiene HH (`nersc_sar_primary`) e HV (`nersc_sar_secondary`) già calibrati, corretti dal
rumore termico, portati a 80 m e standardizzati; le carte del ghiaccio sono rasterizzate
sulla stessa griglia. La variabile `SOD` ha sei classi (0 acqua libera, 1 ghiaccio nuovo,
2 ghiaccio giovane, 3 FYI sottile, 4 FYI spesso, 5 ghiaccio vecchio) e il valore 255 per
terra, dati mancanti e poligoni ambigui.

**Cache 512×512** (`skadi/preprocess.py`). Per rendere il confronto con TinyIceNet diretto,
ogni scena viene ridimensionata a 512×512:

* HH e HV con media d'area, ignorando i pixel NaN (si divide per la frazione di pixel
  validi invece di trattare i NaN come zeri, che scurirebbe i bordi);
* SOD con il vicino più prossimo, così le etichette restano quelle della carta;
* si salvano anche angolo di incidenza, latitudine, longitudine (come seno e coseno, per
  evitare il salto a ±180°) e mese di acquisizione, usati solo dalla variante «+geo».

L'intero insieme di addestramento occupa circa 1 GB e viene tenuto in memoria: ogni batch
è una semplice indicizzazione di tensori, senza processi di caricamento (lenti da avviare
su Windows).

**Suddivisione.** Trenta scene di addestramento, estratte una volta per tutte con seme
fisso, formano l'insieme di validazione e servono a scegliere l'epoca migliore. Le 20
scene di test non vengono mai usate per decisioni di progetto.

## 4. Metodo

### 4.1 Architettura

```
ingresso 2×512×512 (HH, HV)
  encoder MobileNetV3-Small:  16 @ 1/2 → 16 @ 1/4 → 24 @ 1/8 → 48 @ 1/16 → 96 @ 1/32
  ponte:    conv separabile 96 → 128 @ 1/32
  decoder:  [upsampling ×2 + concatenazione con l'encoder + 2 conv separabili] × 4
            96 @ 1/16 → 64 @ 1/8 → 48 @ 1/4 → 48 @ 1/2
  testa:    dropout, conv 1×1 → 6 classi @ 1/2, upsampling bilineare ×2
uscita 6×512×512 → probabilità OW / FYI / MYI per somma
```

*Encoder.* MobileNetV3-Small di torchvision, troncato dopo l'ultimo blocco residuo (si
scarta la conv 96→576, pensata per la classificazione). Le convoluzioni depthwise
separabili sono ciò che tiene basso il numero di operazioni. I pesi iniziali sono quelli
di ImageNet; la prima convoluzione, nata per tre canali RGB, viene adattata a due canali
mediando i filtri sui tre colori e riscalando per 3/2, così l'ampiezza delle attivazioni
resta la stessa.

*Decoder.* Quattro stadi U-Net. Ogni stadio raddoppia la risoluzione con interpolazione
bilineare, concatena la mappa dell'encoder alla stessa scala e applica due convoluzioni
depthwise separabili con BatchNorm e ReLU. Nel decoder si usa ReLU e non Hardswish perché
è più semplice da quantizzare e da realizzare in logica programmabile.

*Testa.* Il classificatore lavora a metà risoluzione e i logit vengono interpolati ×2. Le
carte del ghiaccio sono poligoni ampi: uno stadio a piena risoluzione sarebbe il più
costoso dell'intera rete e non aggiungerebbe informazione utile.

*Perché skip connection e contesto.* TinyIceNet rinuncia alle skip connection e riduce la
risoluzione solo di 8 volte, quindi ogni pixel «vede» qualche decina di chilometri. Il
nostro encoder scende a 1/32: sullo stadio più profondo il campo ricettivo copre l'intera
scena, cosa utile perché lo stesso valore di retrodiffusione può appartenere a classi
diverse (acqua increspata dal vento e ghiaccio giovane, ad esempio) e solo il contesto le
distingue. Le skip connection riportano poi i bordi a 1/2.

| | Skadi U-Net | TinyIceNet |
|---|---:|---:|
| Parametri | 942 398 | 146 599 (7 uscite; 146,6 k nell'articolo) |
| GMAC, ingresso 2×512×512 | 0,91 | 2,91 nel nostro conteggio; 2,97 nell'articolo |
| Dimensione INT8 | 1,27 MB | 0,17 MB |

I MAC sono contati sulle sole convoluzioni (`count_macs` in `skadi/models.py`). La nostra
rete ha 6,4 volte i parametri di TinyIceNet ma richiede 3,2 volte meno operazioni, perché
lavora quasi sempre a bassa risoluzione, mentre TinyIceNet esegue convoluzioni 3×3 piene a
512×512 e 256×256. Per un acceleratore il costo dominante sono le operazioni; 1,27 MB di
pesi stanno comunque nella memoria on-chip. Il documento di missione stimava circa
5 milioni di parametri: il valore reale è cinque volte più basso.

*Sei classi invece di tre.* La rete predice le sei classi SOD e la mappa Skadi si ottiene
sommando le probabilità: OW = classe 0, FYI = classi 1–4, MYI = classe 5. In questo modo
(a) il risultato è confrontabile con la letteratura, che valuta sulle sei classi, (b) la
rete riceve una supervisione più ricca, e (c) il costo in inferenza è nullo.

### 4.2 Funzione di costo

Come previsto dal documento (Sezione 3.3):

L = CE pesata + 0,5 · Dice

* **Cross-entropy pesata.** Il peso della classe *c* è (1/f_c)^0,5, con f_c frazione di
  pixel di addestramento della classe, normalizzato a media 1 e limitato a 4. L'esponente
  0,5 attenua il bilanciamento: con il peso inverso pieno le classi rare dominerebbero e
  la precisione sull'acqua libera calerebbe.
* **Peso extra sul MYI.** Il peso del ghiaccio vecchio è moltiplicato per 1,5: non vedere
  ghiaccio pluriennale è l'errore più pericoloso per una nave, un falso allarme costa solo
  una deviazione.
* **Dice.** Calcolato sull'intero batch (non per immagine) e solo sulle classi presenti,
  così una classe assente da una scena non produce un termine instabile.
* I pixel con etichetta 255 sono esclusi da entrambi i termini.

### 4.3 Data augmentation

Tutta l'augmentation è eseguita a batch sulla GPU (`skadi/data.py`, classe `Augment`).

*Radiometrica, solo su HH e HV:*

* **scalatura dipendente dall'angolo di incidenza**: una rampa lineare lungo il range con
  pendenza casuale, applicata a HH e dimezzata su HV, perché la retrodiffusione del
  ghiaccio dipende dall'angolo più in co-polarizzazione che in cross-polarizzazione
  (Aldenhoff et al., 2020);
* **offset di guadagno** casuale per canale, che simula differenze di calibrazione;
* **iniezione di speckle**: rumore moltiplicativo con distribuzione Gamma e numero di look
  casuale fra 8 e 64, convertito in dB e sommato ai canali (che sono in dB standardizzati).

*Geometrica, su immagini ed etichette insieme:* riflessione, rotazione libera,
ingrandimento fra 0,7 e 1,4, anisotropia fino al 15 % e traslazione. L'anisotropia è
voluta: ridimensionare ogni scena a 512×512 la deforma in modo diverso a seconda delle sue
proporzioni, e la rete non deve dipendere da una deformazione particolare. Le etichette
sono ricampionate con il vicino più prossimo e le zone che escono dalla scena diventano
255.

### 4.4 Addestramento

| Iperparametro | Valore |
|---|---|
| Ottimizzatore | AdamW, lr 1·10⁻³, weight decay 1·10⁻⁴ |
| Schedulazione | warm-up lineare (3 %), poi coseno fino all'1 % del lr |
| Budget | 150 epoche × 200 iterazioni, batch 16 (480 mila scene viste) |
| Precisione | mista (AMP) su GPU |
| Regolarizzazione | dropout 0,1 sulla testa, clipping del gradiente a 5 |
| Pesi valutati | media mobile esponenziale (EMA, 0,999) |
| Selezione | epoca con il miglior F1 di validazione |

### 4.5 Quantizzazione a 8 bit

`skadi/quantize.py` usa la quantizzazione FX di PyTorch con pesi **e attivazioni** a 8 bit
(pesi simmetrici per canale, attivazioni affini). Produce tre modelli e li valuta sulle
stesse scene:

1. **FP32**, il modello di partenza;
2. **PTQ**: si osservano le attivazioni su 64 scene di addestramento e si converte, senza
   riaddestrare;
3. **QAT**: 15 epoche di fine-tuning con la quantizzazione simulata nel grafo (lr 2·10⁻⁴);
   a metà si congelano le statistiche di BatchNorm e nell'ultimo quarto gli intervalli di
   quantizzazione; si tiene l'epoca migliore in validazione e la si converte in un vero
   modello intero, valutato su CPU ed esportato in TorchScript.

Rispetto a TinyIceNet, che quantizza i pesi, la nostra impostazione è più severa perché
quantizza anche le attivazioni.

### 4.6 Dal raster al prodotto

`skadi/infer.py` realizza la parte software dei punti 3 e 5 della pipeline di bordo e
della Sezione 3.5 del documento:

* **inferenza a tasselli**: scene più grandi di 512×512 sono divise in tasselli
  sovrapposti, fusi con una finestra a coseno rialzato per evitare discontinuità ai bordi;
* **confidenza**: 1 − entropia normalizzata della distribuzione OW/FYI/MYI, mediata su ogni
  poligono; sotto 0,5 il poligono è marcato come incerto;
* **fuori distribuzione**: massimo z-score di media e deviazione standard di HH e HV della
  scena rispetto alle scene di addestramento; sopra 4 la scena è segnalata;
* **vettorializzazione**: contorni chiusi (con i buchi), semplificati con Douglas-Peucker
  ed esportati in GeoJSON.

## 5. Protocollo sperimentale

**Metrica principale.** F1 pesato sulle sei classi SOD, su tutti i pixel etichettati delle
20 scene di test presi insieme, calcolato sulle scene ridimensionate a 512×512: è la
quantità riportata da TinyIceNet. L'implementazione è verificata contro scikit-learn
(`tests/test_core.py`).

**Metriche secondarie.** F1 macro, F1 per classe, matrice di confusione, F1 per scena, F1
sulle tre classi Skadi, e F1 alla risoluzione nativa di 80 m (le predizioni vengono
riportate sulla griglia originale e confrontate con la carta piena, come in AutoICE).

**Riferimenti.**

1. *TinyIceNet come da articolo*: 75,168 % (FP32) e 75,216 % (QAT a 8 bit).
2. *TinyIceNet riaddestrata da noi* (`tinyicenet_repro`): stessa architettura, verificata
   contando i parametri (146 599, contro i 146,6 k dichiarati), addestrata con la nostra
   identica procedura. Serve a separare l'effetto dell'architettura da quello della
   ricetta di addestramento: senza questo controllo, una differenza rispetto al numero
   dell'articolo potrebbe dipendere solo da pre-elaborazione o augmentation diverse.

**Esperimenti.**

| Nome | Descrizione |
|---|---|
| `skadi_sar` | modello principale, ingressi HH e HV |
| `tinyicenet_repro` | architettura TinyIceNet, stessa procedura |
| `skadi_geo` | HH, HV + angolo di incidenza, posizione, mese |
| `abl_scratch` | senza pesi ImageNet |
| `abl_noaug` | senza augmentation |
| `abl_ce_only` | sola cross-entropy non pesata (la loss di TinyIceNet) |
| `abl_ia` | HH, HV + angolo di incidenza |

`skadi_geo` usa informazioni che il satellite possiede comunque (geometria di
acquisizione, GNSS, orologio), senza altri sensori; non è però un risultato «solo SAR» e
va confrontato con TinyIceNet tenendone conto.

**Criterio.** Il modello è considerato migliore di TinyIceNet se `skadi_sar`, con soli HH e
HV e quantizzato a INT8, supera 75,216 % di F1 sul test **e** supera `tinyicenet_repro`.

## 6. Risultati

**Non ancora disponibili.** L'addestramento richiede le scene AI4Arctic e una GPU.
L'ambiente in cui il codice è stato sviluppato non aveva accesso né all'una né alle altre:
il server che distribuisce il dataset risponde su una porta non raggiungibile da lì e la
macchina disponeva di sole CPU.

Dopo l'esecuzione di `python scripts/run_experiments.py`, `docs/RESULTS.md` conterrà:

* la tabella dei modelli in FP32 (parametri, GMAC, F1, F1 con TTA, F1 a 80 m, F1 macro, F1 OW/FYI/MYI);
* la tabella della quantizzazione (FP32, PTQ, QAT, dimensione);
* i punteggi per classe e per scena del modello principale;
* quattro figure: curve di validazione, F1 per classe, matrice di confusione, accuratezza contro costo computazionale.

Ciò che è stato misurato finora non dipende dai dati reali:

| Grandezza | Valore | Come è stata ottenuta |
|---|---:|---|
| Parametri Skadi U-Net | 942 398 | conteggio diretto |
| GMAC Skadi U-Net | 0,91 | hook sulle convoluzioni, ingresso 2×512×512 |
| Parametri TinyIceNet reimplementata | 146 599 | conteggio diretto, 7 uscite |
| GMAC TinyIceNet reimplementata | 2,91 | come sopra |
| Dimensione Skadi U-Net INT8 | 1,27 MB | `state_dict` del modello convertito |
| Dimensione Skadi U-Net FP32 | 3,94 MB | `state_dict` |

**Verifica del codice.** L'intera catena (cache, addestramento, valutazione, PTQ, QAT,
inferenza, generazione del report) è stata eseguita su scene sintetiche con lo stesso
formato dei file reali. Queste scene hanno statistiche inventate: i numeri ottenuti su di
esse non dicono nulla sul ghiaccio reale e non sono riportati.

## 7. Discussione e limiti

* **Il risultato principale manca.** Finché `skadi_sar` non è stato addestrato e valutato,
  la superiorità rispetto a TinyIceNet è un'ipotesi di progetto, motivata dal maggiore
  contesto spaziale, dalle skip connection, dal pre-addestramento e dalla loss, non un
  fatto misurato.
* **Tetto dei soli dati SAR.** TinyIceNet e MMSeaIce (solo SAR) si fermano entrambe intorno
  al 75 %. È possibile che il limite sia l'informazione contenuta in HH e HV a 800 m più
  che l'architettura; in tal caso il guadagno di `skadi_sar` sarebbe piccolo e la strada
  più promettente diventerebbe `skadi_geo`.
* **Iperparametri non ottimizzati.** I valori della Sezione 4.4 sono scelte ragionevoli
  fatte prima di vedere i dati reali.
* **Protocollo a 512×512.** Ridimensionare ogni scena deforma le proporzioni e porta il
  pixel a circa 800 m, lontano dai 20–25 m dello ScanSAR di Skadi. È il protocollo del
  lavoro di riferimento e le carte del ghiaccio non hanno comunque dettaglio più fine; per
  l'uso operativo andrà valutato un addestramento a risoluzione maggiore.
* **Etichette.** Le carte sono disegnate a mano per poligoni e assegnano a ogni poligono
  il tipo dominante: una parte dell'errore misurato è rumore delle etichette.
* **Dominio.** Sentinel-1 EW è simile ma non identico al sensore di Skadi (risoluzione,
  rumore, geometria); l'augmentation radiometrica riduce questo divario, non lo elimina.
* **Parti non realizzate rispetto al documento.** Carte NIC come dati aggiuntivi, filtro
  di Lee, rivelatore CFAR, sintesi su FPGA con misure di latenza e consumo.

## 8. Riproducibilità

Tutto il codice è nella cartella `skadi/`; ogni numero di `docs/RESULTS.md` è letto dai
file JSON scritti da `skadi.evaluate` e `skadi.quantize`. I comandi sono nel `README.md`.
Semi casuali, suddivisione di validazione e configurazione di ogni esperimento sono
salvati in `results/<nome>/config.json`.

## Riferimenti

1. M. R. Al Koutayni, M. Selim, G. Reis, A. Pagani, D. Stricker. *TinyIceNet: Low-Power SAR Sea Ice Segmentation for On-Board FPGA Inference.* arXiv:2603.03075, 2026.
2. J. Buus-Hinkler et al. *AI4Arctic Sea Ice Challenge Dataset.* DTU Data, 2022.
3. A. Stokholm et al. *The AutoICE Challenge.* The Cryosphere, 2024.
4. X. Chen et al. *MMSeaIce: a collection of techniques for improving sea ice mapping with a multi-task model.* The Cryosphere, 2024.
5. A. Howard et al. *Searching for MobileNetV3.* ICCV, 2019.
6. O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
7. W. Aldenhoff, L. E. B. Eriksson, Y. Ye, C. Heuzé. *First-Year and Multiyear Sea Ice Incidence Angle Normalization of Dual-Polarized Sentinel-1 SAR Images in the Beaufort Sea.* IEEE JSTARS, 2020.
8. F. Milletari, N. Navab, S.-A. Ahmadi. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV, 2016.
