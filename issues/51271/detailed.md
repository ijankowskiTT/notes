# Notatki o defektach Quasar - #51271 / #56226

<details>
<summary><h2>Item 3 - dest stride na unaligned same-width reshard</h2></summary>

- branch: `51271-item3-56226-quasar-reshard`, commit `1d9d50df3f3`
- na tym samym shape jest drugi, niezależny bug: #56226

### Problem

- Op zbiera HEIGHT_SHARDED tensor z 4 cores na 2, kopiując row po row. Shape: `(1,1,128,3)` bf16, row-major.
- Każdy row ma tylko 3 elementy bf16, czyli 6 B prawdziwych danych (`unit_size` = packed).
- W L1 ten sam row i tak zajmuje 16 B (`local_unit_size_padded`), bo layout jest alignowany.
- Po skopiowaniu kernel przesuwał dest pointer o packed 6 B zamiast o padded 16 B.
- Kolejny row zaczynał się w środku poprzedniego - dane się nakładały.
- Call wracał bez błędu, output był po prostu zły (silent corruption). Porównanie z torch to łapie; sam „op się skończył” nie.
- Host już liczył te 16 B, ale do kernela podawał tylko `remote_unit_size_padded`. Local dest stride kernel nie znał.
- Na Quasar NOC może czytać od dowolnego bajtu (align 1), ale L1 layout nadal ma 16 B. Factory bierze `buffer()->alignment()` = L1, więc `unaligned` i `+= 6` zostają także tam.

### Podejścia

- **Wybrane:** factory podaje kernelowi `local_unit_size_padded` jako CTA, a kernel po każdym row przesuwa dest o te 16 B. NOC dalej kopiuje tylko packed 6 B. To ten sam pattern co mainline po #53028.
> **CTA (compile-time argument)** - liczba, którą host wkleja w kernel przy JIT, nie przy każdym enqueue. Kernel czyta ją przez `constexpr get_arg`, więc **musi** być CompileTimeArg. `local_unit_size_padded` jest stałe dla tego programu (tu 16), stąd CTA. RTA (runtime argument) to wartości per launch, np. ile readów; w `constexpr get_arg` nie wejdą. Dwa RISC-e na tym samym `.cpp` mogą dostać **różne** CTA (patrz `is_reader` w #56226).
- Nie da się „upchnąć” rows w L1 bez padu. Unpacker i layout L1 wymagają 16 B - tego alignmentu nie ruszamy.
- Nie warto też wysyłać 16 B przez NOC „żeby stride i transfer były równe”. Dest i tak musi skoczyć o 16; bajty padu to śmieci, nie payload.
- Nie skipujemy UNALIGNED na Quasar tylko dlatego, że NOC align = 1. Flaga `unaligned` liczy się z L1 16, nie z NOC. Kernel i tak by robił `+= 6`.
- Host musi liczyć `local_start_offset` tym samym padded stride’em. Inaczej dwa RISC-e dostaną nakładające się dest regiony, nawet gdy każdy z osobna hopa dobrze.

### Rozwiązanie

- Factory wstawia `local_unit_size_padded` do compile-time args kernela. Po każdym skopiowanym row dest pointer skacze o 16 B (początek następnego row w L1).
- NOC nadal przenosi tylko 6 B prawdziwych danych. Padu nie wysyłamy.
- Na hoście `local_start_offset` też idzie padded stride’em, żeby drugi RISC nie zaczął w połowie regionu pierwszego.
- Zmiana w `reshard_same_width_reader.cpp` i `reshard_program_factory_same_width.cpp`.


### Wynik

- Sam ten commit nie naprawia testu: nadal **fail**, bo dwa RISC-e dalej piszą w ten sam scratch (#56226).
- Repro: `test_quasar_reshard_same_width_row_major_local_stride`.

## #56226 - dwa RISC-e, jeden scratch

- twin #51217; mainline fix #53028 (`is_reader` w fold)
- commit `92ced891cef`

### Problem

- Jeden DM core ma **dwa** niezależne RISC-e. Przy tym reshardzie oba dostają ten sam reader kernel.
- Output shard ma 64 rows, podzielone 32 + 32 między te dwa RISC-e.
- Unaligned path najpierw stage’uje remote rows do wspólnego scratch DFB, dopiero potem copy na dest.
- Oba RISC-e piszą pod `scratch + src_offset`. `src_offset` wraca do 0 na każdym nowym remote core, więc drugi nadpisuje pierwszego (clobber).
- Na dest lecą już śmieci ze scratcha - nawet gdy dest stride z item 3 jest poprawne 16 B.
- Na `(1,1,128,3)` widać miss m.in. na `[0,0,31,2]`.
- Quasar Tensix ma te same dwa RISC-e; to nie jest quirk firmware WH.

### Podejścia

- **Wybrane:** podwoić scratch (`num_entries * 2`) i rozdzielić połowy flagą CTA `is_reader`. Tak robi fold / #53028. Nie dokładamy drugiego DFB.
- Osobny DFB na każdy RISC też by rozdzielił zapisy, ale zjada więcej L1 i zmienia ProgramSpec. Nie ten twin.
- Zostawienie tylko jednego RISC-a na unaligned path uniknęłoby race, ale jest wolniejsze i większe niż mainline.
- Serializacja obu RISC-ów na tym samym scratch (jeden czeka, drugi pisze) jest hang-prone i nie ten pattern.
- Sam offset bez podwojenia `num_entries` wychodzi poza scratch (OOB) - drugi RISC pisze w pamięć, której bufor nie ma.

### Rozwiązanie

- Scratch ma `num_entries = 2 * remote_units_per_shard`: miejsce na obie połowy.
- Każdy RISC dostaje CTA `is_reader` = 1 albo 0. Drugi startuje od `remote_units_per_shard * remote_unit_size_padded`, więc pisze obok pierwszego, nie w niego.
- W tym samym commicie są extras, których nie ma w issue (są w #53028). Bez nich unaligned path w ogóle nie wstaje:
  - `#define UNALIGNED` leci zawsze gdy packed != padded, nie tylko gdy `use_scratch`.
  - Writer kernel (`reshard_same_width_writer.cpp`) też hopa padded, gdy local jest input. Nasz test to gather into output, więc oba sloty kompilują **reader** i ta ścieżka nie leci.
  - Borrowed shard DFB: `entry_size = local_unit_size_padded`, a `num_entries` jest clampowane do packed bytes z `TensorSpec`, bo spec-time check nie widzi jeszcze `Buffer`. Gdy packed < jeden padded row, `entry_size` schodzi do packed i `num_entries = 1` (Copilot; patrz review).

> **CTA `is_reader`** - oba RISC-e kompilują ten sam `.cpp`, więc przy JIT dostają różne compile-time args: 1 vs 0. Z tego kernel liczy, która połowa scratcha jest jego. To musi być CTA, bo offset idzie w `constexpr get_arg`; RTA by się tam nie załapało.

### Wynik

- Sam scratch-fix: test nadal **fail**, bo dest hop zostaje 6 B (item 3).
- Sam dest-stride: test nadal **fail**, bo RISC-e dalej clobberują scratch.
- Oba commity razem: **PASS**.
- 2026-09-14 po rebuild: ten test plus 4 stare slice case’y w `test_tm_ops.py` - wszystkie PASSED.

## Review PR #56450 - Copilot

### 1. Writer path bez testu

Copilot: pytest zawsze robi L1 out, więc nowy branch `UNALIGNED` w `reshard_same_width_writer.cpp` nigdy nie leci. Sugeruje ten sam shape z DRAM out (albo parametr L1/DRAM), żeby złapać writer.

To prawda: ten test nie wykonuje writera. Ale ten PR naprawia L1 gather - dest stride (item 3) i dwa RISC-e na jednym scratchu (#56226). DRAM-out nie alokuje scratcha, więc **nie pokryłby tych bugów**. Writer hops są extras z parity z mainline; PR już pisze „writer is not covered”.

Nie blokujemy merge. DRAM case to follow-up, nie ten ticket. Copilot powtórzył to samo po commicie DFB (*0 new comments*) - ta sama luka, nie nowy finding.

### 2. Zero-entry borrowed DFB

Borrowed DFB to nie prawdziwy bufor - kernel bierze z niego tylko bazowy adres (`get_write_ptr()`). Host i tak musi go opisać (`entry_size` × `num_entries`). ProgramSpec sprawdza to względem **packed** bajtów tensora (ile danych naprawdę jest), zanim powstanie Buffer, i **odrzuca `num_entries == 0`**.

Commit 2 ustawił `entry_size` na padded 16 B (jak L1 layout) i `num_entries = min(padded shard, packed) / 16`. Na repro `(1,1,128,3)` packed = 768 → 32, OK. Na węższym `(1,1,1,3)` packed = 6 < 16 → dzielenie daje **0** → fatal przy walidacji programu, zanim kernel wstanie. Stary opis (`entry_size = 6`, `num_entries = 1`) był git. `num_entries = 1` przy `entry_size = 16` też pada, bo 16 > 6.

To dziura w extras DFB, nie w dest stride i nie w scratch. Mainline ma to samo. Copilot ma rację - trzeba było zrobić. Shrink `entry_size` do packed, `num_entries = 1`. Kernel nadal hopa 16 B; DFB tylko daje adres. Probe na WH PASSED. Tego tiny case’a nie dodajemy do PR (nie ten ticket).
</details>


<details>
<summary><h2>Item 5 - tensor-args slice, self-loop DFB</h2></summary>

- branch: `51271-quasar-slice-tensor-args-dfb`, commit `735398ff08e`
- quasar-specific (nie ma twina w mainline `ttnn.slice`). Item 6 to inny bug w slice (width-stride), już na `main`.

### Problem

- `experimental.quasar.slice` ma dwa call-e na zakres. Listy w Pythonie (`slice(x, [0,0,…], [1,1,…])`) host już zna liczby i wpisuje je w argumenty kernela — ten program w ogóle nie czyta `start`/`end` z device i **nie kompiluje** `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp`. Bug jest tylko w drugim callu: `start`/`end` jako tensory na device (`slice(x, start_t, end_t, slice_dim=…, num_devices=…)`). TILE + step=1; host stawia `use_tensor_args=true` i buduje `SliceTileTensorArgsProgramFactory`.
- Reader musi raz ściągnąć te indeksy do L1, policzyć `start_offset` w tile'ach, i dopiero potem pchać payload do `c0` (normalny DFB: reader produkuje, writer konsumuje).
- Staging indeksów zrobili jako drugi DFB `c1` (`cb_tensor`) na tym samym DM readerze: PRODUCER i CONSUMER naraz. To jest **self-loop**.
- ProgramSpec na Gen2/Quasar takiego DFB nie obniża: credit machinery wymaga rozłącznych RISC-ów na producer i consumer. `TT_FATAL`: *Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures. Consider using a scratchpad.* Program w ogóle nie wstaje. Na WH (Gen1) self-loop jest git - DFB to zwykły CB - więc ten sam kod na tej skrzynce by przeszedł walidację.
- Ban w ProgramSpec (`c5d9695`, 2026-06-24) wszedł dwa dni po tym kernelu. RM sharded slice i I2S zdążyły zejść na scratchpad / TensorAccessor; tensor-args został starym handshake'iem `reserve → push → wait → pop`.

### Podejścia

- **Wybrane:** `c1` jako `ScratchpadSpec`, w kernelu `Scratchpad<uint32_t>`. NOC czyta start/end na ten region, `async_read_barrier()`, indeksy przez `[]`. `c0` bez zmian. Tak samo I2S / padded_slice / reshape; sam ProgramSpec podpowiada scratchpad.
> **Scratchpad vs DFB** - DFB to FIFO z kredytami: ktoś musi być PRODUCER, ktoś CONSUMER, na Gen2 na różnych RISC-ach. Scratchpad to surowy kawałek L1 na czas programu, bez handshake'u. Indeksy czytamy raz na starcie i wyrzucamy - nie ma downstream consumera, więc to nie jest job na DFB.
- Zostawienie `c1` tylko jako PRODUCER nie przejdzie census: każdy DFB na nodzie musi mieć dokładnie jednego producera i jednego consumera.
- Dwa kernele (jeden wypełnia DFB, drugi czyta) rozdzieliłyby role, ale to dwa JIT-y na dwa tile'e indeksów.
- `LocalTensorAccessor` / `CoreLocalMem` z issue też by zadziałało. Scratchpad jest tym, co host już deklaruje w ProgramSpec.
- Wyłączenie walidacji nic nie daje: backend DFB i tak padnie na nakładających się `producer_risc_mask` / `consumer_risc_mask`.
- I2S z issue (ten sam leftover) na tym drzewie już jest scratchpadem. Nie ruszamy.

### Rozwiązanie

- Factory: wyrzucony `DataflowBufferSpec` `c1` i oba bindingi PRODUCER+CONSUMER. Zamiast tego `ScratchpadSpec` (`size_per_node` = jeden tile) i `ScratchpadBinding` `cb_tensor`.
- Kernel: bez `reserve_back` / `push_back` / `wait_front` / `pop_front` na indeksach. `noc.async_read` + barrier, potem `cb_tensor[i]`.
- `c0` dalej zwykły DFB między readerem a writerem. To nie jest self-loop.
- Draft PR-a „only the host binding changes” jest zły: kernel też przestaje udawać FIFO.

### Wynik

- Repro: `test_quasar_slice_tile_tensor_args` — `(1,1,64,64)` TILE bf16, start `[0,0,32,0]`. Start niezerowy, żeby zły `start_offset` nie przeszedł jako „pierwsze 32 wiersze”.
- Kernel wciąż czyta `end` do `end_indices` i tego nie używa (offset tylko ze `start`). Warning przy JIT, nie ten ticket.

### Pytania

**Co usuwamy i dlaczego?**  
Nie usuwamy stage’owania indeksów. Kernel nadal ściąga `start`, potem `end`, liczy `start_offset`, kopiuje payload przez `c0`. `c0` zostaje.

Wylatuje **udawanie FIFO** wokół kawałka L1, którego FIFO nie potrzebuje:

- host: `c1` jako DFB i dwa bindingi na readerze — PRODUCER **i** CONSUMER tego samego `c1` (self-loop)
- kernel: `reserve_back` / `push_back` / `wait_front` / `pop_front` oraz `get_read_ptr()` + ręczny pointer

To nie była kolejka między dwoma kernelami. Jeden RISC sam rezerwował slot, pisał NOC-em, udawał „push”, sam czekał, czytał, sam zdejmował. Żeby DFB to przyjął, host musiał związać obie strony. Na Quasarze ProgramSpec tego nie złoży.

**Co to scratchpad?**  
Kawałek L1 na czas programu. Host: `ScratchpadSpec` z `size_per_node` = jeden tile. Kernel: `Scratchpad<uint32_t> cb_tensor(...)`, indeksy przez `cb_tensor[i]`.

Bez kredytów, bez PRODUCER/CONSUMER, bez `push`/`pop`. Surowa pamięć. NOC może w nią pisać; po `async_read_barrier()` wolno czytać.

**Dlaczego DFB → scratchpad?**  
DFB jest od przekazywania tile’y między rolami. Payload (`c0`) tak działa: reader wypełnia, writer opróżnia — zostawiamy.

Indeksy to notatnik na starcie: dwa małe tensory, jeden kernel, zero downstream. DFB wymaga pary ról, więc scratchpad stawał się nielegalnym self-loopem. Scratchpad jest na working memory. Sam `TT_FATAL` mówi: *Consider using a scratchpad.*

Czytanie zostaje (`noc.async_read` + barrier). Zmieniamy typ miejsca, nie algorytm slice’a.

</details>