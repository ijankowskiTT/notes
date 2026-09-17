# Notatki o defektach Quasar - #51271 / #56226


<details>
<summary><h2>Item 1 - matmul fused bias, hang na circular bufferze</h2></summary>

<details>
<summary><h2>Follow-up po item 1 — fused bias z złego banku DRAM (interleaved)</h2></summary>
- znalezione na review #56597 (Copilot), nie jest w #51271 ani #51222
- twin: ten sam `bank_id` na mainline (height + non-batched DRAM-sharded). Regular matmul już czyta bias przez `TensorAccessor`
- **nie** wrzucać w PR hangu. Osobny issue, potem osobny PR
### O co chodzi
- Worker DRAM-sharded matmula ma `dram_bank_id` = bank **wag** in1 (`HEIGHT_SHARDED` w DRAM). Każdy worker czyta swój shard in1 z tego banku — to jest OK.
- Fused bias jest **interleaved** DRAM (`DRAM_MEMORY_CONFIG`). Tile'e są pocięte na strony po bankach, nie skopiowane pod ten sam adres w każdym banku.
- Kernel biasu robi `{.bank_id = dram_bank_id, .addr = in3_tensor_addr}` — ten sam offset co in1, ale w **banku workera**. To nie jest mapa stron biasu.
- 1 tile biasu (`N=32`): cała strona 0 siedzi w banku 0. Worker 0 ma dobry bias. Worker 1..N czyta ten sam adres w swoim banku → śmieci w fused output dla tych batchy.
- Większe `N`: strony striped po bankach. Wtedy nawet worker 0 jest zły, jeśli czyta ciągły zakres bajtów z jednego banku zamiast `page_id`.
To **nie** jest regresja hoistu. Hoist tylko przeniósł ten sam read przed pętlę. Przed hoistem i tak był ten adres; hang ucinał program zanim dało się to zobaczyć na liczbach.
### Scope
Quasar:
- batched height: `reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` + factory `matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp` (dziś zero `TensorAccessorArgs` / `tensor::bias` na tym readerze)
- non-batched: `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` — ten sam `dram_bank_id` burst
Mainline: te same dwa kernele pod `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/`
Najmniejszy pierwszy PR: tylko quasar batched height + factory + pełny PCC w istniejącym teście. Reszta checkboxy na issue.
### Czemu nie było widać wcześniej
- **Hang pierwszy.** `batches_per_core > 1` → deadlock na drugim `reserve_back`. Drugi batch nigdy nie wracał, nie było pełnego PCC.
- **Publiczny `linear(..., bias=)` na WH** idzie w post-process `add()`, nie kompiluje `FUSE_BIAS` na tej factory. Testy publicznego API tego readu nie wołają.
- **Ticket #51271 / #51222** cytował `bank_id` + `in3_tensor_addr` tylko jako „ten sam bias, zero stride batch” (dowód na zbędny re-push). Nie audytowali mapy stron interleaved DRAM.
- Test regresji hangu **świa­domie** ucina PCC do `ref[:, :batches_per_core]` (bank 0), żeby udowodnić że call wraca, bez udawania że fused bias jest poprawny na wszystkich shardach. Komentarz w teście to właśnie ten bug.
- Copilot złapał to, bo slice + komentarz w teście robią defect oczywistym po zniknięciu hangu.
### Podejścia
- **Wybrane na follow-up:** `TensorAccessor` (albo `tensor::bias` jak metal2 padding reader) i pętla `{.page_id = t}`. Wszyscy workerzy czytają te same strony biasu, niezależnie od banku in1.
- Nie zostawiać `bank_id` workera i nie zakładać, że bias jest zreplikowany w każdym banku.
- Nie naprawiać tego w PR #56597 (inny kontrakt: CTA/binding + rebuild factory, nie sam hoist JIT).
### Rozwiązanie (gdy PR)
- Kernel: zamiast jednego `async_read` z `dram_bank_id`, pętla po `in3_block_tiles` przez accessor.
- Factory: `TensorAccessorArgs(*bias_tensor).append_to(...)` albo named `tensor::bias`. Uważać na offset CTA (dziś 10–12 pod FUSE_BIAS) i slot RTA (`in3` = arg 2, `dram_bank_id` = 3).
- Test: `assert_with_pcc(ref, got)` — bez slice do banku 0.
### Pytania / odpowiedzi
**Czemu worker 0 przechodzi PCC?**  
Przy 1-tile biasu strona 0 jest w banku 0. Worker 0 czyta właściwy bank przypadkiem. To nie znaczy, że addressing jest OK.
**Czemu in1 może używać `dram_bank_id`, a bias nie?**  
in1 jest HEIGHT_SHARDED w DRAM: shard workera leży w jego banku. Bias jest interleaved: kolejne page_id skaczą po bankach.
**To ten sam bug co hang?**  
Nie. Hang = za dużo `push` na CB `c_3`. Ten bug = skąd NOC bierze bajty biasu. Po hoiscie hang znika, liczby na shardach ≠ bank 0 zostają złe.
</details>

- branch: `51271-item1-quasar-matmul-bias-cb`, commit `040eee5ed9b`
- twin: #51222 item 1. Mainline `reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` nadal pcha bias w pętli batch.

- found bug: read from the wrong weight bank

### Problem

- Batched HEIGHT_SHARDED DRAM matmul, fused bias. CB `c_3` = jeden block. Hang gdy `batches_per_core > 1`.
- Reader pushował **ten sam** bias (addr bez offsetu) wewnątrz `for (batch …)`.
- Compute: `wait` raz, `pop` raz na końcu. Po batch 0 CB pełny → drugi `reserve_back` hang.

### Podejścia

- **Wybrane:** ten sam `reserve` / read / `push` przed pętlę. Jeden `push`, jeden `wait`, jeden `pop`.
- Nie rosnąć CB i nie pop co batch — compute ma trzymać bias do końca.
- Nie `TT_FATAL` na `batches_per_core > 1`.

### Rozwiązanie

- W `reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp`: `cb_in3.reserve_back` / `noc.async_read` / barrier / `push_back` przed `for (batch …)`. W pętli zostaje tylko in1 i zapis outputu.
- Factory bez zmian (rozmiar CB, `FUSE_BIAS`, `num_blocks_w_dim = 1`).

### Wynik

- Repro: `test_quasar_batched_dram_sharded_matmul_fused_bias_multi_batch` — descriptor + `generic_op`, `FUSE_BIAS=1`, `batches_per_core > 1`. Publiczny `linear` tego nie złapie.

### Pytania / odpowiedzi

**Czemu nie publiczny `linear(..., bias=)`?**  
Na WH batched in1 → post-process `add()`, nie fused kernel. Test woła factory bezpośrednio.

**Czemu nie rosnnie CB?**  
Compute i tak popuje raz na końcu. Większy CB bez zmiany compute nadal zostawia kredyty niezgodne, albo wymaga ruszania compute. Hoist pasuje do istniejącego kontraktu.
</details>


<details>
<summary><h2>Item 2 - refactor to FATAL on gather_in0 DRAM bank map miss</h2></summary>





</details>


<details>
<summary><h2>Item 3 - dest stride na unaligned same-width reshard</h2></summary>

Na tym samym shape jest drugi, niezależny bug: #56226.

### Problem

HEIGHT_SHARDED gather 4→2 cores, shape `(1,1,128,3)` bf16 row-major. Packed row 6 B, L1 slot 16 B. Kernel `+= 6` zamiast 16 → overlapping rows, silent corruption. Host liczył 16 B, do kernela szło tylko `remote_unit_size_padded`. Na Quasar NOC align = 1, ale `unaligned` liczy się z L1 16.

### Podejścia

**Wybrane:** CTA `local_unit_size_padded`, dest hop 16 B, NOC kopiuje 6 B.

Odrzucone: upchnąć rows bez padu; wysyłać 16 B przez NOC; skip UNALIGNED na Quasar; `local_start_offset` bez padded stride (dwa RISC-e by się nakładały).

### Rozwiązanie

Factory wstawia `local_unit_size_padded`. Dest skacze o 16 B, padu nie wysyłamy. `local_start_offset` też padded.

</details>


<details>
<summary><h2>#56226 - dwa RISC-e, jeden scratch</h2></summary>

### Problem

Oba DM RISC-e dostają ten sam reader. Unaligned path stage’uje do wspólnego scratch DFB. `src_offset` reset na każdym remote core → nadpisanie (clobber). Dest dostaje śmieci nawet przy hopie 16 B z item 3. Miss m.in. `[0,0,31,2]`. To nie quirk firmware WH.

### Podejścia

**Wybrane:** `num_entries * 2` + CTA `is_reader`. Nie drugi DFB.

Odrzucone: osobny DFB per RISC; jeden RISC na unaligned; serializacja na tym samym scratch; sam offset bez podwojenia `num_entries` (OOB).

### Rozwiązanie

Scratch `2 * remote_units_per_shard`. RISC z `is_reader` = 0 startuje od drugiej połowy.

Potrzebne też: `#define UNALIGNED` gdy packed != padded; writer hopa padded gdy local jest input; borrowed DFB — gdy packed < jeden padded row, `entry_size` = packed, `num_entries = 1`.

</details>


<details>
<summary><h2>Item 5 - tensor-args slice, self-loop DFB</h2></summary>

### Problem

Bug tylko gdy `start`/`end` to tensory na device (TILE). Listy w Pythonie nie składają `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp`.

Indeksy stage’owane jako DFB `c1` na tym samym DM readerze: PRODUCER+CONSUMER = self-loop. Gen2 ProgramSpec: `TT_FATAL` (*Consider using a scratchpad*). Na WH (Gen1) self-loop jest git. Ban ProgramSpec wszedł dwa dni po kernelu; tensor-args został przy `reserve → push → wait → pop`.

### Podejścia

**Wybrane:** `c1` = `ScratchpadSpec` / `Scratchpad<uint32_t>`. `c0` bez zmian. Jak I2S / padded_slice / reshape.

Odrzucone: `c1` tylko PRODUCER (census); dwa kernele na dwa tile’e indeksów; wyłączenie walidacji (backend i tak padnie). `LocalTensorAccessor` / `CoreLocalMem` zadziałałyby — scratchpad jest tym, co host już deklaruje. I2S z issue na tym drzewie już jest scratchpadem.

### Rozwiązanie

Wyrzucony DFB `c1` i bindingi PRODUCER+CONSUMER. Scratchpad jeden tile, `noc.async_read` + barrier, `cb_tensor[i]`. `c0` zostaje DFB reader→writer. Kernel też przestaje udawać FIFO, nie tylko host.

</details>


<details>
<summary><h2>Item 7 - tilize DRAM-sharded in, zero-copy factory</h2></summary>

### Problem

Sharded optimized tilize pożycza shard jako CB (`borrowed_from`). CB tylko L1. Guard sprawdzał DRAM tylko na **output**. DRAM-in + L1-out → throw. Case: RM `(1,1,2048,64)`, HEIGHT_SHARDED 4 cores, shard `(512,64)`, in DRAM, out L1. Mainline łapał DRAM out; dziura to DRAM in.

### Podejścia

**Wybrane:** guard in i out = L1, jak mainline. Stare per-layout checki na DRAM out zbędne.

Odrzucone: borrow z DRAM; kopiowanie DRAM→L1 w optimized factory (to job default); check tylko na input (zostawi DRAM-out dziurę).

### Rozwiązanie

`buffer_type() != L1` na in i out → `return false` → default factory. WIDTH TILE_HEIGHT check zostaje.

</details>


<details>
<summary><h2>Item 9 - untilize HS shard height, kilka macierzy na core</h2></summary>

### Problem

`to_layout(..., ROW_MAJOR)` na sharded TILE woła `untilize_with_unpadding`. Output shard height zawsze `round_up(div_up(fused_height, num_cores), tile_height)`. OK przy `batch == 1`. Przy `batch > 1` writer robi `out_shard_h / batch` — wysokość musi być `batch * logical_H`.

Shape `(32, 32, 17, 64)`, 64 cores, shard `(512, 64)`, `batch = 16`. Stary wzór: 288 zamiast 272. Writer: 18 wierszy zamiast 17. Call wraca, output zły.

### Podejścia

**Wybrane:** to samo `batch` co writer. `batch > 1` → `batch * output_shape[-2]`. `batch == 1` → stary round-up.

Odrzucone: zmieniać writer na `logical_H`; zawsze `batch * logical_H` (przy batch==1 round-up nadal potrzebny).

### Rozwiązanie

`batch` ze `shard_volume / (padded_H * padded_W)`, potem if/else: przy `batch > 1` wysokość sharda = `batch * logical_H`.

</details>
