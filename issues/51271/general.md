# Notatki o defektach Quasar - #51271 / #56226

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
